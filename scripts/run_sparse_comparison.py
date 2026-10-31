#!/usr/bin/env python3
"""Prepare/run the project CSR algorithm through official LLVM sparse lowering.

Host preparation is partial (exit 2). GPU execution is a measured hybrid, with
all host work and domain transfers inside the same resident boundary as CUB.
"""
from __future__ import annotations
import argparse
import hashlib
import math
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.reproduce_gpu import (Blocked, Failed, SOURCE_ROOTS, bounded_run, digest, driver_preflight, read_json, write_json)
from scripts.sparse_comparison_support import (SOURCE_FILES, DOMAIN, CANDIDATE_COSTS, CUB_COSTS,
    case_identity, coordinates, make_case, semantic_cases, render_module, scope_decision, valid_timing)
from scripts.sparse_comparison_support import distribution_qualification

ENVIRONMENT_KEYS = frozenset(("LANG", "LC_ALL", "TMPDIR", "LD_LIBRARY_PATH", "PYTHONNOUSERSITE", "PYTHONPATH",
                             "CUDA_ROOT", "CUDA_HOME", "CUDA_PATH", "CUDA_VISIBLE_DEVICES", "PATH"))


def validate_tool_version(name, text):
    """Require a unique exact version record, not a substring in a banner."""
    if name in ("llvm-version", "translate-version", "clang-version"):
        prefix = "clang version" if name == "clang-version" else "LLVM version"
        suffix = r"(?:[ \t]+\([^\n)]*\))?" if name == "clang-version" else ""
        prefixes = re.findall(r"(?m)^[ \t]*" + re.escape(prefix) + r"[^\n]*$", text)
        candidates = re.findall(r"(?m)^[ \t]*" + re.escape(prefix) + r"[ \t]+(\S+)" + suffix + r"[ \t]*$", text)
        if len(prefixes) != 1 or candidates != ["23.1.3"]:
            raise Blocked(f"{name} must report one exact 23.1.3 version record")
        return {"version": "23.1.3"}
    if name in ("ptxas-version", "nvcc-version"):
        prefixes = re.findall(r"(?m)^\s*Cuda compilation tools[^\n]*$", text)
        candidates = re.findall(r"(?m)^\s*Cuda compilation tools, release ([^,\s]+), V([^\s]+)\s*$", text)
        if len(prefixes) != 1 or candidates != [("13.2", "13.2.86")]:
            raise Blocked("ptxas must report one exact release 13.2, V13.2.86 version record")
        return {"release": "13.2", "version": "13.2.86"}
    raise ValueError("unknown tool version record")


def validate_cub_build(library, cuda_root):
    library, cuda_root = Path(library).resolve(), Path(cuda_root).resolve()
    path = library.parent / "build.json"
    if not path.is_file():
        raise Blocked("CUB comparison requires the actual adjacent build.json provenance")
    manifest = read_json(path)
    command = manifest.get("command")
    source = ROOT / "baselines/cuda/baselines.cu"
    if (manifest.get("schema") != 1 or type(manifest.get("exit_code")) is not int or manifest["exit_code"] != 0
            or manifest.get("source_unchanged") is not True or manifest.get("cccl_expected") != "3.5.0"
            or manifest.get("source_sha256") != digest(source) or manifest.get("library_sha256") != digest(library)
            or not isinstance(command, list) or not command or not all(isinstance(item, str) for item in command)
            or Path(command[0]).resolve() != (cuda_root / "bin/nvcc").resolve()
            or command.count("-arch=sm_89") != 1 or str(source) not in command
            or command[-2:] != ["-o", str(library)]
            or any(item.startswith(("--gpu-architecture", "-gencode", "--generate-code")) or
                   (item.startswith("-arch") and item != "-arch=sm_89") for item in command)):
        raise Failed("CUB build manifest does not bind current source/library, CCCL 3.5 and sm89 producer")
    validate_tool_version("nvcc-version", manifest.get("nvcc_version", ""))
    return {**file_identity(path), "manifest": manifest}


def file_identity(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": digest(path)}


def source_identity():
    files = {name: digest(ROOT / name) for name in SOURCE_FILES}
    combined = hashlib.sha256("".join(f"{key}:{value}\n" for key, value in sorted(files.items())).encode()).hexdigest()
    return {"files": files, "sha256": combined}


def child_environment(cuda_root, driver=None):
    # Keep inherited HOME unchanged, never repurpose it. Python overrides and
    # CUDA selectors are explicit; no inherited PYTHONPATH or user site imports.
    env = {key: os.environ[key] for key in ("LANG", "LC_ALL", "TMPDIR", "LD_LIBRARY_PATH") if key in os.environ}
    env.update(PYTHONNOUSERSITE="1", PYTHONPATH=str(ROOT), CUDA_ROOT=str(cuda_root), CUDA_HOME=str(cuda_root),
               CUDA_PATH=str(cuda_root), PATH=str(cuda_root / "bin") + os.pathsep + os.defpath)
    if driver and driver.get("status") == "ready":
        env["CUDA_VISIBLE_DEVICES"] = driver["device"]["uuid"]
    return env


def validate_arguments(args):
    for key, lo, hi in (("segments", 1, 1_000_000), ("elements", 1, 2**31-1), ("seed", 0, 2**32-2),
                        ("warmup", 0, 1000), ("repeats", 1, 10000), ("max_host_bytes", 1, 2**63-1)):
        value = getattr(args, key)
        if type(value) is not int or not lo <= value <= hi:
            raise ValueError(f"{key} outside [{lo},{hi}]")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    estimate = 128 * args.elements + 128 * (args.segments + 1)
    if estimate > args.max_host_bytes:
        raise ValueError(f"conservative preparation size estimate {estimate} exceeds --max-host-bytes")
    original = args.output.absolute()
    output = original.resolve(strict=False)
    if original.exists() or original.is_symlink() or output.exists():
        raise ValueError("output must be absent; no overwrite or retry")
    if output == ROOT or ROOT.is_relative_to(output):
        raise ValueError("output cannot contain the source tree")
    for entry in SOURCE_ROOTS:
        protected = (ROOT / entry).resolve()
        if output == protected or output.is_relative_to(protected):
            raise ValueError("output cannot be inside source paths")
    return output, estimate


class Recorder:
    def __init__(self, output, report, env, timeout, runner):
        self.output, self.report, self.env, self.timeout, self.runner = output, report, env, timeout, runner
        (output / "stages").mkdir()

    def run(self, name, argv, *, allowed=(0,)):
        argv = [str(item) for item in argv]
        stdout, stderr = self.output / "stages" / (name + ".stdout"), self.output / "stages" / (name + ".stderr")
        record = {"name": name, "argv": argv, "cwd": str(ROOT),
                  "environment": {key: value for key, value in self.env.items() if key in ENVIRONMENT_KEYS}, "timeout_seconds": self.timeout,
                  "status": "failed", "exit_code": None}
        self.report["stages"].append(record)
        started = time.perf_counter_ns()
        try:
            with stdout.open("wb") as out, stderr.open("wb") as err:
                result = self.runner(argv, cwd=ROOT, env=self.env, stdout=out, stderr=err, timeout=self.timeout, check=False)
            record["exit_code"] = result.returncode
            if result.returncode not in allowed:
                raise Failed(f"{name} exited {result.returncode}; see {stderr}")
            # Upstream CUDA wrappers print errors and return without throwing.
            if name == "execute-gpu" and re.search(r"'[^'\n]+' failed with '[^'\n]+'", stderr.read_text(errors="replace")):
                raise Failed("upstream CUDA wrapper reported failure even if worker exit was zero")
            record["status"] = "passed" if result.returncode == 0 else "retained_failure"
        except Exception as error:
            record["exit_code"] = getattr(error, "returncode", record["exit_code"])
            record["error"] = {"type": type(error).__name__, "message": str(error)}
            raise
        finally:
            record["wall_ms"] = (time.perf_counter_ns()-started)/1e6
            record.update(stdout=file_identity(stdout), stderr=file_identity(stderr))
            write_json(self.output / "comparison.json", self.report)
        return stdout, stderr


def prepare_inputs(output, args):
    import numpy as np
    from bench.distributions import describe
    directory = output / "inputs"
    directory.mkdir()
    cases = {"long_tail": make_case(args.segments, args.elements, args.seed), **semantic_cases()}
    result = {}
    for name, (values, offsets, coords) in cases.items():
        path = directory / (name + ".npz")
        np.savez(path, values=values, offsets=offsets, coordinates=coords)
        result[name] = {**file_identity(path), "identity": case_identity(values, offsets, coords),
                        "max_length": int(np.diff(offsets).max(initial=0)), "length_statistics": describe(np.diff(offsets)),
                        "generation": {"distribution": "long_tail", "seed": args.seed, "values_seed": args.seed+1,
                                       "normalization": "fixed_total_elements", "explicit_zero_rule": "values[::257]=+0.0"}
                        if name == "long_tail" else {"distribution": "explicit_semantic_case"}}
    return result


def extract_cubins(path, output):
    # gpu.binary prints arbitrary bytes as hexadecimal escapes. No implicit PTX
    # fallback is accepted; each object must actually be an ELF cubin.
    text = path.read_text()
    matches = re.findall(r'bin\s*=\s*"((?:\\[0-9A-Fa-f]{2}|[^"\\])+)"', text)
    if not matches:
        raise Failed("official lowering emitted no serialized GPU binary")
    result = []
    for number, escaped in enumerate(matches):
        data = bytearray()
        index = 0
        while index < len(escaped):
            if escaped[index] == "\\":
                data.append(int(escaped[index+1:index+3], 16))
                index += 3
            else:
                data.extend(escaped[index].encode("ascii"))
                index += 1
        if not data.startswith(b"\x7fELF"):
            raise Failed("serialized GPU object is not an ELF cubin")
        artifact = output / f"program-{number}.cubin"
        artifact.write_bytes(data)
        result.append(file_identity(artifact))
    return result


def build_module(name, b, width, output, args, recorder):
    directory = output / "modules" / name
    directory.mkdir(parents=True)
    source = directory / "input.mlir"
    source.write_text(render_module(b, width))
    result = {"B": b, "width": width, "source": file_identity(source), "shape_specialization": "sparse_tensor.assemble requires static B and maximum row width",
              "ownership": "enable-runtime-library=false; assembled input buffers borrowed; owned sparse/dense results deallocated",
              "implementation": "project CSR algorithm using public sparse_tensor unary/reduce and official LLVM 23.1.3 sparsifier",
              "gpu_kind": "hybrid: upstream reductions on GPU, exp/normalization and packed copies on host",
              "expected_reduction_calls_per_op": {"max": 1, "softmax": 2}, "launch_count_scope": "compiled call paths, not a measured profiler launch count"}
    opt, translate = args.llvm_root / "bin/mlir-opt", args.llvm_root / "bin/mlir-translate"
    for mode in ("cpu", "gpu"):
        lowered, llvm, library = (directory / f"{mode}.{suffix}" for suffix in ("llvm.mlir", "ll", "so"))
        options = "enable-runtime-library=false enable-buffer-initialization=true"
        if mode == "gpu":
            options += " parallelization-strategy=dense-outer-loop gpu-triple=nvptx64-nvidia-cuda gpu-chip=sm_89 gpu-features=+ptx80 gpu-format=bin"
        _, log = recorder.run(name + "-" + mode + "-sparsifier", [opt, source, "--sparsifier=" + options,
                                               "--mlir-print-ir-after=sparse-gpu-codegen", "-o", lowered])
        recorder.run(name + "-" + mode + "-translate", [translate, lowered, "--mlir-to-llvmir", "-o", llvm])
        command = [args.clang, "-shared", "-fPIC", "-O2", llvm, "-Wl,-z,defs", "-lm"]
        if mode == "gpu":
            command += [args.gpu_runtime, "-Wl,-rpath," + str(args.gpu_runtime.parent)]
        recorder.run(name + "-" + mode + "-link", command + ["-o", library])
        result[mode + "_library"] = file_identity(library)
        result[mode + "_llvm_ir"] = file_identity(llvm)
        result[mode + "_lowered_mlir"] = file_identity(lowered)
        if mode == "gpu":
            result["cubins"] = extract_cubins(lowered, directory)
            for index, cubin in enumerate(result["cubins"]):
                recorder.run(name + f"-cubin-{index}-resources", [args.cuda_root / "bin/cuobjdump", "--dump-resource-usage", cubin["path"]])
            intermediate = log.read_text(errors="replace")
            entries = sorted(set(re.findall(r"gpu\.func\s+@([\w.$]+)", intermediate)))
            if len(entries) != 2:
                raise Failed(f"expected max/sum reduction entries, observed {entries}")
            result["embedded_gpu_entries"] = entries
            # These operations still exist in the translated host IR. Save the
            # actual IR, not a claim that softmax has become a pure GPU program.
            host_ir = llvm.read_text()
            if "llvm.exp.f32" not in host_ir or "fdiv float" not in host_ir:
                raise Failed("expected hybrid host exp/div path disappeared; review required")
    return result


def load_saved_case(record):
    import numpy as np
    if file_identity(record["path"]) != {key: record[key] for key in ("path", "sha256")}:
        raise Failed("input archive changed")
    with np.load(record["path"], allow_pickle=False) as data:
        values, offsets, coords = (data[key] for key in ("values", "offsets", "coordinates"))
    if values.dtype != np.float32 or offsets.dtype != np.int64 or coords.dtype != np.int64:
        raise Failed("saved input dtypes changed")
    if (values.shape != (record["identity"]["N"], 1) or offsets.shape != (record["identity"]["B"]+1,)
            or coords.shape != (record["identity"]["N"],) or any(not item.flags.c_contiguous for item in (values, offsets, coords))):
        raise Failed("saved packed input shapes/layout differ from the D1 ABI")
    np.testing.assert_array_equal(coords, coordinates(offsets))
    if case_identity(values, offsets, coords) != record["identity"]:
        raise Failed("saved input identities differ")
    return values, offsets, coords


def validate_execution(path, config, mode):
    """Bind actual raw outputs, AOT modules and complete-call records independently."""
    import numpy as np
    from oracle import assert_matches
    report = read_json(path)
    if (report.get("schema") != 1 or report.get("mode") != mode or report.get("source") != config["source"]
            or report.get("inputs") != config["inputs"] or source_identity() != config["source"]
            or report.get("actual_device_execution") is not (mode == "gpu")
            or set(report.get("cases", {})) != set(config["inputs"])):
        raise Failed("worker source/input/execution identity mismatch")
    expected_backend = "upstream-sparse-gpu-hybrid" if mode == "gpu" else "upstream-sparse-cpu-AOT"
    if report.get("backend") != expected_backend or set(report.get("modules", {})) != set(config["modules"]):
        raise Failed("worker backend/module inventory mismatch")
    environment = report.get("host_environment", {})
    if (environment.get("python") != file_identity(sys.executable) or environment.get("python_version") != sys.version
            or environment.get("numpy_version") != np.__version__ or environment.get("numpy_module") != file_identity(np.__file__)):
        raise Failed("actual worker Python/NumPy environment differs from parent qualification")
    for name, record in report["modules"].items():
        expected = config["modules"][name][mode + "_library"]
        if {key: record.get(key) for key in ("path", "sha256")} != expected or file_identity(expected["path"]) != expected:
            raise Failed("loaded AOT module differs from compiled artifact")
        callbacks = record.get("callbacks", {})
        if set(callbacks) != {"_mlir_ciface_run_max", "_mlir_ciface_run_softmax"}:
            raise Failed("actual loaded AOT callback inventory is missing")
        for symbol, callback in callbacks.items():
            if ({key: callback.get(key) for key in ("path", "sha256")} != expected or callback.get("symbol") != symbol
                    or type(callback.get("address")) is not int or callback["address"] <= 0):
                raise Failed("actual loaded callback does not belong to the linked AOT artifact")
        if mode == "gpu":
            module = config["modules"][name]
            if module["gpu_library"] == module["cpu_library"]:
                raise Failed("CPU library cannot stand in for actual GPU candidate")
            binary = Path(expected["path"]).read_bytes()
            if b"mgpuLaunchKernel\x00" not in binary or not module.get("cubins"):
                raise Failed("linked GPU AOT object lacks actual upstream launch imports/cubins")
            for cubin in module["cubins"]:
                if file_identity(cubin["path"]) != cubin or Path(cubin["path"]).read_bytes() not in binary:
                    raise Failed("linked GPU shared object does not embed the compiled cubin")
            for kind in ("source", "gpu_llvm_ir", "gpu_lowered_mlir"):
                if file_identity(module[kind]["path"]) != module[kind]:
                    raise Failed("GPU compilation provenance artifact changed")
            runtime = config["gpu_runtime"]
            callback = record.get("cuda_runtime_callback", {})
            if ({key: callback.get(key) for key in ("path", "sha256")} != runtime
                    or file_identity(runtime["path"]) != runtime or callback.get("symbol") != "mgpuLaunchKernel"
                    or type(callback.get("address")) is not int or callback["address"] <= 0):
                raise Failed("loaded upstream CUDA runtime differs from the linked runtime")
        if type(record.get("load_wall_ms")) not in (int, float) or not math.isfinite(record["load_wall_ms"]) or record["load_wall_ms"] < 0:
            raise Failed("missing actual persistent module load measurement")
    if mode == "gpu":
        if (report.get("driver", {}).get("status") != "ready" or report["driver"].get("device") != config["driver"]["device"]
                or report.get("domain") != DOMAIN or report.get("all_complete_costs_included") is not True
                or report.get("warmup") != config["warmup"] or report.get("repeats") != config["repeats"]
                or report.get("input") != config["inputs"]["long_tail"]["identity"] or report.get("cpu_oracle") != config["cpu_oracle"]
                or report.get("cub_library") != config["cub_library"] or file_identity(config["cub_library"]["path"]) != config["cub_library"]
                or report.get("cub_provenance", {}).get("library_sha256") != config["cub_library"]["sha256"]
                or report.get("cub_build") != config["cub_build"]
                or report.get("cub_provenance", {}).get("build") != config["cub_build"]["manifest"]
                or file_identity(config["cub_build"]["path"]) != {key: config["cub_build"][key] for key in ("path", "sha256")}):
            raise Failed("GPU worker Driver/domain/native/measurement contract differs")
        framework = report.get("framework", {})
        if (framework.get("version") != "2.14.1+cu132" or framework.get("cuda") != "13.2"
                or file_identity(framework.get("module", {}).get("path", "")) != framework.get("module")):
            raise Failed("GPU framework identity is absent or incompatible")
        callbacks = report.get("cub_callbacks", {})
        if set(callbacks) != {"ragged_cub_reduce", "ragged_cub_transform"}:
            raise Failed("actual CUB callback identities are absent")
        for symbol, callback in callbacks.items():
            if ({key: callback.get(key) for key in ("path", "sha256")} != config["cub_library"]
                    or callback.get("symbol") != symbol or type(callback.get("address")) is not int or callback["address"] <= 0):
                raise Failed("CUB callback is not from the measured native artifact")
    qualified = {"max": True, "softmax": True}
    semantic = {"max": True, "softmax": True}
    failed = False
    for name, record in config["inputs"].items():
        values, offsets, coords = load_saved_case(record)
        if set(report["cases"][name]) != {"max", "softmax"}:
            raise Failed("worker omitted an operation")
        for op, methods in report["cases"][name].items():
            if set(methods) != ({"candidate", "cub"} if mode == "gpu" else {"candidate"}):
                raise Failed("worker omitted a comparison method")
            for method, cell in methods.items():
                if cell.get("input") != record["identity"]:
                    raise Failed("method does not use the exact same saved input")
                observed_oracle = False
                if "output" in cell:
                    item = cell["output"]
                    expected_path = Path(path).parent / f"{name}-{op}-{method}.npy"
                    if item.get("path") != str(expected_path.resolve()) or file_identity(expected_path) != item:
                        raise Failed("raw output artifact missing or changed")
                    actual = np.load(expected_path, allow_pickle=False)
                    if actual.dtype != np.float32:
                        raise Failed("candidate output changed source dtype")
                    try:
                        assert_matches(actual, values, offsets, operation=op, input_dtype="f32")
                        observed_oracle = True
                    except AssertionError:
                        pass
                if cell.get("oracle_passed") is not observed_oracle:
                    raise Failed("worker oracle claim differs from saved raw output")
                if observed_oracle and method == "candidate" and op == "softmax":
                    expected_structure = {"positions_sha256": record["identity"]["offsets_sha256"],
                        "coordinates_sha256": record["identity"]["coordinates_sha256"], "stored_entry_count": len(values),
                        "explicit_stored_zero_preserved": True}
                    if cell.get("structure") != expected_structure:
                        raise Failed("sparse structure did not preserve all original packed entries")
                status = cell.get("status")
                if status not in ("failed", "qualified", "measured") or (status != "failed" and not observed_oracle):
                    raise Failed("method status contradicts raw-output qualification")
                if status == "failed":
                    failed = True
                if not observed_oracle or status == "failed":
                    qualified[op] = False
                    if name != "long_tail":
                        semantic[op] = False
                if "timing" in cell:
                    expected_costs = list(CANDIDATE_COSTS if method == "candidate" else CUB_COSTS)
                    if (mode != "gpu" or name != "long_tail" or status != "measured" or config["cpu_oracle"][op] is not True
                            or cell.get("domain") != DOMAIN or cell.get("costs") != expected_costs
                            or not valid_timing(cell["timing"], config["warmup"], config["repeats"])):
                        raise Failed("timing lacks full boundary, sample count or oracle qualification")
                elif status == "measured":
                    raise Failed("measured status has no actual samples")
    if report.get("status") != ("failed" if failed else "complete"):
        raise Failed("overall worker status contradicts per-operation evidence")
    if mode == "gpu":
        if report.get("operations") != report["cases"]["long_tail"]:
            raise Failed("timed comparison differs from qualified long-tail cells")
        report["parent_artifact_validation"] = True
        report["gpu_semantic_oracle"] = semantic
        _, offsets, _ = load_saved_case(config["inputs"]["long_tail"])
        report["distribution"] = distribution_qualification(offsets)
        report["distribution_qualified"] = report["distribution"]["qualified"]
    return qualified, report


def run(args, *, runner=bounded_run, preflight=driver_preflight):
    output, estimate = validate_arguments(args)
    output.mkdir(parents=True)
    report = {"schema": 1, "status": "failed", "mode": args.mode, "stages": [], "scope_decision": scope_decision({}),
              "parameters": {"B": args.segments, "N": args.elements, "D": 1, "dtype": "f32", "offsets_dtype": "i64",
                  "seed": args.seed, "warmup": args.warmup, "repeats": args.repeats},
              "preparation_memory_bound_bytes": estimate,
              "candidate": "project CSR algorithm, official sparse lowering; GPU hybrid with host exp/normalization, allocations and transfers",
              "no_total_gpu_memory_claim": "upstream mgpu allocations are outside PyTorch allocator accounting; device peak is unmeasured"}
    code = 1
    try:
        if args.mode == "gpu":
            report["driver"] = preflight(device=0)
            if report["driver"].get("status") != "ready":
                raise Blocked(report["driver"].get("reason", "CUDA Driver device unavailable"))
        args.llvm_root, args.cuda_root = args.llvm_root.resolve(), args.cuda_root.resolve()
        args.clang = (args.clang or args.llvm_root / "bin/clang").resolve()
        args.gpu_runtime = args.gpu_runtime.resolve()
        args.cub_library = args.cub_library.resolve()
        required = {"mlir_opt": args.llvm_root / "bin/mlir-opt", "mlir_translate": args.llvm_root / "bin/mlir-translate",
                    "clang": args.clang, "ptxas": args.cuda_root / "bin/ptxas", "cuobjdump": args.cuda_root / "bin/cuobjdump",
                    "gpu_runtime": args.gpu_runtime, "libdevice": args.cuda_root / "nvvm/libdevice/libdevice.10.bc"}
        if args.mode == "gpu":
            required["cub_library"] = args.cub_library
            required["nvcc"] = args.cuda_root / "bin/nvcc"
        for name, path in required.items():
            if not path.is_file():
                raise Blocked(f"required provisioned {name} missing: {path}")
        report["source"] = source_identity()
        report["tools"] = {name: file_identity(path) for name, path in required.items()}
        env = child_environment(args.cuda_root, report.get("driver"))
        recorder = Recorder(output, report, env, args.timeout, runner)
        report["versions"] = {}
        probes = [("llvm-version", [required["mlir_opt"], "--version"]),
                 ("translate-version", [required["mlir_translate"], "--version"]),
                 ("clang-version", [args.clang, "--version"]), ("ptxas-version", [required["ptxas"], "--version"])]
        if args.mode == "gpu":
            probes.append(("nvcc-version", [required["nvcc"], "--version"]))
            report["cub_build"] = validate_cub_build(args.cub_library, args.cuda_root)
        for name, command in probes:
            stdout, stderr = recorder.run(name, command)
            report["versions"][name] = validate_tool_version(name, stdout.read_text() + stderr.read_text())
        version_file = args.cuda_root / "version.json"
        if read_json(version_file).get("cuda", {}).get("version") != "13.2.2":
            raise Blocked("CUDA toolkit must be 13.2.2")
        report["tools"]["cuda_version_json"] = file_identity(version_file)
        inputs = prepare_inputs(output, args)
        report["inputs"] = inputs
        _, offsets, _ = load_saved_case(inputs["long_tail"])
        report["distribution"] = distribution_qualification(offsets)
        modules = {"long_tail": build_module("long_tail", args.segments, max(1, inputs["long_tail"]["max_length"]), output, args, recorder),
                   "semantics": build_module("semantics", 6, 4, output, args, recorder)}
        report["modules"] = modules
        config = {"schema": 1, "source": report["source"], "inputs": inputs, "modules": modules,
                  "warmup": args.warmup, "repeats": args.repeats, "gpu_runtime": report["tools"]["gpu_runtime"]}
        config_path = output / "config.json"
        write_json(config_path, config)
        recorder.run("execute-cpu", [sys.executable, ROOT / "scripts/sparse_comparison_worker.py", "--config", config_path,
                                     "--mode", "cpu", "--output", output / "execution-cpu"], allowed=(0, 1))
        cpu_qualified, cpu = validate_execution(output / "execution-cpu/execution.json", config, "cpu")
        report["cpu_execution"] = file_identity(output / "execution-cpu/execution.json")
        report["cpu_oracle"] = cpu_qualified
        report["scope_decision"] = scope_decision({"cpu_oracle": cpu_qualified,
            "distribution_qualified": report["distribution"]["qualified"]})
        if args.mode == "host":
            report["status"] = "partial" if all(cpu_qualified.values()) else "failed"
            report["pending"] = ["actual GPU output qualification and same-domain complete-call timings", "1.5x sparse-vs-CUB scope decision"]
            code = 2 if report["status"] == "partial" else 1
        else:
            config.update(cpu_oracle=cpu_qualified, driver=report["driver"], cub_library=file_identity(args.cub_library),
                          cub_build=report["cub_build"])
            write_json(config_path, config)
            recorder.run("execute-gpu", [sys.executable, ROOT / "scripts/sparse_comparison_worker.py", "--config", config_path,
                                         "--mode", "gpu", "--output", output / "execution-gpu"], allowed=(0, 1))
            _, gpu = validate_execution(output / "execution-gpu/execution.json", config, "gpu")
            report["gpu_execution"] = file_identity(output / "execution-gpu/execution.json")
            report["scope_decision"] = scope_decision(gpu)
            if gpu["status"] == "failed" or not all(cpu_qualified.values()):
                report["status"], code = "failed", 1
            elif report["scope_decision"]["status"] == "eligible":
                report["status"], code = "complete", 0
            else:
                report["status"], code = "partial", 2
        if source_identity() != report["source"] or any(file_identity(item["path"]) != item for item in report["tools"].values()):
            raise Failed("producer source/tool identity changed during execution")
        report["configuration"] = file_identity(config_path)
    except Blocked as error:
        report.update(status="blocked", error={"type": type(error).__name__, "message": str(error)})
        code = 2
    except Exception as error:
        report.update(status="failed", error={"type": type(error).__name__, "message": str(error)}, scope_decision=scope_decision({}))
        code = 1
    report["exit_code"] = code
    write_json(output / "comparison.json", report)
    return code


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output", required=True, type=Path)
    result.add_argument("--mode", choices=("host", "gpu"), default="host")
    result.add_argument("--llvm-root", type=Path, default=ROOT / ".deps/LLVM-23.1.3-Linux-X64")
    result.add_argument("--clang", type=Path)
    result.add_argument("--cuda-root", type=Path, default=Path("/usr/local/cuda-13.2"))
    result.add_argument("--gpu-runtime", type=Path, default=ROOT / ".deps/llvm-build/lib/libmlir_cuda_runtime.so.23.1")
    result.add_argument("--cub-library", type=Path, default=ROOT / "build-baselines/libragged_baselines.so")
    result.add_argument("--segments", type=int, default=16384)
    result.add_argument("--elements", type=int, default=1048576)
    result.add_argument("--seed", type=int, default=711)
    result.add_argument("--warmup", type=int, default=3)
    result.add_argument("--repeats", type=int, default=10)
    result.add_argument("--timeout", type=float, default=600)
    result.add_argument("--max-host-bytes", type=int, default=1 << 30)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        return run(args)
    except (ValueError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
