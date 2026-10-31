#!/usr/bin/env python3
"""Fresh GPU reproduction using provisioned SDKs; blocked=2, failed=1, passed=0.

The first probe uses only stdlib ctypes and never imports the invoking Python's
Torch. No child, installation, download or snapshot starts before Driver readiness.
The reproduction is bounded: GPU correctness and a few same-input comparisons,
not the full benchmark matrix.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import shutil
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
import zipfile

try:
    from scripts.gpu_validation_scope import (MARKER, checked_nodeids, collected_nodeids,
                                              partition_sanitizer_nodeids, validate_suppression_file)
except ModuleNotFoundError as error:
    if error.name != "scripts":
        raise
    from gpu_validation_scope import (MARKER, checked_nodeids, collected_nodeids,
                                      partition_sanitizer_nodeids, validate_suppression_file)

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOTS = ("CMakeLists.txt", "README.md", "environment.lock", "pyproject.toml",
                "cmake", "include", "lib", "tools", "runtime", "python", "scripts", "test",
                "oracle", "baselines", "bench", "docs")
FOCUSED_TESTS = (
    "test/gpu/test_correctness.py",
    "test/gpu/test_classified.py::test_debug_visits_and_worklist_membership_exactly_once",
    "test/gpu/test_classified.py::test_static_split_accumulation_and_partial_tail",
    "test/gpu/test_checked.py::test_checked_invalid_inplace_softmax_preserves_all_values",
    "test/gpu/test_checked.py::test_checked_success_uses_the_same_numerical_oracle",
    "test/gpu/test_mask.py::test_scale_and_mask_keep_separate_f32_rounding",
    "test/gpu/test_optimizations.py::test_all_optimizations_off_and_long_accumulation",
    "test/gpu/test_optimizations.py::test_online_statistics_small_mass_accumulation_meets_fixed_tolerance",
)
COMMON_METHODS = ("ragged-naive", "ragged-auto", "njt-eager", "njt-compile", "cuda-naive", "cub", "triton")


class Blocked(RuntimeError):
    pass


class Failed(RuntimeError):
    pass


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def read_json(path):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise Failed(f"duplicate JSON field in {path}: {key}")
            value[key] = item
        return value
    def finite(text):
        value = float(text)
        if not math.isfinite(value):
            raise Failed(f"nonfinite JSON number in {path}")
        return value
    def invalid(text):
        raise Failed(f"invalid JSON constant {text} in {path}")
    return json.loads(Path(path).read_text(), object_pairs_hook=unique, parse_float=finite, parse_constant=invalid)


def bounded_run(argv, *, cwd, env, stdout, stderr, check=False, timeout):
    """Bound children and keep managed GPU descendants in the wrapper's group.

    Standalone invocations own a child group and forward TERM/HUP cleanup.
    Managed invocations must be inside gpu-slot.py, which owns final group
    cleanup; they never create a session that could escape that cleanup.
    """
    managed = env.get("RAGGED_GPU_MANAGED") == "1"
    with subprocess.Popen(argv, cwd=cwd, env=env, stdout=stdout, stderr=stderr,
                          start_new_session=not managed) as process:
        previous = {}

        def interrupted(signum, frame):
            raise SystemExit(128 + signum)

        try:
            if threading.current_thread() is threading.main_thread():
                for signum in (signal.SIGTERM, signal.SIGHUP):
                    previous[signum] = signal.signal(signum, interrupted)
            code = process.wait(timeout=timeout)
        except BaseException as error:
            try:
                if managed:
                    process.kill()
                else:
                    os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            error.returncode = process.returncode
            raise
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)
        if check and code:
            raise subprocess.CalledProcessError(code, argv)
        return subprocess.CompletedProcess(argv, code)


def driver_preflight(*, device=0):
    """Query the Driver directly, with no subprocess or third-party imports."""
    result = {"status": "blocked", "probe": "stdlib ctypes CUDA Driver API", "device_ordinal": device}
    try:
        driver = ctypes.CDLL("libcuda.so.1")
        def query(name, argument_types, *arguments):
            function = getattr(driver, name)
            function.argtypes, function.restype = argument_types, ctypes.c_int
            code = function(*arguments)
            result[name + "_status"] = code
            if code:
                raise Blocked(f"{name} returned CUDA status {code}")
        query("cuInit", [ctypes.c_uint], 0)
        count, version, ordinal = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
        pointer = ctypes.POINTER(ctypes.c_int)
        query("cuDeviceGetCount", [pointer], ctypes.byref(count))
        if not 0 <= device < count.value:
            raise Blocked(f"requested device {device}, available count {count.value}")
        query("cuDeviceGet", [pointer, ctypes.c_int], ctypes.byref(ordinal), device)
        query("cuDriverGetVersion", [pointer], ctypes.byref(version))
        result["driver_supported_cuda_version"] = version.value
        if version.value < 13020:
            raise Blocked(f"Driver supports CUDA {version.value}; this reproduction requires at least 13020")
        major, minor = ctypes.c_int(), ctypes.c_int()
        query("cuDeviceGetAttribute", [pointer, ctypes.c_int, ctypes.c_int], ctypes.byref(major), 75, ordinal.value)
        query("cuDeviceGetAttribute", [pointer, ctypes.c_int, ctypes.c_int], ctypes.byref(minor), 76, ordinal.value)
        if (major.value, minor.value) != (8, 9):
            raise Blocked(f"requires sm_89, observed sm_{major.value}{minor.value}")
        name, uuid = ctypes.create_string_buffer(256), (ctypes.c_ubyte * 16)()
        query("cuDeviceGetName", [ctypes.c_void_p, ctypes.c_int, ctypes.c_int], name, len(name), ordinal.value)
        query("cuDeviceGetUuid_v2", [ctypes.c_void_p, ctypes.c_int], ctypes.byref(uuid), ordinal.value)
        raw = bytes(uuid).hex()
        result.update(status="ready", device_count=count.value, driver_supported_cuda_version=version.value,
                      device={"name": name.value.decode(errors="replace"), "capability": [8, 9],
                              "uuid": "GPU-" + "-".join((raw[:8], raw[8:12], raw[12:16], raw[16:20], raw[20:]))})
    except (OSError, AttributeError, Blocked) as error:
        result["reason"] = str(error)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="must not already exist")
    parser.add_argument("--llvm-prefix", type=Path, required=True, help="provisioned LLVM/MLIR 23.1.3 SDK")
    parser.add_argument("--cuda-root", type=Path, default=Path("/usr/local/cuda-13.2"))
    parser.add_argument("--cccl-root", type=Path, default=ROOT / ".deps/cccl-3.5.0")
    parser.add_argument("--llvm-source-archive", type=Path)
    parser.add_argument("--cmake-prefix", type=Path, action="append", default=[])
    parser.add_argument("--python", default=getattr(sys, "_base_executable", sys.executable), help="Python 3.12 with stdlib venv")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--device", type=int, default=0, help="Driver ordinal, pinned by UUID for all children")
    parser.add_argument("--sanitizers", action="store_true", help="also run the full nonoptional GPU suite under three sanitizer tools")
    parser.add_argument("--stage-timeout-seconds", type=int, default=7200)
    args = parser.parse_args(argv)
    if args.jobs < 1 or args.device < 0 or args.stage_timeout_seconds < 1:
        parser.error("jobs/timeout must be positive and device nonnegative")
    return args


def prerequisites(args):
    """Only called after Driver readiness; no automatic provisioning of SDKs."""
    llvm, cuda, cccl = args.llvm_prefix.resolve(), args.cuda_root.resolve(), args.cccl_root.resolve()
    cccl_version = cccl / "libcudacxx/include/cuda/std/__cccl/version.h"
    required = [llvm / "lib/cmake/mlir/MLIRConfig.cmake", llvm / "lib/cmake/llvm/LLVMConfig.cmake",
                cuda / "version.json", cuda / "bin/nvcc", cuda / "bin/ptxas",
                cuda / "nvvm/libdevice/libdevice.10.bc", cccl / "cub/cub/version.cuh",
                cccl_version]
    llvm_config = llvm / "bin/llvm-config"
    if llvm_config.is_file():
        required.append(llvm_config)
    if args.sanitizers:
        required.append(cuda / "bin/compute-sanitizer")
    for path in required:
        if not path.is_file():
            raise Blocked(f"provisioned prerequisite is missing: {path}")
    if not re.search(r'set\(LLVM_PACKAGE_VERSION\s+"?23\.1\.3"?\s*\)', (llvm / "lib/cmake/llvm/LLVMConfig.cmake").read_text()):
        raise Blocked("the provisioned LLVMConfig.cmake does not identify version 23.1.3")
    if read_json(cuda / "version.json").get("cuda", {}).get("version") != "13.2.2":
        raise Blocked("CUDA Toolkit version.json must identify 13.2.2")
    if read_json(cuda / "version.json").get("cuda_nvcc", {}).get("version") != "13.2.86":
        raise Blocked("CUDA Toolkit must identify the pinned nvcc 13.2.86 component")
    if not re.search(r"#define\s+CCCL_VERSION\s+3005000\b", cccl_version.read_text()):
        raise Blocked("the provisioned CCCL headers must be 3.5.0")
    tools = {}
    for tool in ("cmake", "ninja", "cc", "c++"):
        resolved = shutil.which(tool, path="/usr/bin:/bin")
        if not resolved:
            raise Blocked(f"required system build executable is missing: {tool}")
        tools[tool] = str(Path(resolved).resolve())
    python = shutil.which(str(args.python))
    if not python:
        raise Blocked(f"base Python executable does not exist: {args.python}")
    tools["python"] = str(Path(python).resolve())
    archive = None
    if args.llvm_source_archive:
        archive = args.llvm_source_archive.resolve()
        if not archive.is_file() or digest(archive) != "c44186a7762ed28954be72e5ff6df9808e0779d4f1bf014ecc4e7e211d31ee34":
            raise Blocked("LLVM source archive is not the pinned official 23.1.3 archive")
        required.append(archive)
    revision = llvm / "include/llvm/Support/VCSRevision.h"
    revision_text = revision.read_text() if revision.is_file() else ""
    if "0d261d1ca552c95a8f007e061c787ac7132fbcbc" not in revision_text and archive is None:
        raise Blocked("SDK needs the pinned LLVM revision header or the verified official source archive")
    if revision.is_file():
        if re.search(r"^\s*#\s*define\s+LLVM_REVISION\b", revision_text, re.MULTILINE) and "0d261d1ca552c95a8f007e061c787ac7132fbcbc" not in revision_text:
            raise Blocked("SDK revision header differs from the pinned LLVM commit")
        required.append(revision)
    for prefix in args.cmake_prefix:
        if not prefix.resolve().is_dir():
            raise Blocked(f"CMake prerequisite prefix is missing: {prefix}")
    return {"llvm": str(llvm), "cuda": str(cuda), "cccl": str(cccl), "tools": tools,
            "llvm_config": str(llvm_config) if llvm_config.is_file() else None,
            "llvm_version_gate": "llvm-config, fresh CMake and compiler provenance" if llvm_config.is_file() else "fresh CMake exact-version check and fresh compiler exact-commit/version provenance",
            "archive": str(archive) if archive else None,
            "identity_files": [{"path": str(path), "sha256": digest(path)} for path in required],
            "build_executables": [{"path": path, "sha256": digest(path)} for path in sorted(set(tools.values()))],
            "cmake_prefixes": [str(path.resolve()) for path in args.cmake_prefix]}


def snapshot_source(source, destination):
    entries = []
    def copy(path):
        if path.is_symlink():
            raise Failed(f"source snapshot refuses symlinks: {path}")
        if (path.is_dir() and path.name.startswith("build")) or path.name in {"__pycache__", ".pytest_cache", ".deps", ".venv", "evidence", ".cache"}:
            return
        if path.is_dir():
            for child in sorted(path.iterdir()):
                copy(child)
            return
        if path.suffix in {".so", ".pyc", ".pyo", ".o", ".a"}:
            return
        if not path.is_file():
            raise Failed(f"nonregular source input: {path}")
        relative = path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        before = digest(path)
        shutil.copy2(path, target)
        if digest(target) != before or digest(path) != before:
            raise Failed(f"source changed while snapshotting: {relative}")
        entries.append({"path": str(relative), "sha256": before, "bytes": target.stat().st_size})
    for name in SOURCE_ROOTS:
        copy(source / name)
    return {"files": entries, "digest": hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest(),
            "policy": "explicit source roots; no inherited build, native .so, cache, dependency tree or symlink"}


def child_environment(output, dependencies, device):
    # Deliberately start empty. No inherited PYTHON*, PYTEST*, PIP_*, RAGGED_*,
    # loader/preload, compiler flags, parent virtualenv or user-site settings.
    cuda, llvm = dependencies["cuda"], dependencies["llvm"]
    return {"PATH": f"{output}/venv/bin:{cuda}/bin:{llvm}/bin:/usr/bin:/bin",
            "TMPDIR": str(output / "tmp"),
            "XDG_CACHE_HOME": str(output / "cache"), "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
            "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONSAFEPATH": "1",
            "PYTHONHASHSEED": "0", "PIP_CONFIG_FILE": os.devnull, "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INPUT": "1", "PIP_NO_CACHE_DIR": "1", "CUDA_HOME": cuda, "CUDA_PATH": cuda,
            "CUDA_VISIBLE_DEVICES": device["uuid"], "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "LD_LIBRARY_PATH": f"{cuda}/lib64:{llvm}/lib",
            "TRITON_PTXAS_PATH": f"{cuda}/bin/ptxas",
            "TRITON_LIBDEVICE_PATH": f"{cuda}/nvvm/libdevice/libdevice.10.bc",
            "TORCHINDUCTOR_CACHE_DIR": str(output / "cache/inductor"), "TRITON_CACHE_DIR": str(output / "cache/triton"),
            "TORCH_HOME": str(output / "cache/torch"), "CUDA_CACHE_PATH": str(output / "cache/cuda"),
            "RAGGED_COMPILE": str(output / "build/bin/ragged-compile"),
            "RAGGED_BASELINE_LIBRARY": str(output / "native/libragged_baselines.so"),
            "RAGGED_GPU_FAILURE_DIR": str(output / "gpu-failures")}


def junit_counts(path):
    cases = ET.parse(path).getroot().findall(".//testcase")
    counts = {"tests": len(cases), **{name: sum(case.find(element) is not None for case in cases)
                                    for name, element in (("skipped", "skipped"), ("failures", "failure"), ("errors", "error"))}}
    if not counts["tests"] or any(counts[key] for key in ("skipped", "failures", "errors")):
        raise Failed(f"strict device tests require nonempty execution and zero skips/failures/errors: {counts}")
    return counts


def validate_identity(path, output, expected_uuid, pins, lock_sha256):
    record = read_json(path)
    venv = (output / "venv").resolve()
    if (record.get("status") != "verified" or record.get("isolated") is not True
            or record.get("user_site_enabled") is not False or Path(record.get("prefix", "/")).resolve() != venv
            or record.get("source_lock_sha256") != lock_sha256 or record.get("torch_cuda") != "13.2"
            or record.get("device", {}).get("uuid") != expected_uuid or record.get("device", {}).get("capability") != [8, 9]):
        raise Failed("worker identity does not establish the isolated pinned GPU environment")
    if (record.get("executable") != str(venv / "bin/python")
            or not isinstance(record.get("base_prefix"), str) or not record["base_prefix"]
            or Path(record["base_prefix"]).resolve() == venv):
        raise Failed("worker executable/base-prefix contradicts its fresh venv identity")
    paths = record.get("sys_path")
    if not isinstance(paths, list) or any(not isinstance(path, str) for path in paths):
        raise Failed("worker search-path identity is missing")
    for raw_path in paths:
        path = Path(raw_path or ".").resolve()
        if (path == (output / "source/python").resolve()
                or (path.name == "python" and not path.is_relative_to(venv))
                or ("site-packages" in path.parts and not path.is_relative_to(venv))):
            raise Failed("worker search path exposes source Python or foreign site packages")
    if record.get("driver", {}).get("status") != "ready" or record["driver"].get("device") != record["device"]:
        raise Failed("worker CUDA Driver identity contradicts its device identity")
    for name in ("ragged", "ragged._native", "torch", "torch._C", "triton", "numpy", "pytest"):
        info = record.get("modules", {}).get(name, {})
        module_path = Path(info.get("file", "/")).resolve()
        if not module_path.is_relative_to(venv) or not module_path.is_file() or digest(module_path) != info.get("sha256"):
            raise Failed(f"worker {name} identity is missing, altered or outside the fresh venv")
    observed = {}
    for item in record.get("packages", []):
        name = item["name"].lower().replace("_", "-")
        if name in observed or not Path(item.get("location", "/")).resolve().is_relative_to(venv):
            raise Failed("worker distribution inventory has duplicate or external packages")
        observed[name] = item["version"]
    if any(observed.get(name) != version for name, version in pins.items()):
        raise Failed("worker distribution inventory does not match every direct lock pin")
    return {"status": "verified", "modules": list(record["modules"]), "device_uuid": expected_uuid}


def validate_pip_report(path, *, require_torch=False):
    record = read_json(path)
    installs = record.get("install")
    if not isinstance(installs, list) or not installs:
        raise Failed("pip report must retain its resolved installations")
    for item in installs:
        metadata = item.get("metadata", {})
        download = item.get("download_info", {})
        sha = download.get("archive_info", {}).get("hashes", {}).get("sha256", "")
        if not metadata.get("name") or not metadata.get("version") or not download.get("url", "").startswith("https://") or not re.fullmatch(r"[a-f0-9]{64}", sha):
            raise Failed("pip report lacks a concrete version, HTTPS source or wheel SHA256")
    if require_torch and not any(item["metadata"]["name"].lower() == "torch" and item["metadata"]["version"] == "2.14.1+cu132" for item in installs):
        raise Failed("pip report did not install the requested CUDA Torch build")
    return {"resolved_install_count": len(installs)}


def validate_benchmark(path, op, methods, compiler_hash, *, expected_repeats=None, expected_warmup=None):
    report = read_json(path)
    cells = report.get("methods", [])
    if report.get("status") != "complete" or len(cells) != len(methods) or {cell.get("method") for cell in cells} != set(methods):
        raise Failed(f"{op} benchmark lacks the complete requested method set")
    for cell in cells:
        if cell.get("status") != "measured" or not isinstance(cell.get("oracle"), dict):
            raise Failed(f"{op}/{cell.get('method')} did not pass its oracle and measurement")
        oracle = cell["oracle"]
        for field in ("finite_values", "nan_values"):
            if type(oracle.get(field)) is not int or not 0 <= oracle[field] <= 2**63 - 1:
                raise Failed("benchmark lacks valid fixed-oracle element counts")
        for field in ("max_abs_error", "max_fraction_of_tolerance"):
            if type(oracle.get(field)) not in (int, float) or not math.isfinite(oracle[field]) or oracle[field] < 0:
                raise Failed("benchmark lacks finite fixed-oracle error evidence")
        if oracle["max_fraction_of_tolerance"] > 1:
            raise Failed("benchmark exceeds the fixed oracle tolerance")
        if cell["method"].startswith("ragged-") and cell.get("compiler_sha256") != compiler_hash:
            raise Failed("benchmark used a different compiler")
        for key in ("complete_call_cuda_event_ms", "complete_call_wall_until_complete_ms"):
            # The unchanged harness retains samples under these explicit keys.
            summary = cell.get(key, {})
            samples = summary.get("samples")
            if not isinstance(samples, list) or not samples or any(type(value) not in (int, float) or value < 0 or not math.isfinite(value) for value in samples):
                raise Failed(f"{op}/{cell['method']} lacks finite actual timing samples")
            if type(summary.get("count")) is not int or summary["count"] != len(samples):
                raise Failed("benchmark timing count disagrees with actual samples")
            if expected_repeats is not None and summary["count"] != expected_repeats:
                raise Failed("benchmark timing count differs from the requested repeats")
            ordered = sorted(samples)
            position = (len(samples) - 1) * 0.9
            low, high = math.floor(position), math.ceil(position)
            expected = {"median": statistics.median(samples), "min": min(samples), "max": max(samples),
                        "p90": ordered[low] + (ordered[high] - ordered[low]) * (position - low)}
            for field, value in expected.items():
                actual = summary.get(field)
                if type(actual) not in (int, float) or not math.isfinite(actual) or not math.isclose(actual, value, rel_tol=1e-12, abs_tol=1e-12):
                    raise Failed("benchmark timing summary disagrees with retained samples")
        if cell["complete_call_cuda_event_ms"]["count"] != cell["complete_call_wall_until_complete_ms"]["count"]:
            raise Failed("benchmark event and wall counts disagree")
        if expected_warmup is not None and (type(cell.get("warmup_calls")) is not int or cell["warmup_calls"] != expected_warmup):
            raise Failed("benchmark warmup count differs from the requested warmup")
    case = report.get("case", {})
    if case.get("op") != op or any(not re.fullmatch(r"[0-9a-f]{64}", case.get(key, ""))
                                     for key in ("values_sha256", "offsets_sha256")):
        raise Failed("benchmark is missing its exact input identity")
    return {"methods": list(methods), "values_sha256": case["values_sha256"], "offsets_sha256": case["offsets_sha256"]}


def validate_benchmark_proof(identity_path, destination, source, methods, compiler_hash, native_hash):
    proof = read_json(identity_path).get("benchmark_data_validation", {})
    expected = {"status": "verified", "case_npz_sha256": digest(destination / "case.npz"),
                "result_sha256": digest(destination / "results.json"), "harness_sha256": digest(source / "bench/run.py"),
                "compiler_sha256": compiler_hash, "native_library_sha256": native_hash, "validated_methods": list(methods)}
    if any(proof.get(key) != value for key, value in expected.items()):
        raise Failed("benchmark worker lacks a matching strict input/implementation validation proof")
    return expected


def validate_pytest_selection(xml, identity_path, expected_nodeids=None):
    counts = junit_counts(xml)
    selection = read_json(identity_path).get("pytest_selection", {})
    nodes = selection.get("nodeids")
    if (not isinstance(nodes, list) or any(not isinstance(node, str) or not node for node in nodes)
            or len(set(nodes)) != len(nodes) or type(selection.get("collected")) is not int
            or selection["collected"] != counts["tests"]
            or len(nodes) != counts["tests"] or selection.get("deselected") != []):
        raise Failed("device pytest did not execute exactly its retained, undeselected collection")
    executed = []
    for case in ET.parse(xml).getroot().findall(".//testcase"):
        properties = [item.get("value") for item in case.findall("./properties/property")
                      if item.get("name") == "ragged_nodeid"]
        if len(properties) != 1 or not properties[0]:
            raise Failed("device JUnit testcase lacks its unique collected node identity")
        executed.append(properties[0])
    if len(set(executed)) != len(executed) or set(executed) != set(nodes):
        raise Failed("device JUnit test identities differ from the actual pytest collection")
    if expected_nodeids is not None:
        try:
            expected = checked_nodeids(expected_nodeids)
        except ValueError as error:
            raise Failed(str(error)) from error
        if set(nodes) != set(expected):
            raise Failed("device pytest collection differs from the explicitly requested sanitizer scope")
    return counts


def run(args, *, preflight=driver_preflight, runner=bounded_run):
    requested_output = args.output.absolute()
    output = requested_output.resolve()
    # Reject reuse even if empty. Do not change an existing evidence path.
    if output.exists() or requested_output.is_symlink():
        print(f"failed: output must not exist: {output}", file=sys.stderr)
        return 1
    if output.is_relative_to(ROOT) and output.relative_to(ROOT).parts[0] in SOURCE_ROOTS:
        print("failed: output cannot be inside a snapshotted source directory", file=sys.stderr)
        return 1
    output.mkdir(parents=True)
    report = {"schema": 1, "status": "running", "scope": "fresh bounded GPU correctness and same-input comparisons; not the full benchmark matrix",
              "started_unix_ns": time.time_ns(), "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
              "output": str(output), "stages": [], "optional_fbgemm": "not installed or qualified by this entry point",
              "runner_source_sha256": digest(__file__), "device_execution_passed": False}
    def save():
        write_json(output / "reproduction.json", report)
    try:
        report["preflight"] = preflight(device=args.device)
        save()
        if report["preflight"].get("status") != "ready":
            raise Blocked(report["preflight"].get("reason", "CUDA Driver is not ready"))
        device = report["preflight"].get("device", {})
        if device.get("capability") != [8, 9] or not re.fullmatch(r"GPU-[0-9a-fA-F-]{36}", device.get("uuid", "")):
            raise Failed("Driver readiness lacks a concrete sm_89 UUID")
        report["dependencies"] = deps = prerequisites(args)
        for name in ("tmp", "cache", "logs", "results", "work"):
            (output / name).mkdir()
        env = child_environment(output, deps, device)
        report["environment_policy"] = "fresh explicit environment; inherited environment is not forwarded"
        source = output / "source"
        report["snapshot"] = snapshot_source(ROOT, source)
        write_json(output / "source-manifest.json", report["snapshot"])
        lock = read_json(source / "environment.lock")
        pins = lock["reproduction"]["packages"]
        expected_names = {"pip", "numpy", "pytest", "lit", "nanobind", "scikit-build-core", "packaging", "torch", "triton"}
        if set(pins) != expected_names or any(not isinstance(value, str) or not re.fullmatch(r"[0-9]+(?:\.[0-9]+){1,2}(?:\+[a-z0-9]+)?", value) for value in pins.values()):
            raise Failed("environment.lock must contain exact versions for all reproduction dependencies")
        if pins["torch"] != "2.14.1+cu132" or pins["triton"] != "3.8.0" or lock["reproduction"]["torch_index_url"] != "https://download.pytorch.org/whl/cu132":
            raise Failed("reproduction lock does not match the required Torch/Triton CUDA target")
        report["pins"] = pins
        requirements = output / "direct-pins.txt"
        requirements.write_text("".join(f"{name}=={version}\n" for name, version in pins.items()))
        report["pin_scope"] = "direct requirements fixed here; all resolved wheel URLs/hashes retained in pip reports and installed inventory"
        save()

        def stage(name, argv, *, cwd=source, artifacts=(), timeout=None, validator=None):
            item = {"name": name, "argv": list(map(str, argv)), "cwd": str(cwd), "env": env.copy(),
                    "timeout_seconds": timeout or args.stage_timeout_seconds, "status": "running", "artifacts": []}
            report["stages"].append(item)
            log = output / "logs" / f"{len(report['stages']):02d}-{name}.log"
            item["log"] = str(log.relative_to(output))
            save()
            print(f"[{len(report['stages'])}] {name}", flush=True)
            started = time.monotonic()
            try:
                with log.open("w") as stream:
                    completed = runner(item["argv"], cwd=str(cwd), env=env.copy(), stdout=stream,
                                       stderr=subprocess.STDOUT, check=False, timeout=item["timeout_seconds"])
                item["exit_code"] = completed.returncode
                if completed.returncode != 0:
                    raise Failed(f"{name} exited {completed.returncode}; see {log}")
                for artifact in artifacts:
                    path = Path(artifact)
                    if not path.is_file():
                        raise Failed(f"{name} did not produce required artifact {path}")
                    item["artifacts"].append({"path": str(path.relative_to(output)), "sha256": digest(path), "bytes": path.stat().st_size})
                if validator:
                    item["validation"] = validator(item)
                item["status"] = "passed"
            except (Exception, KeyboardInterrupt) as error:
                item["exit_code"] = getattr(error, "returncode", item.get("exit_code"))
                if isinstance(error, subprocess.TimeoutExpired):
                    item["timed_out"] = True
                item.update(status="failed", error=f"{type(error).__name__}: {error}")
                raise Failed(str(error)) from error
            finally:
                item["wall_seconds"] = time.monotonic() - started
                if log.is_file():
                    item["log_sha256"] = digest(log)
                save()
            return item

        base = deps["tools"]["python"]
        stage("python-version", [base, "-I", "-c", "import json,sys,venv,ensurepip; print(json.dumps({'version':list(sys.version_info[:2]),'full':sys.version}))"])
        if read_json(output / report["stages"][-1]["log"])["version"] != [3, 12]:
            raise Blocked("the base Python must be version 3.12")
        if deps["llvm_config"]:
            stage("sdk-version", [deps["llvm_config"], "--version"])
            if (output / report["stages"][-1]["log"]).read_text().strip() != "23.1.3":
                raise Blocked("the provisioned LLVM SDK does not report version 23.1.3")
        stage("create-venv", [base, "-I", "-m", "venv", str(output / "venv")], artifacts=[output / "venv/pyvenv.cfg"])
        config = (output / "venv/pyvenv.cfg").read_text()
        if not re.search(r"include-system-site-packages\s*=\s*false", config):
            raise Failed("fresh venv unexpectedly enables system site packages")
        python = str(output / "venv/bin/python")
        pip = [python, "-I", "-m", "pip", "--disable-pip-version-check"]
        tools_report, torch_report = output / "results/pip-tools.json", output / "results/pip-torch.json"
        stage("install-pinned-tools", [*pip, "install", "--no-input", "--no-cache-dir", "--index-url", "https://pypi.org/simple",
              "--constraint", requirements, "--report", tools_report,
              *[f"{name}=={version}" for name, version in pins.items() if name != "torch"]], artifacts=[tools_report],
              validator=lambda _: validate_pip_report(tools_report))
        stage("install-pinned-torch", [*pip, "install", "--no-input", "--no-cache-dir", "--index-url", lock["reproduction"]["torch_index_url"],
              "--extra-index-url", "https://pypi.org/simple", "--constraint", requirements,
              "--report", torch_report, f"torch=={pins['torch']}"], artifacts=[torch_report],
              validator=lambda _: validate_pip_report(torch_report, require_torch=True))
        stage("dependency-consistency", [*pip, "check"])
        build = output / "build"
        cmake = deps["tools"]["cmake"]
        configure = [cmake, "-S", source, "-B", build, "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
                     f"-DMLIR_DIR={deps['llvm']}/lib/cmake/mlir", "-DRAGGED_INCLUDE_TESTS=OFF",
                     "-DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF", "-DCMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=OFF",
                     f"-DCMAKE_C_COMPILER={deps['tools']['cc']}", f"-DCMAKE_CXX_COMPILER={deps['tools']['c++']}"]
        if deps["archive"]:
            configure.append("-DRAGGED_LLVM_SOURCE_ARCHIVE=" + deps["archive"])
        if deps["cmake_prefixes"]:
            configure.append("-DCMAKE_PREFIX_PATH=" + ";".join(deps["cmake_prefixes"]))
        stage("configure-compiler", configure, artifacts=[build / "CMakeCache.txt"])
        compiler = build / "bin/ragged-compile"
        stage("build-compiler", [cmake, "--build", build, "--parallel", str(args.jobs)], artifacts=[compiler, build / "bin/ragged-opt"])
        report["compiler_sha256"] = digest(compiler)
        def compiler_provenance_check(item):
            provenance = read_json(output / item["log"])
            if (provenance.get("llvm_commit") != lock["required"]["llvm_commit"] or provenance.get("llvm_version") != "23.1.3"
                    or provenance.get("chip") != "sm_89" or not re.search(r"\bV13\.2\.86\b", provenance.get("ptxas_version", ""))):
                raise Failed("fresh compiler did not retain the pinned LLVM/CUDA target identity")
            return provenance
        report["compiler_toolchain"] = stage("compiler-provenance", [compiler, "--toolchain-json"], validator=compiler_provenance_check)["validation"]
        def wheel_check(_):
            wheels = list((output / "wheels").glob("ragged_mlir-*.whl"))
            if len(wheels) != 1:
                raise Failed("fresh build must produce exactly one native ragged wheel")
            with zipfile.ZipFile(wheels[0]) as archive:
                native_members = [name for name in archive.namelist() if name.endswith(".so")]
                if len(native_members) != 1 or not native_members[0].startswith("ragged/_native."):
                    raise Failed("fresh ragged wheel must contain exactly its newly built native module")
                package_files = {}
                for name in archive.namelist():
                    if name.startswith("ragged/") and not name.endswith("/"):
                        relative = name.removeprefix("ragged/")
                        if ".." in Path(name).parts or relative in package_files:
                            raise Failed("wheel contains an ambiguous package member")
                        with archive.open(name) as stream:
                            package_files[relative] = hashlib.file_digest(stream, "sha256").hexdigest()
            return {"path": str(wheels[0].relative_to(output)), "sha256": digest(wheels[0]), "package_files": package_files,
                    "native_member": native_members[0].removeprefix("ragged/")}
        report["wheel"] = stage("build-wheel", [*pip, "wheel", "--no-deps", "--no-build-isolation", "--wheel-dir", output / "wheels", source],
                                validator=wheel_check)["validation"]
        report["stages"][-1]["artifacts"].append(report["wheel"])
        stage("install-wheel", [*pip, "install", "--no-deps", output / report["wheel"]["path"]])
        native = output / "native/libragged_baselines.so"
        def native_check(_):
            native_report = read_json(output / "native/build.json")
            if (native_report.get("exit_code") != 0 or native_report.get("source_unchanged") is not True
                    or native_report.get("library_sha256") != digest(native)
                    or native_report.get("source_sha256") != digest(source / "baselines/cuda/baselines.cu")):
                raise Failed("native baseline build did not retain its actual source/library identity")
            return {"library_sha256": digest(native), "source_sha256": native_report["source_sha256"]}
        stage("build-native-baselines", [python, "-I", source / "scripts/build_baselines.py", "--nvcc", deps["cuda"] + "/bin/nvcc",
              "--cccl", deps["cccl"], "--output", output / "native"], artifacts=[native, output / "native/build.json"], validator=native_check)
        report["native_library_sha256"] = digest(native)
        worker = source / "scripts/reproduce_gpu_worker.py"
        def worker_command(action, identity):
            return [python, "-I", worker, action, "--source", source, "--venv", output / "venv", "--identity-output", identity,
                    "--expected-uuid", device["uuid"]]
        identity = output / "results/fresh-identity.json"
        def check_identity(path):
            validation = validate_identity(path, output, device["uuid"], pins, digest(source / "environment.lock"))
            info = read_json(path)
            package_root = Path(info["modules"]["ragged"]["file"]).parent
            for module, relative in (("ragged", "__init__.py"), ("ragged._native", report["wheel"]["native_member"])):
                loaded = info["modules"][module]
                if (Path(loaded["file"]).resolve() != (package_root / relative).resolve()
                        or loaded["sha256"] != report["wheel"]["package_files"].get(relative)):
                    raise Failed("loaded ragged module differs from its exact freshly built wheel member: " + module)
            for relative, expected_hash in report["wheel"]["package_files"].items():
                installed = package_root / relative
                if not installed.resolve().is_relative_to((output / "venv").resolve()) or not installed.is_file() or digest(installed) != expected_hash:
                    raise Failed("installed ragged package differs from the freshly built wheel: " + relative)
            return validation
        stage("fresh-identity", worker_command("identity", identity), artifacts=[identity], validator=lambda _: check_identity(identity))
        report["installed_identity"] = read_json(identity)
        ini = output / "pytest.ini"
        ini.write_text("[pytest]\nmarkers =\n    " + MARKER + "\n")
        xml, test_identity = output / "results/gpu.xml", output / "results/gpu-process-identity.json"
        test_command = [*worker_command("pytest", test_identity), "--", "-c", ini, "-o", f"pythonpath={source}", "--rootdir", source,
                        "-ra", f"--junitxml={xml}", f"--basetemp={output / 'work/gpu-correctness'}", *FOCUSED_TESTS]
        stage("gpu-correctness", test_command, artifacts=[xml, test_identity],
              validator=lambda _: {"junit": validate_pytest_selection(xml, test_identity), "identity": check_identity(test_identity)})
        report["gpu_correctness"] = junit_counts(xml)
        benchmark_inputs = []
        for op in ("sum", "max", "softmax"):
            destination = output / "comparisons" / op
            identity = output / "results" / f"{op}-process-identity.json"
            methods = COMMON_METHODS + (("torch-segment",) if op != "softmax" else ())
            command = [*worker_command("benchmark", identity), "--", "--op", op, "--dtype", "f32", "--offsets-dtype", "i64",
                       "--d", "1", "--distribution", "long_tail", "--segments", "128", "--total-elements", "8192", "--seed", "711",
                       "--length-hints", "--warmup", "3", "--repeats", "10", "--compiler", compiler, "--native-library", native,
                       "--methods", *methods, "--output", destination]
            stage("comparison-" + op, command, artifacts=[destination / "results.json", destination / "case.npz", identity],
                  validator=lambda _: {"benchmark": validate_benchmark(destination / "results.json", op, methods, report["compiler_sha256"], expected_repeats=10, expected_warmup=3),
                                       "identity": check_identity(identity),
                                       "proof": validate_benchmark_proof(identity, destination, source, methods, report["compiler_sha256"], report["native_library_sha256"])})
            benchmark_inputs.append(validate_benchmark(destination / "results.json", op, methods, report["compiler_sha256"], expected_repeats=10, expected_warmup=3))
        if len({(item["values_sha256"], item["offsets_sha256"]) for item in benchmark_inputs}) != 1:
            raise Failed("sum/max/softmax comparisons did not retain the same generated input")
        report["comparisons"] = benchmark_inputs
        if args.sanitizers:
            report["sanitizers"] = {}
            # FBGEMM is optional and deliberately not installed by this command.
            # Collect the complete remaining suite in the actual fresh-wheel
            # process first. Run each selected node exactly once per tool.
            collection = output / "results/sanitizer-collection-identity.json"
            collection_command = [*worker_command("pytest", collection), "--", "-c", ini,
                                  "-o", f"pythonpath={source}", "--rootdir", source,
                                  "--collect-only", "-q", "test/gpu", "--ignore=test/gpu/test_fbgemm_baseline.py"]
            def validate_sanitizer_collection(_):
                identity_result = check_identity(collection)
                nodes = collected_nodeids(read_json(collection))
                return {"identity": identity_result, "groups": partition_sanitizer_nodeids(nodes)}
            stage("sanitizer-collection", collection_command, artifacts=[collection],
                  validator=validate_sanitizer_collection)
            # Use the same validated selection for every tool rather than
            # recollecting a broad path under each one.
            groups = partition_sanitizer_nodeids(collected_nodeids(read_json(collection)))
            suppression = validate_suppression_file(source / "test/gpu/driver-load-api-suppressions.xml")
            report["sanitizer_scope"] = {"collection_identity": str(collection.relative_to(output)),
                                         "groups": groups, "suppression": suppression}
            for tool in ("memcheck", "racecheck", "synccheck"):
                report["sanitizers"][tool] = {}
                for group, nodes in groups.items():
                    if not nodes:
                        continue
                    label = f"{tool}-{group}"
                    xml = output / "results" / f"{label}.xml"
                    identity = output / "results" / f"{label}-process-identity.json"
                    prefix = [deps["cuda"] + "/bin/compute-sanitizer", "--tool", tool,
                              "--target-processes", "all", "--error-exitcode", "86"]
                    if group == "loader_api":
                        prefix += ["--suppressions", suppression["path"]]
                    command = [*prefix, *worker_command("pytest", identity), "--", "-c", ini,
                               "-o", f"pythonpath={source}", "--rootdir", source,
                               "-ra", f"--junitxml={xml}", f"--basetemp={output / 'work' / label}", *nodes]
                    stage("sanitizer-" + label, command, artifacts=[xml, identity],
                          validator=lambda _: {"junit": validate_pytest_selection(xml, identity, expected_nodeids=nodes),
                                               "identity": check_identity(identity)})
                    report["sanitizers"][tool][group] = {"junit": junit_counts(xml), "nodeids": nodes,
                                                         "suppression": suppression if group == "loader_api" else None}
        for entry in report["snapshot"]["files"]:
            if digest(source / entry["path"]) != entry["sha256"]:
                raise Failed("snapshot source was modified during execution: " + entry["path"])
        report.update(status="passed", exit_code=0, device_execution_passed=True)
    except Blocked as error:
        report.update(status="blocked", exit_code=2, reason=str(error))
    except Exception as error:
        report.update(status="failed", exit_code=1, reason=f"{type(error).__name__}: {error}")
    report["finished_unix_ns"] = time.time_ns()
    save()
    print(f"{report['status']}: {output / 'reproduction.json'}", flush=True)
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
