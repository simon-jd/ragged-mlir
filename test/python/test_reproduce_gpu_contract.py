"""Independent GPU reproduction orchestration tests with synthetic host fixtures.

Driver identities, installed package records, artifacts and measurements created
by this module are synthetic test inputs. No dependency is installed, no source
archive is downloaded, and no GPU qualification or clean GPU reproduction is
claimed by these tests.
"""
from __future__ import annotations

import builtins
import ctypes
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import xml.etree.ElementTree as ET
import zipfile

import pytest

from scripts import reproduce_gpu as reproduction
from scripts import gpu_validation_scope as sanitizer_scope


class _DriverFunction:
    """A ctypes-compatible host spy, with no CUDA execution."""
    def __init__(self, body):
        self.body = body
    def __call__(self, *args):
        return self.body(*args)


class _SyntheticDriver:
    def __init__(self, *, init_result=0, capability=(8, 9)):
        self.calls = []
        self.init_result, self.capability = init_result, capability
        functions = {
            "cuInit": lambda flags: self.init_result,
            "cuDriverGetVersion": lambda pointer: self._set(pointer, ctypes.c_int, 13020),
            "cuDeviceGetCount": lambda pointer: self._set(pointer, ctypes.c_int, 1),
            "cuDeviceGet": lambda pointer, ordinal: self._set(pointer, ctypes.c_int, 0),
            "cuDeviceGetAttribute": self._attribute,
            "cuDeviceGetName": self._name,
            "cuDeviceGetUuid_v2": self._uuid,
            "cuDeviceGetUuid": self._uuid,
            "cuDeviceTotalMem_v2": lambda pointer, device: self._set(pointer, ctypes.c_size_t, 24 << 30),
            "cuDeviceTotalMem": lambda pointer, device: self._set(pointer, ctypes.c_size_t, 24 << 30),
        }
        for name, body in functions.items():
            def observed(*args, _name=name, _body=body):
                self.calls.append(_name)
                return _body(*args)
            setattr(self, name, _DriverFunction(observed))

    def _set(self, pointer, kind, value):
        ctypes.cast(pointer, ctypes.POINTER(kind))[0] = value
        return 0

    def _attribute(self, pointer, attribute, device):
        values = {75: self.capability[0], 76: self.capability[1], 16: 128, 38: 72 << 20}
        if attribute not in values:
            raise AssertionError(f"unmodeled host Driver fixture attribute {attribute}")
        return self._set(pointer, ctypes.c_int, values[attribute])

    def _name(self, pointer, length, device):
        value = b"synthetic Driver fixture; not GPU evidence\x00"
        ctypes.memmove(pointer, value, min(length, len(value)))
        return 0

    def _uuid(self, pointer, device):
        ctypes.memmove(pointer, bytes(range(16)), 16)
        return 0


def _forbidden(*args, **kwargs):
    raise AssertionError("this host contract must not install, download or execute child commands")


def _arguments(tmp_path, *extra):
    return reproduction.parse_args(["--output", str(tmp_path / "reproduction"),
                                    "--llvm-prefix", str(tmp_path / "missing-sdk"), *extra])


def _reject_torch_import(monkeypatch):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise AssertionError("hardware preflight must not depend on parent Torch")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)


def test_driver_no_device_probe_uses_stdlib_and_never_parent_torch(monkeypatch):
    driver = _SyntheticDriver(init_result=100)
    monkeypatch.setattr(ctypes, "CDLL", lambda *args, **kwargs: driver)
    _reject_torch_import(monkeypatch)
    result = reproduction.driver_preflight(device=0)
    assert result["status"] == "blocked"
    assert "100" in result["reason"] or "NO_DEVICE" in result["reason"] or "no device" in result["reason"].lower()
    assert "cuInit" in driver.calls and "cuDeviceGet" not in driver.calls


def test_missing_driver_library_is_blocked_without_parent_torch(monkeypatch):
    monkeypatch.setattr(ctypes, "CDLL", lambda *args, **kwargs: (_ for _ in ()).throw(OSError(
        "synthetic missing CUDA Driver library")))
    _reject_torch_import(monkeypatch)
    result = reproduction.driver_preflight(device=0)
    assert result["status"] == "blocked" and result["reason"]


def test_ready_driver_preflight_is_independent_of_parent_cpu_torch(monkeypatch):
    driver = _SyntheticDriver()
    monkeypatch.setattr(ctypes, "CDLL", lambda *args, **kwargs: driver)
    _reject_torch_import(monkeypatch)
    result = reproduction.driver_preflight(device=0)
    assert result["status"] == "ready"
    assert result["device"]["capability"] == [8, 9]
    assert "synthetic" in result["device"]["name"]


def test_early_no_device_keeps_zero_stages_and_never_creates_bootstrap_work(tmp_path):
    args = _arguments(tmp_path)
    observed = []
    def preflight(*, device):
        observed.append(device)
        return {"status": "blocked", "reason": "synthetic CUDA_ERROR_NO_DEVICE 100"}
    code = reproduction.run(args, preflight=preflight, runner=_forbidden)
    assert code == 2 and observed == [0]
    report = json.loads((args.output / "reproduction.json").read_text())
    assert report["status"] == "blocked" and report["exit_code"] == 2
    assert report["stages"] == []
    assert sorted(path.name for path in args.output.iterdir()) == ["reproduction.json"]


def test_existing_output_is_not_overwritten_or_retried(tmp_path):
    args = _arguments(tmp_path)
    args.output.mkdir()
    sentinel = args.output / "reproduction.json"
    sentinel.write_text("synthetic previous evidence\n")
    before = sentinel.read_bytes()
    assert reproduction.run(args, preflight=_forbidden, runner=_forbidden) == 1
    assert sentinel.read_bytes() == before


def _minimal_source(path):
    path.mkdir()
    for name in reproduction.SOURCE_ROOTS:
        child = path / name
        if name in {"CMakeLists.txt", "README.md", "environment.lock", "pyproject.toml"}:
            child.write_text("synthetic source fixture; not an executable project\n")
        else:
            child.mkdir()
            (child / "fixture.txt").write_text("synthetic source fixture\n")
    (path / "environment.lock").write_bytes((Path(__file__).resolve().parents[2] / "environment.lock").read_bytes())
    return path


def test_snapshot_excludes_old_native_binary_build_and_cache_trees(tmp_path):
    source = _minimal_source(tmp_path / "checkout")
    ignored = ["python/ragged/_native.so", "python/ragged/__pycache__/legacy.pyc",
               "runtime/build-old/object.o", "bench/.cache/old.json", "test/.pytest_cache/old.txt"]
    for relative in ignored:
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic forbidden inherited artifact\n")
    for relative in (".venv/legacy.txt", ".deps/legacy.txt", "evidence/legacy.txt", "build/legacy.txt"):
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic forbidden root artifact\n")
    destination = tmp_path / "snapshot"
    manifest = reproduction.snapshot_source(source, destination)
    assert manifest["files"]
    assert all(not (destination / relative).exists() for relative in ignored)
    assert not any((destination / name).exists() for name in (".venv", ".deps", "evidence", "build"))
    for entry in manifest["files"]:
        path = destination / entry["path"]
        assert path.read_bytes() == (source / entry["path"]).read_bytes()
        assert entry["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_snapshot_refuses_source_symlink_escape(tmp_path):
    source = _minimal_source(tmp_path / "checkout")
    external = tmp_path / "external.txt"
    external.write_text("synthetic external source must not be read or copied\n")
    (source / "python/escape.py").symlink_to(external)
    with pytest.raises(reproduction.Failed, match="symlink"):
        reproduction.snapshot_source(source, tmp_path / "snapshot")


def test_snapshot_rejects_source_change_during_copy(tmp_path, monkeypatch):
    source = _minimal_source(tmp_path / "checkout")
    original = reproduction.shutil.copy2
    changed = []
    def changing_copy(source_path, target, *args, **kwargs):
        result = original(source_path, target, *args, **kwargs)
        if not changed:
            Path(source_path).write_text("synthetic changed source\n")
            changed.append(source_path)
        return result
    monkeypatch.setattr(reproduction.shutil, "copy2", changing_copy)
    with pytest.raises(reproduction.Failed, match="changed"):
        reproduction.snapshot_source(source, tmp_path / "snapshot")


def test_child_environment_drops_inherited_import_loader_package_compiler_and_secret_settings(tmp_path, monkeypatch):
    contaminated = {name: "synthetic inherited setting must not pass" for name in (
        "PYTHONPATH", "PYTHONHOME", "PYTEST_ADDOPTS", "VIRTUAL_ENV", "PIP_INDEX_URL",
        "PIP_EXTRA_INDEX_URL", "PIP_FIND_LINKS", "PIP_TRUSTED_HOST", "LD_PRELOAD",
        "LD_LIBRARY_PATH", "CFLAGS", "CXXFLAGS", "RAGGED_COMPILE", "RAGGED_BASELINE_LIBRARY",
        "AWS_SECRET_ACCESS_KEY", "GH_TOKEN", "HTTP_PROXY", "HTTPS_PROXY")}
    for name, value in contaminated.items():
        monkeypatch.setenv(name, value)
    output = tmp_path / "fresh-output"
    deps = {"cuda": str(tmp_path / "provisioned-cuda"), "llvm": str(tmp_path / "provisioned-llvm")}
    device = {"uuid": "GPU-00010203-0405-0607-0809-0a0b0c0d0e0f"}
    env = reproduction.child_environment(output, deps, device)
    assert not any(value == "synthetic inherited setting must not pass" for value in env.values())
    assert env["CUDA_VISIBLE_DEVICES"] == device["uuid"] and env["PYTHONNOUSERSITE"] == "1"
    assert env["PIP_CONFIG_FILE"] == reproduction.os.devnull
    assert env["RAGGED_COMPILE"] == str(output / "build/bin/ragged-compile")
    assert env["RAGGED_BASELINE_LIBRARY"] == str(output / "native/libragged_baselines.so")
    assert not any(name in env for name in ("PYTHONPATH", "PYTHONHOME", "LD_PRELOAD", "AWS_SECRET_ACCESS_KEY", "GH_TOKEN"))


def test_output_inside_snapshot_source_directory_rejects_before_preflight(tmp_path, monkeypatch):
    source = _minimal_source(tmp_path / "checkout")
    monkeypatch.setattr(reproduction, "ROOT", source)
    args = _arguments(tmp_path)
    args.output = source / "python/new-reproduction"
    assert reproduction.run(args, preflight=_forbidden, runner=_forbidden) == 1
    assert not args.output.exists()


def test_output_parent_symlink_cannot_escape_snapshot_path_guard(tmp_path, monkeypatch):
    source = _minimal_source(tmp_path / "checkout")
    monkeypatch.setattr(reproduction, "ROOT", source)
    alias = tmp_path / "innocent-parent-alias"
    alias.symlink_to(source / "python", target_is_directory=True)
    args = _arguments(tmp_path)
    args.output = alias / "new-reproduction"
    assert reproduction.run(args, preflight=_forbidden, runner=_forbidden) == 1
    assert not (source / "python/new-reproduction").exists()


@pytest.mark.parametrize("content", [
    "not XML", "<testsuites/>", '<testsuites><testsuite><testcase><skipped/></testcase></testsuite></testsuites>',
    '<testsuites><testsuite><testcase><failure/></testcase></testsuite></testsuites>',
    '<testsuites><testsuite><testcase><error/></testcase></testsuite></testsuites>',
])
def test_strict_device_junit_never_accepts_empty_skipped_failed_error_or_malformed_results(tmp_path, content):
    path = tmp_path / "synthetic-gpu.xml"
    path.write_text(content)
    with pytest.raises((reproduction.Failed, ET.ParseError)):
        reproduction.junit_counts(path)


def _benchmark_record(methods=("ragged-auto", "triton")):
    summary = {"samples": [1.0, 2.0, 3.0], "count": 3, "median": 2.0, "min": 1.0, "max": 3.0, "p90": 2.8}
    return {
        "schema": 1, "status": "complete", "synthetic_host_fixture": True,
        "case": {"op": "sum", "values_sha256": "a" * 64, "offsets_sha256": "b" * 64},
        "methods": [{"method": method, "status": "measured", "compiler_sha256": "c" * 64,
                     "oracle": {"finite_values": 1, "nan_values": 0, "max_abs_error": 0.0, "max_fraction_of_tolerance": 0.0},
                     "complete_call_wall_until_complete_ms": dict(summary),
                     "complete_call_cuda_event_ms": dict(summary)} for method in methods],
    }


@pytest.mark.parametrize("status", ["not_applicable", "unmeasured", "failed", "planned", "blocked"])
def test_required_comparison_cell_must_be_measured_not_optional_or_inapplicable(tmp_path, status):
    record = _benchmark_record()
    record["methods"][1]["status"] = status
    path = tmp_path / "synthetic-comparison.json"
    path.write_text(json.dumps(record))
    with pytest.raises(reproduction.Failed):
        reproduction.validate_benchmark(path, "sum", ("ragged-auto", "triton"), "c" * 64)


@pytest.mark.parametrize("status", ["partial", "planned", "failed", "blocked"])
def test_partial_or_planned_comparison_report_cannot_pass_reproduction(tmp_path, status):
    record = _benchmark_record()
    record["status"] = status
    path = tmp_path / "synthetic-comparison.json"
    path.write_text(json.dumps(record))
    with pytest.raises(reproduction.Failed):
        reproduction.validate_benchmark(path, "sum", ("ragged-auto", "triton"), "c" * 64)


@pytest.mark.parametrize("mutation", ["empty_oracle", "exceeds_tolerance", "nonfinite_oracle", "tampered_summary"])
def test_measured_label_cannot_hide_invalid_oracle_or_timing_evidence(tmp_path, mutation):
    record = _benchmark_record()
    cell = record["methods"][0]
    if mutation == "empty_oracle":
        cell["oracle"] = {}
    elif mutation == "exceeds_tolerance":
        cell["oracle"]["max_fraction_of_tolerance"] = 1.001
    elif mutation == "nonfinite_oracle":
        cell["oracle"]["max_abs_error"] = float("nan")
    else:
        cell["complete_call_wall_until_complete_ms"]["median"] = 0.001
    path = tmp_path / "synthetic-comparison.json"
    path.write_text(json.dumps(record))
    with pytest.raises((reproduction.Failed, ValueError)):
        reproduction.validate_benchmark(path, "sum", ("ragged-auto", "triton"), "c" * 64)


PIPELINE_STAGES = (
    "python-version", "sdk-version", "create-venv", "install-pinned-tools", "install-pinned-torch",
    "dependency-consistency", "configure-compiler", "build-compiler", "compiler-provenance",
    "build-wheel", "install-wheel", "build-native-baselines", "fresh-identity", "gpu-correctness",
    "comparison-sum", "comparison-max", "comparison-softmax",
)


def _synthetic_identity(output, source, uuid):
    venv = output / "venv"
    packages = venv / "lib/python3.12/site-packages"
    packages.mkdir(parents=True, exist_ok=True)
    modules = {}
    for name in ("ragged", "ragged._native", "torch", "torch._C", "triton", "numpy", "pytest"):
        if name == "ragged._native":
            relative = "ragged/_native.cpython-312-x86_64-linux-gnu.so"
        elif name == "torch._C":
            relative = "torch/_C.synthetic.so"
        else:
            relative = name.replace(".", "/") + "/__init__.py"
        path = packages / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(b"synthetic wheel module fixture\n" if name == "ragged" else
                             b"synthetic installed module fixture; not executable\n")
        modules[name] = {"file": str(path), "sha256": reproduction.digest(path)}
    pins = json.loads((source / "environment.lock").read_text())["reproduction"]["packages"]
    device = {"uuid": uuid, "capability": [8, 9], "name": "synthetic installed Driver identity fixture"}
    return {
        "schema": 1, "status": "verified", "synthetic_host_fixture": True,
        "python": "3.12 synthetic isolated worker", "executable": str(venv / "bin/python"),
        "prefix": str(venv), "base_prefix": "/synthetic/base-python", "isolated": True,
        "user_site_enabled": False, "sys_path": [str(source), str(packages)],
        "modules": modules, "packages": [{"name": name, "version": version,
            "location": str(packages), "record_sha256": "f" * 64} for name, version in pins.items()],
        "torch_cuda": "13.2", "device": device, "driver": {"status": "ready", "device": device},
        "source_lock_sha256": reproduction.digest(source / "environment.lock"),
    }


def _write_fixture_json(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, allow_nan=False) + "\n")


def _comparison_fixture(output, source, op, command):
    import numpy as np
    from bench.distributions import describe, make_lengths, offsets_from_lengths
    destination = output / "comparisons" / op
    destination.mkdir(parents=True)
    lengths = make_lengths("long_tail", 128, seed=711, total_elements=8192, mean_length=64)
    offsets = offsets_from_lengths(lengths, "i64")
    values = np.zeros((8192, 1), dtype=np.float32)
    case = describe(lengths) | {
        "op": op, "dtype": "f32", "D": 1, "offsets_dtype": "i64", "distribution": "long_tail",
        "normalization": "fixed_elements", "seed": 711, "histogram_sha256": None,
        "mean_length_parameter": 64, "values_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
        "offsets_sha256": hashlib.sha256(offsets.tobytes()).hexdigest(), "scale": 1.0, "mask_sha256": None,
        "disabled_optimizations": [], "feature_tile": None,
        "length_bounds": {"min_len": int(lengths.min()), "max_len": int(lengths.max())},
        "values_bytes": values.nbytes, "offsets_bytes": offsets.nbytes, "mask_bytes": 0,
        "input_bytes": values.nbytes + offsets.nbytes,
    }
    methods = list(reproduction.COMMON_METHODS) + (["torch-segment"] if op != "softmax" else [])
    record = _benchmark_record(methods)
    record["case"] = case
    record.update(harness_sha256=reproduction.digest(source / "bench/run.py"),
                  device={"name": "synthetic installed Driver identity fixture", "capability": [8, 9],
                          "total_memory_bytes": 24 << 30, "multiprocessors": 128},
                  cuda_driver={"device_uuid_hex": "000102030405060708090a0b0c0d0e0f", "cuda_supported_version": 13020},
                  torch="2.14.1+cu132", torch_cuda="13.2", measured_method_count=len(methods),
                  command=[str(source / "bench/run.py"), *command[command.index("--") + 1:]])
    for cell in record["methods"]:
        cell["oracle"]["finite_values"] = 8192 if op == "softmax" else 128
        cell["compiler_sha256"] = reproduction.digest(output / "build/bin/ragged-compile")
        for timer in ("complete_call_wall_until_complete_ms", "complete_call_cuda_event_ms"):
            cell[timer] = {"samples": [1.0] * 10, "count": 10, "median": 1.0,
                           "min": 1.0, "max": 1.0, "p90": 1.0}
        cell.update(peak_additional_allocated_bytes=16, allocator_baseline_allocated_bytes=32,
                    allocator_peak_allocated_bytes=48, allocator_baseline_reserved_bytes=64,
                    allocator_peak_reserved_bytes=64, warmup_calls=3,
                    timing_scope="synthetic complete-call fixture; not GPU evidence",
                    allocation_scope="synthetic allocator fixture; not GPU evidence")
        if cell["method"] in ("cuda-naive", "cub"):
            cell["library_sha256"] = reproduction.digest(output / "native/libragged_baselines.so")
            cell["cccl"] = "3.5.0"
        elif cell["method"] == "triton":
            cell.update(triton="3.8.0", source_sha256=reproduction.digest(source / "baselines/triton_reference.py"),
                        ptxas_version="synthetic fixture ptxas V13.2.86")
        elif not cell["method"].startswith("ragged-"):
            cell["source_sha256"] = reproduction.digest(source / "baselines/torch_reference.py")
    np.savez(destination / "case.npz", values=values, offsets=offsets)
    _write_fixture_json(destination / "results.json", record)
    return destination


def _pipeline(tmp_path, monkeypatch, *, fail_stage=None, failure="nonzero", mutate_stage=None,
              mutate=None, sanitizers=False):
    source_root = _minimal_source(tmp_path / "synthetic-checkout")
    if sanitizers:
        suppression = source_root / "test/gpu/driver-load-api-suppressions.xml"
        suppression.parent.mkdir(parents=True, exist_ok=True)
        suppression.write_bytes((Path(__file__).resolve().parents[2] / "test/gpu/driver-load-api-suppressions.xml").read_bytes())
    for relative in ("baselines/cuda/baselines.cu", "baselines/triton_reference.py", "baselines/torch_reference.py",
                     "bench/run.py", "scripts/reproduce_gpu_worker.py"):
        path = source_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic source for orchestration fixture; never executed\n")
    monkeypatch.setattr(reproduction, "ROOT", source_root)
    args = _arguments(tmp_path)
    args.sanitizers = sanitizers
    args.stage_timeout_seconds = 17
    output, calls = args.output, []
    deps = {"llvm": str(tmp_path / "synthetic-sdk"), "cuda": str(tmp_path / "synthetic-cuda"),
            "cccl": str(tmp_path / "synthetic-cccl"), "archive": None, "cmake_prefixes": [],
            "llvm_config": str(tmp_path / "synthetic-sdk/bin/llvm-config"),
            "identity_files": [], "tools": {name: f"/synthetic/tools/{name}" for name in ("python", "cmake", "ninja", "cc", "c++")}}
    monkeypatch.setattr(reproduction, "prerequisites", lambda _: deps)
    uuid = "GPU-00010203-0405-0607-0809-0a0b0c0d0e0f"
    preflight = lambda **kwargs: {"status": "ready", "device": {"uuid": uuid, "capability": [8, 9],
                                                               "name": "synthetic installed Driver identity fixture"}}
    def runner(command, **kwargs):
        name = Path(kwargs["stdout"].name).name.split("-", 1)[1].removesuffix(".log")
        calls.append((name, list(command), kwargs))
        stream = kwargs["stdout"]
        snapshot = output / "source"
        if name == fail_stage:
            stream.write("synthetic failed/timeout child output; not GPU evidence\n")
            if failure == "timeout":
                raise subprocess.TimeoutExpired(command, kwargs["timeout"])
            if failure == "missing_executable":
                raise FileNotFoundError("synthetic missing child executable")
            return SimpleNamespace(returncode=9)
        if name == "python-version":
            stream.write(json.dumps({"version": [3, 12], "full": "synthetic base Python fixture"}) + "\n")
        elif name == "sdk-version":
            stream.write("23.1.3\n")
        else:
            stream.write("synthetic child output; no subprocess or GPU execution\n")
        if name == "create-venv":
            (output / "venv").mkdir()
            (output / "venv/pyvenv.cfg").write_text("include-system-site-packages = false\n")
        elif name in ("install-pinned-tools", "install-pinned-torch"):
            pins = json.loads((snapshot / "environment.lock").read_text())["reproduction"]["packages"]
            selected = {key: value for key, value in pins.items() if (key == "torch") == (name == "install-pinned-torch")}
            record = {"version": "1", "synthetic_host_fixture": True, "install": [
                {"metadata": {"name": key, "version": value}, "download_info": {
                    "url": f"https://synthetic.invalid/{key}.whl", "archive_info": {"hashes": {"sha256": "a" * 64}}}}
                for key, value in selected.items()]}
            _write_fixture_json(output / "results" / ("pip-torch.json" if name.endswith("torch") else "pip-tools.json"), record)
        elif name == "configure-compiler":
            (output / "build").mkdir()
            (output / "build/CMakeCache.txt").write_text("synthetic fresh CMake configuration\n")
        elif name == "build-compiler":
            (output / "build/bin").mkdir()
            for binary in ("ragged-compile", "ragged-opt"):
                (output / "build/bin" / binary).write_text("synthetic fresh compiler binary; not executable\n")
        elif name == "compiler-provenance":
            lock = json.loads((snapshot / "environment.lock").read_text())
            stream.seek(0)
            stream.truncate()
            stream.write(json.dumps({"llvm_commit": lock["required"]["llvm_commit"], "llvm_version": "23.1.3",
                                    "chip": "sm_89", "ptxas_version": "synthetic ptxas V13.2.86"}) + "\n")
        elif name == "build-wheel":
            (output / "wheels").mkdir()
            with zipfile.ZipFile(output / "wheels/ragged_mlir-0.1.0-cp312-cp312-linux_x86_64.whl", "w") as wheel:
                wheel.writestr("ragged/__init__.py", b"synthetic wheel module fixture\n")
                wheel.writestr("ragged/_native.cpython-312-x86_64-linux-gnu.so", b"synthetic installed module fixture; not executable\n")
        elif name == "build-native-baselines":
            (output / "native").mkdir()
            library = output / "native/libragged_baselines.so"
            library.write_text("synthetic foreign native library; not executable\n")
            _write_fixture_json(output / "native/build.json", {"exit_code": 0, "source_unchanged": True,
                "source_sha256": reproduction.digest(snapshot / "baselines/cuda/baselines.cu"),
                "library_sha256": reproduction.digest(library)})
        elif name == "fresh-identity":
            _write_fixture_json(output / "results/fresh-identity.json", _synthetic_identity(output, snapshot, uuid))
        elif name == "sanitizer-collection":
            identity = _synthetic_identity(output, snapshot, uuid)
            nodes = ["test/gpu/test_correctness.py::synthetic_host_execution_contract_fixture", *sanitizer_scope.FAULT_NODEIDS]
            identity["pytest_selection"] = {"collected": len(nodes), "nodeids": nodes, "deselected": []}
            _write_fixture_json(output / "results/sanitizer-collection-identity.json", identity)
        elif name.startswith("sanitizer-"):
            label = name.removeprefix("sanitizer-")
            nodes = [arg for arg in command if arg.startswith("test/gpu/")]
            xml_root = ET.Element("testsuites")
            suite = ET.SubElement(xml_root, "testsuite", name="synthetic-host-fixture-no-GPU")
            for node in nodes:
                case = ET.SubElement(suite, "testcase", name=node.split("::")[-1])
                properties = ET.SubElement(case, "properties")
                ET.SubElement(properties, "property", name="ragged_nodeid", value=node)
            ET.ElementTree(xml_root).write(output / f"results/{label}.xml")
            identity = _synthetic_identity(output, snapshot, uuid)
            identity["pytest_selection"] = {"collected": len(nodes), "nodeids": nodes, "deselected": []}
            _write_fixture_json(output / f"results/{label}-process-identity.json", identity)
        elif name == "gpu-correctness":
            tool = "gpu" if name == "gpu-correctness" else name.removeprefix("sanitizer-")
            nodeid = "synthetic-fixture::synthetic_gpu_execution_contract_fixture"
            (output / f"results/{tool}.xml").write_text(
                '<testsuites><testsuite name="synthetic-host-fixture" tests="1">'
                '<testcase name="synthetic_gpu_execution_contract_fixture"><properties>'
                f'<property name="ragged_nodeid" value="{nodeid}"/>'
                '</properties></testcase></testsuite></testsuites>')
            identity = _synthetic_identity(output, snapshot, uuid)
            identity["pytest_selection"] = {"collected": 1, "nodeids": [nodeid], "deselected": []}
            _write_fixture_json(output / f"results/{tool}-process-identity.json", identity)
        elif name.startswith("comparison-"):
            op = name.removeprefix("comparison-")
            destination = _comparison_fixture(output, snapshot, op, command)
            identity = _synthetic_identity(output, snapshot, uuid)
            identity["benchmark_data_validation"] = {"status": "verified", "case_npz_sha256": reproduction.digest(destination / "case.npz"),
                "result_sha256": reproduction.digest(destination / "results.json"),
                "harness_sha256": reproduction.digest(snapshot / "bench/run.py"),
                "compiler_sha256": reproduction.digest(output / "build/bin/ragged-compile"),
                "native_library_sha256": reproduction.digest(output / "native/libragged_baselines.so"),
                "validated_methods": list(reproduction.COMMON_METHODS) + (["torch-segment"] if op != "softmax" else [])}
            _write_fixture_json(output / f"results/{op}-process-identity.json", identity)
        if name == mutate_stage:
            mutate(output, snapshot, stream)
        stream.flush()
        return SimpleNamespace(returncode=0)
    code = reproduction.run(args, preflight=preflight, runner=runner)
    return output, json.loads((output / "reproduction.json").read_text()), calls, code


def test_complete_synthetic_pipeline_exercises_real_gates_without_installation_or_device_claim(tmp_path, monkeypatch):
    output, report, calls, code = _pipeline(tmp_path, monkeypatch)
    assert code == 0 and report["status"] == "passed"
    assert [name for name, *_ in calls] == list(PIPELINE_STAGES)
    assert all(stage["status"] == "passed" for stage in report["stages"])
    assert report["compiler_sha256"] == reproduction.digest(output / "build/bin/ragged-compile")
    assert report["native_library_sha256"] == reproduction.digest(output / "native/libragged_baselines.so")
    assert len(report["comparisons"]) == 3
    for name, command, options in calls:
        assert options["timeout"] == 17 and options["check"] is False
        assert options["cwd"] == str(output / "source")
        assert "shell" not in options and "PYTHONPATH" not in options["env"]


@pytest.mark.parametrize("stage", PIPELINE_STAGES)
@pytest.mark.parametrize("failure", ["nonzero", "timeout"])
def test_each_stage_failure_or_timeout_stops_once_and_retains_logs(tmp_path, monkeypatch, stage, failure):
    output, report, calls, code = _pipeline(tmp_path, monkeypatch, fail_stage=stage, failure=failure)
    expected_names = list(PIPELINE_STAGES[:PIPELINE_STAGES.index(stage) + 1])
    assert code == 1 and report["status"] == "failed" and report["device_execution_passed"] is False
    assert [name for name, *_ in calls] == expected_names
    failed = report["stages"][-1]
    assert failed["name"] == stage and failed["status"] == "failed"
    log = output / failed["log"]
    assert "synthetic failed/timeout child output" in log.read_text()
    assert failed["log_sha256"] == reproduction.digest(log)
    if failure == "nonzero":
        assert failed["exit_code"] == 9
    else:
        assert "TimeoutExpired" in failed["error"]


def _change_json(path, change):
    value = json.loads(path.read_text())
    change(value)
    _write_fixture_json(path, value)


ARTIFACT_FAULTS = (
    ("create-venv", "system-site"), ("configure-compiler", "missing-cache"),
    ("build-compiler", "missing-compiler"), ("compiler-provenance", "wrong-commit"),
    ("build-wheel", "missing-wheel"), ("build-wheel", "duplicate-wheel"), ("build-wheel", "corrupt-wheel"),
    ("build-native-baselines", "missing-library"), ("build-native-baselines", "wrong-library-hash"),
    ("build-native-baselines", "wrong-source-hash"), ("fresh-identity", "missing-identity"),
    ("fresh-identity", "malformed-json"), ("fresh-identity", "wrong-uuid"),
    ("fresh-identity", "wrong-package-version"), ("fresh-identity", "wrong-module-hash"),
    ("fresh-identity", "installed-native-not-from-built-wheel"),
    ("gpu-correctness", "missing-xml"), ("gpu-correctness", "skipped-xml"),
    ("gpu-correctness", "failed-process-identity"), ("comparison-sum", "missing-npz"),
    ("comparison-sum", "tampered-npz"), ("comparison-max", "wrong-compiler"),
    ("comparison-softmax", "unmeasured-cell"), ("comparison-softmax", "planned-result"),
    ("comparison-softmax", "missing-strict-proof"), ("comparison-softmax", "changed-source"),
)


def _artifact_mutation(fault):
    def mutate(output, source, stream):
        if fault == "system-site":
            (output / "venv/pyvenv.cfg").write_text("include-system-site-packages = true\n")
        elif fault == "missing-cache":
            (output / "build/CMakeCache.txt").unlink()
        elif fault == "missing-compiler":
            (output / "build/bin/ragged-compile").unlink()
        elif fault == "wrong-commit":
            stream.seek(0)
            stream.truncate()
            stream.write(json.dumps({"llvm_commit": "0" * 40, "llvm_version": "23.1.3", "chip": "sm_89", "ptxas_version": "V13.2.86"}))
        elif fault in ("missing-wheel", "duplicate-wheel", "corrupt-wheel"):
            wheel, = (output / "wheels").glob("*.whl")
            if fault == "missing-wheel":
                wheel.unlink()
            elif fault == "duplicate-wheel":
                (output / "wheels/ragged_mlir-duplicate.whl").write_bytes(wheel.read_bytes())
            else:
                wheel.write_text("synthetic corrupt wheel archive\n")
        elif fault == "missing-library":
            (output / "native/libragged_baselines.so").unlink()
        elif fault in ("wrong-library-hash", "wrong-source-hash"):
            key = "library_sha256" if fault == "wrong-library-hash" else "source_sha256"
            _change_json(output / "native/build.json", lambda record: record.update({key: "0" * 64}))
        elif fault == "missing-identity":
            (output / "results/fresh-identity.json").unlink()
        elif fault == "malformed-json":
            (output / "results/fresh-identity.json").write_text("not JSON\n")
        elif fault == "wrong-uuid":
            _change_json(output / "results/fresh-identity.json", lambda record: record["device"].update(uuid="GPU-ffffffff-ffff-ffff-ffff-ffffffffffff"))
        elif fault == "wrong-package-version":
            _change_json(output / "results/fresh-identity.json", lambda record: record["packages"][0].update(version="9.9.9"))
        elif fault == "wrong-module-hash":
            _change_json(output / "results/fresh-identity.json", lambda record: record["modules"]["ragged._native"].update(sha256="0" * 64))
        elif fault == "installed-native-not-from-built-wheel":
            path = output / "results/fresh-identity.json"
            record = json.loads(path.read_text())
            module = Path(record["modules"]["ragged._native"]["file"])
            module.write_text("synthetic different installed native module, not the freshly built wheel\n")
            record["modules"]["ragged._native"]["sha256"] = reproduction.digest(module)
            _write_fixture_json(path, record)
        elif fault == "missing-xml":
            (output / "results/gpu.xml").unlink()
        elif fault == "skipped-xml":
            (output / "results/gpu.xml").write_text('<testsuites><testsuite><testcase><skipped/></testcase></testsuite></testsuites>')
        elif fault == "failed-process-identity":
            _change_json(output / "results/gpu-process-identity.json", lambda record: record.update(status="failed"))
        elif fault in ("missing-npz", "tampered-npz"):
            path = output / "comparisons/sum/case.npz"
            if fault == "missing-npz":
                path.unlink()
            else:
                path.write_bytes(path.read_bytes() + b"synthetic NPZ changed after worker validation")
        elif fault == "wrong-compiler":
            _change_json(output / "comparisons/max/results.json", lambda record: record["methods"][0].update(compiler_sha256="0" * 64))
        elif fault == "unmeasured-cell":
            _change_json(output / "comparisons/softmax/results.json", lambda record: record["methods"][0].update(status="unmeasured"))
        elif fault == "planned-result":
            _change_json(output / "comparisons/softmax/results.json", lambda record: record.update(status="planned"))
        elif fault == "missing-strict-proof":
            _change_json(output / "results/softmax-process-identity.json", lambda record: record.pop("benchmark_data_validation"))
        elif fault == "changed-source":
            (source / "baselines/torch_reference.py").write_text("synthetic changed source after snapshot\n")
        else:
            raise AssertionError(f"unknown synthetic artifact fault: {fault}")
    return mutate


@pytest.mark.parametrize("stage,fault", ARTIFACT_FAULTS)
def test_missing_changed_or_invalid_artifacts_never_pass_reproduction(tmp_path, monkeypatch, stage, fault):
    _, report, calls, code = _pipeline(tmp_path, monkeypatch, mutate_stage=stage, mutate=_artifact_mutation(fault))
    assert code == 1 and report["status"] == "failed" and report["device_execution_passed"] is False
    assert [name for name, *_ in calls] == list(PIPELINE_STAGES[:PIPELINE_STAGES.index(stage) + 1])


def test_sanitizer_option_reaches_all_three_tools_without_narrowing_test_selection(tmp_path, monkeypatch):
    output, report, calls, code = _pipeline(tmp_path, monkeypatch, sanitizers=True)
    assert code == 0 and report["status"] == "passed"
    labels = [f"sanitizer-{tool}-{group}" for tool in ("memcheck", "racecheck", "synccheck")
              for group in sanitizer_scope.GROUPS]
    assert [name for name, *_ in calls][-10:] == ["sanitizer-collection", *labels]
    assert set(report["sanitizers"]) == {"memcheck", "racecheck", "synccheck"}
    collection_command = calls[-10][1]
    assert "--collect-only" in collection_command
    assert "--ignore=test/gpu/test_fbgemm_baseline.py" in collection_command
    assert not any(arg.startswith("--basetemp=") for arg in collection_command)
    temps = []
    for name, command, _ in calls[-9:]:
        assert command[command.index("--error-exitcode") + 1] == "86"
        assert command[command.index("--target-processes") + 1] == "all"
        assert "test/gpu" not in command and "-k" not in command
        group = next(group for group in sanitizer_scope.GROUPS if name.endswith("-" + group))
        assert [arg for arg in command if arg.startswith("test/gpu/")] == report["sanitizer_scope"]["groups"][group]
        if group == "loader_api":
            assert command.count("--suppressions") == 1
            assert command[command.index("--suppressions") + 1] == str(output / "source/test/gpu/driver-load-api-suppressions.xml")
        else:
            assert not any("suppress" in arg for arg in command)
        temp = str(output / "work" / name.removeprefix("sanitizer-"))
        assert f"--basetemp={temp}" in command and Path(temp).resolve().is_relative_to(output.resolve())
        temps.append(temp)
    assert len(temps) == len(set(temps)) == 9
    for tool in report["sanitizers"].values():
        seen = [node for group in tool.values() for node in group["nodeids"]]
        assert len(seen) == len(set(seen)) == 7
        assert all(group["junit"]["skipped"] == 0 for group in tool.values())


@pytest.mark.parametrize("tool", ["memcheck", "racecheck", "synccheck"])
def test_sanitizer_skips_cannot_pass_even_when_process_exits_zero(tmp_path, monkeypatch, tool):
    def skip(output, source, stream):
        xml = output / f"results/{tool}-loader_api.xml"
        tree = ET.parse(xml)
        ET.SubElement(tree.getroot()[0][0], "skipped")
        tree.write(xml)
    _, report, calls, code = _pipeline(tmp_path, monkeypatch, sanitizers=True,
                                    mutate_stage=f"sanitizer-{tool}-loader_api", mutate=skip)
    assert code == 1 and report["status"] == "failed" and report["device_execution_passed"] is False
    assert calls[-1][0] == f"sanitizer-{tool}-loader_api"


@pytest.mark.parametrize("mutation", ["outside-executable", "not-a-venv", "leaked-sys-path", "contradictory-driver"])
def test_identity_verified_label_cannot_hide_environment_contradiction(tmp_path, mutation):
    source = _minimal_source(tmp_path / "source")
    output = tmp_path / "output"
    output.mkdir()
    uuid = "GPU-00010203-0405-0607-0809-0a0b0c0d0e0f"
    record = _synthetic_identity(output, source, uuid)
    if mutation == "outside-executable":
        record["executable"] = "/synthetic/outside-venv/python"
    elif mutation == "not-a-venv":
        record["base_prefix"] = record["prefix"]
    elif mutation == "leaked-sys-path":
        record["sys_path"].append(str(source / "python"))
    else:
        record["driver"]["device"]["uuid"] = "GPU-ffffffff-ffff-ffff-ffff-ffffffffffff"
        # The nested driver dict shares the device in the fixture; make the
        # reported device independently correct to expose the contradiction.
        record["device"] = {"uuid": uuid, "capability": [8, 9]}
    path = tmp_path / "synthetic-identity.json"
    _write_fixture_json(path, record)
    pins = json.loads((source / "environment.lock").read_text())["reproduction"]["packages"]
    with pytest.raises(reproduction.Failed):
        reproduction.validate_identity(path, output, uuid, pins, reproduction.digest(source / "environment.lock"))


def _pip_fixture():
    return {"install": [{"metadata": {"name": "torch", "version": "2.14.1+cu132"},
                         "download_info": {"url": "https://synthetic.invalid/torch.whl",
                                           "archive_info": {"hashes": {"sha256": "a" * 64}}}}]}


@pytest.mark.parametrize("mutation", ["no-install", "no-sha", "bad-sha", "http-url", "missing-version", "cpu-torch"])
def test_pip_installation_report_requires_version_source_hash_and_pinned_cuda_torch(tmp_path, mutation):
    record = _pip_fixture()
    item = record["install"][0]
    if mutation == "no-install":
        record["install"] = []
    elif mutation == "no-sha":
        item["download_info"]["archive_info"]["hashes"] = {}
    elif mutation == "bad-sha":
        item["download_info"]["archive_info"]["hashes"]["sha256"] = "not-a-hash"
    elif mutation == "http-url":
        item["download_info"]["url"] = "http://synthetic.invalid/torch.whl"
    elif mutation == "missing-version":
        item["metadata"].pop("version")
    else:
        item["metadata"]["version"] = "2.14.1+cpu"
    path = tmp_path / "synthetic-pip.json"
    _write_fixture_json(path, record)
    with pytest.raises(reproduction.Failed):
        reproduction.validate_pip_report(path, require_torch=True)


@pytest.mark.parametrize("validator", ["identity", "pip", "benchmark"])
@pytest.mark.parametrize("malformation", ["duplicate", "nan", "overflow"])
def test_all_child_json_ingestion_rejects_duplicate_and_nonfinite_fields(tmp_path, validator, malformation):
    source = _minimal_source(tmp_path / "source")
    output = tmp_path / "output"
    output.mkdir()
    uuid = "GPU-00010203-0405-0607-0809-0a0b0c0d0e0f"
    if validator == "identity":
        record = _synthetic_identity(output, source, uuid)
        pins = json.loads((source / "environment.lock").read_text())["reproduction"]["packages"]
        inspect = lambda path: reproduction.validate_identity(path, output, uuid, pins, reproduction.digest(source / "environment.lock"))
    elif validator == "pip":
        record = _pip_fixture()
        inspect = lambda path: reproduction.validate_pip_report(path, require_torch=True)
    else:
        record = _benchmark_record()
        inspect = lambda path: reproduction.validate_benchmark(path, "sum", ("ragged-auto", "triton"), "c" * 64)
    if malformation == "duplicate":
        raw = '{"synthetic_duplicate":1,"synthetic_duplicate":2,' + json.dumps(record)[1:]
    else:
        raw = '{"synthetic_invalid_number":' + ("NaN" if malformation == "nan" else "1e999") + ',' + json.dumps(record)[1:]
    path = tmp_path / "synthetic-malformed-child.json"
    path.write_text(raw)
    with pytest.raises((reproduction.Failed, ValueError)):
        inspect(path)


@pytest.mark.parametrize("fault", ["wrong-collected-count", "duplicate-nodeids", "deselection"])
def test_gpu_xml_must_match_actual_undeselected_worker_collection(tmp_path, monkeypatch, fault):
    def mutate(output, source, stream):
        path = output / "results/gpu-process-identity.json"
        record = json.loads(path.read_text())
        selection = record["pytest_selection"]
        if fault == "wrong-collected-count":
            selection["collected"] = 2
        elif fault == "duplicate-nodeids":
            selection["nodeids"] *= 2
        else:
            selection["deselected"] = ["synthetic skipped collection node"]
        _write_fixture_json(path, record)
    _, report, calls, code = _pipeline(tmp_path, monkeypatch, mutate_stage="gpu-correctness", mutate=mutate)
    assert code == 1 and report["status"] == "failed" and report["device_execution_passed"] is False
    assert calls[-1][0] == "gpu-correctness"


def test_three_operator_comparisons_must_retain_same_inputs_not_only_valid_individual_cells(tmp_path, monkeypatch):
    import numpy as np
    def mutate(output, source, stream):
        destination = output / "comparisons/max"
        with np.load(destination / "case.npz", allow_pickle=False) as archive:
            values, offsets = archive["values"], archive["offsets"]
        values[0, 0] = 1.0
        np.savez(destination / "case.npz", values=values, offsets=offsets)
        _change_json(destination / "results.json", lambda record: record["case"].update(
            values_sha256=hashlib.sha256(values.tobytes()).hexdigest()))
        path = output / "results/max-process-identity.json"
        _change_json(path, lambda record: record["benchmark_data_validation"].update(
            case_npz_sha256=reproduction.digest(destination / "case.npz"),
            result_sha256=reproduction.digest(destination / "results.json")))
    _, report, calls, code = _pipeline(tmp_path, monkeypatch, mutate_stage="comparison-max", mutate=mutate)
    assert code == 1 and report["status"] == "failed" and report["device_execution_passed"] is False
    assert [name for name, *_ in calls] == list(PIPELINE_STAGES)
    assert "same generated input" in report["reason"]
