#!/usr/bin/env python3
"""Isolated fresh-venv worker; no hardware/dependency bypass CLI exists."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import site
import sys
import time


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read_json(path):
    from scripts.reproduce_gpu import read_json as strict_read_json
    return strict_read_json(path)


def module_identity(name, venv):
    module = importlib.import_module(name)
    path = Path(module.__file__).resolve()
    if not path.is_relative_to(venv):
        raise RuntimeError(f"{name} was not imported from the fresh venv: {path}")
    return {"file": str(path), "sha256": file_hash(path)}


def identity(source, venv, expected_uuid):
    """Called in the actual test/benchmark process, including pytest startup."""
    if Path(sys.prefix).resolve() != venv or sys.prefix == sys.base_prefix:
        raise RuntimeError("worker must run inside the requested fresh virtual environment")
    if not sys.flags.isolated or site.ENABLE_USER_SITE:
        raise RuntimeError("worker requires python -I and disabled user site")
    if "PYTHONPATH" in __import__("os").environ or "PYTHONHOME" in __import__("os").environ:
        raise RuntimeError("inherited Python import overrides are forbidden")
    for entry in sys.path:
        path = Path(entry or ".").resolve()
        if path == source / "python" or ("site-packages" in path.parts and not path.is_relative_to(venv)):
            raise RuntimeError(f"unexpected source/system package search path: {path}")
    lock = read_json(source / "environment.lock")
    packages = lock["reproduction"]["packages"]
    for name, expected in packages.items():
        observed = importlib.metadata.version(name)
        if observed != expected:
            raise RuntimeError(f"{name} must be exactly {expected}, found {observed}")
    modules = {name: module_identity(name, venv) for name in ("ragged", "ragged._native", "torch", "torch._C", "triton", "numpy", "pytest")}
    torch = importlib.import_module("torch")
    if str(torch.__version__) != packages["torch"] or torch.version.cuda != "13.2":
        raise RuntimeError("Torch must be the pinned cu132 build, not a CPU or alternate CUDA wheel")
    if not torch.cuda.is_available() or tuple(torch.cuda.get_device_capability()) != (8, 9):
        raise RuntimeError("fresh Torch cannot execute on the required sm_89 device")
    from scripts.reproduce_gpu import driver_preflight
    driver = driver_preflight(device=0)
    if driver.get("status") != "ready" or driver.get("device", {}).get("uuid") != expected_uuid:
        raise RuntimeError("fresh process did not retain the selected Driver device UUID")
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    distributions = []
    for distribution in importlib.metadata.distributions():
        location = Path(distribution.locate_file("")).resolve()
        if not location.is_relative_to(venv):
            raise RuntimeError(f"installed distribution leaked from outside venv: {location}")
        record = distribution.read_text("RECORD")
        distributions.append({"name": distribution.metadata["Name"], "version": distribution.version,
                              "location": str(location), "record_sha256": hashlib.sha256(record.encode()).hexdigest() if record else None})
    return {"schema": 1, "status": "verified", "timestamp_unix_ns": time.time_ns(),
            "python": sys.version, "executable": sys.executable, "prefix": sys.prefix, "base_prefix": sys.base_prefix,
            "isolated": bool(sys.flags.isolated), "user_site_enabled": site.ENABLE_USER_SITE,
            "sys_path": sys.path.copy(), "modules": modules, "packages": sorted(distributions, key=lambda item: item["name"].lower()),
            "torch": str(torch.__version__), "torch_cuda": torch.version.cuda, "device": driver["device"], "driver": driver,
            "torch_device": {"name": properties.name, "capability": [properties.major, properties.minor],
                             "total_memory_bytes": properties.total_memory, "multiprocessors": properties.multi_processor_count},
            "source_lock_sha256": file_hash(source / "environment.lock")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("identity", "pytest", "benchmark"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--venv", type=Path, required=True)
    parser.add_argument("--expected-uuid", required=True)
    parser.add_argument("--identity-output", type=Path, required=True)
    arguments, remainder = parser.parse_known_args(argv)
    if remainder and remainder[0] == "--":
        remainder = remainder[1:]
    source, venv = arguments.source.resolve(), arguments.venv.resolve()
    # Explicit import root supplies bench/oracle/tests, but does not expose
    # snapshot/python/ragged. That module must come from the installed wheel.
    sys.path.insert(0, str(source))
    output = arguments.identity_output
    def record_identity(scope):
        record = identity(source, venv, arguments.expected_uuid)
        record["scope"] = scope
        output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        return record
    try:
        record_identity("actual " + arguments.action + " worker before execution")
        if arguments.action == "identity":
            if remainder:
                raise RuntimeError("identity action takes no forwarded arguments")
            return 0
        if arguments.action == "pytest":
            import pytest
            class InstalledWheelGate:
                deselected = []
                @pytest.hookimpl(trylast=True)
                def pytest_sessionstart(self, session):
                    record_identity("actual pytest session after configuration; installed wheel gate")
                def pytest_deselected(self, items):
                    self.deselected.extend(item.nodeid for item in items)
                def pytest_collection_finish(self, session):
                    for item in session.items:
                        if any(name == "ragged_nodeid" for name, _ in item.user_properties):
                            raise RuntimeError("reserved ragged_nodeid JUnit property is already present")
                        item.user_properties.append(("ragged_nodeid", item.nodeid))
                    record = read_json(output)
                    record["pytest_selection"] = {"collected": len(session.items),
                        "nodeids": [item.nodeid for item in session.items], "deselected": self.deselected}
                    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
            return int(pytest.main(remainder, plugins=[InstalledWheelGate()]))
        if arguments.action == "benchmark":
            # Invoke the unchanged benchmark in this same verified process.
            from bench.run import main as benchmark_main
            sys.argv = [str(source / "bench/run.py"), *remainder]
            try:
                benchmark_main()
            except SystemExit as exit_status:
                if exit_status.code != 0:
                    raise
                from bench.report import _json, _identity, _input_artifact, _measured
                from scripts.reproduce_gpu import COMMON_METHODS
                destination = Path(remainder[remainder.index("--output") + 1])
                result = _json(destination / "results.json")
                record = read_json(output)
                op = remainder[remainder.index("--op") + 1]
                if remainder.count("--repeats") != 1 or remainder.count("--warmup") != 1:
                    raise RuntimeError("benchmark needs unambiguous explicit repeats and warmup counts")
                repeats = int(remainder[remainder.index("--repeats") + 1])
                warmup = int(remainder[remainder.index("--warmup") + 1])
                if op not in ("sum", "max", "softmax") or result.get("command") != [str(source / "bench/run.py"), *remainder]:
                    raise RuntimeError("benchmark command identity differs from the actual worker invocation")
                planned = {"op": op, "dtype": "f32", "D": 1, "offsets_dtype": "i64", "distribution": "long_tail",
                           "normalization": "fixed_elements", "segments": 128, "elements": 8192, "seed": 711,
                           "mean_length": 64, "scale": 1.0, "mask": False, "length_hints": True}
                preflight = {"status": "ready", "device": record["torch_device"], "torch": record["torch"], "torch_cuda": record["torch_cuda"],
                             "cuda_driver": {"device_uuid_hex": arguments.expected_uuid.removeprefix("GPU-").replace("-", ""),
                                             "cuda_supported_version": record["driver"]["driver_supported_cuda_version"]}}
                _identity(result.get("case"), planned, result,
                          {"benchmark_harness_sha256": file_hash(source / "bench/run.py"), "preflight": preflight})
                case_hash = _input_artifact(destination / "case.npz", result["case"])
                methods = COMMON_METHODS + (("torch-segment",) if op != "softmax" else ())
                compiler_hash = file_hash(venv.parent / "build/bin/ragged-compile")
                native_hash = file_hash(venv.parent / "native/libragged_baselines.so")
                if result.get("status") != "complete" or [cell.get("method") for cell in result.get("methods", [])] != list(methods):
                    raise RuntimeError("benchmark omitted or reordered its required comparison methods")
                for cell in result["methods"]:
                    if cell.get("status") != "measured":
                        raise RuntimeError("a required baseline was not oracle-qualified and measured")
                    _measured(cell)
                    if (type(cell.get("warmup_calls")) is not int or cell["warmup_calls"] != warmup
                            or any(type(cell[timer].get("count")) is not int or cell[timer]["count"] != repeats
                                   for timer in ("complete_call_cuda_event_ms", "complete_call_wall_until_complete_ms"))):
                        raise RuntimeError("benchmark samples or warmup differ from the actual requested counts")
                    if cell["method"].startswith("ragged-") and cell.get("compiler_sha256") != compiler_hash:
                        raise RuntimeError("Ragged benchmark did not execute the fresh compiler")
                    if cell["method"] in ("cuda-naive", "cub") and cell.get("library_sha256") != native_hash:
                        raise RuntimeError("native benchmark did not execute the newly built library")
                    if cell["method"] == "triton" and cell.get("source_sha256") != file_hash(source / "baselines/triton_reference.py"):
                        raise RuntimeError("Triton benchmark did not use the snapshot adapter")
                    if cell["method"] in ("torch-segment", "njt-eager", "njt-compile") and cell.get("source_sha256") != file_hash(source / "baselines/torch_reference.py"):
                        raise RuntimeError("Torch/NJT benchmark did not use the snapshot adapter")
                record["benchmark_data_validation"] = {"status": "verified", "case_npz_sha256": case_hash,
                    "result_sha256": file_hash(destination / "results.json"), "harness_sha256": file_hash(source / "bench/run.py"),
                    "compiler_sha256": compiler_hash, "native_library_sha256": native_hash, "validated_methods": list(methods)}
                output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
                return 0
            raise RuntimeError("benchmark main unexpectedly returned without its explicit exit status")
    except Exception as error:
        output.write_text(json.dumps({"schema": 1, "status": "failed", "action": arguments.action,
                                      "error": f"{type(error).__name__}: {error}"}, indent=2) + "\n")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
