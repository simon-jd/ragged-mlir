#!/usr/bin/env python3
"""Actual AOT execution worker; never fabricates device or timing results."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.reproduce_gpu import digest, read_json, write_json, driver_preflight
from scripts.sparse_comparison_support import (SparseLibrary, case_identity, candidate_cuda_callable,
    measure, callback_identity, DOMAIN, CANDIDATE_COSTS, CUB_COSTS)


def file_identity(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": digest(path)}


def load_case(record):
    import numpy as np
    path = Path(record["path"])
    if digest(path) != record["sha256"]:
        raise ValueError("input archive changed after preparation")
    with np.load(path, allow_pickle=False) as data:
        values, offsets, coords = (data[key] for key in ("values", "offsets", "coordinates"))
    if case_identity(values, offsets, coords) != record["identity"]:
        raise ValueError("input contents differ from their recorded identity")
    return values, offsets, coords


def execute(config, mode, output):
    import numpy as np
    from oracle import assert_matches
    output.mkdir()
    report = {"schema": 1, "mode": mode, "backend": "upstream-sparse-gpu-hybrid" if mode == "gpu" else "upstream-sparse-cpu-AOT",
              "source": config["source"], "inputs": config["inputs"], "modules": {}, "cases": {},
              "actual_device_execution": mode == "gpu", "status": "failed"}
    report["host_environment"] = {"python": file_identity(sys.executable), "python_version": sys.version,
                                  "numpy_version": np.__version__, "numpy_module": file_identity(np.__file__)}
    libraries = {}
    native = None
    if mode == "gpu":
        import torch
        from baselines.native import NativeBaselines
        observed = driver_preflight(device=0)
        if observed.get("status") != "ready" or observed["device"] != config["driver"]["device"]:
            raise RuntimeError("worker Driver device differs from selected sm89 identity")
        if (torch.__version__ != "2.14.1+cu132" or torch.version.cuda != "13.2" or not torch.cuda.is_available()
                or tuple(torch.cuda.get_device_capability(0)) != (8, 9)):
            raise RuntimeError("requires actual Torch 2.14.1+cu132 on sm89")
        torch.cuda.set_device(0)
        started = time.perf_counter_ns()
        native = NativeBaselines(config["cub_library"]["path"])
        report.update(driver=observed, framework={"version": torch.__version__, "cuda": torch.version.cuda,
                      "module": file_identity(torch.__file__)}, cub_library=config["cub_library"],
                      cub_provenance=native.provenance, cub_load_wall_ms=(time.perf_counter_ns()-started)/1e6,
                      domain=DOMAIN, warmup=config["warmup"], repeats=config["repeats"],
                      all_complete_costs_included=True, input=config["inputs"]["long_tail"]["identity"],
                      cpu_oracle=config["cpu_oracle"], operations={})
        if ({key: config["cub_build"][key] for key in ("path", "sha256")} != file_identity(config["cub_build"]["path"])
                or read_json(config["cub_build"]["path"]) != config["cub_build"]["manifest"]
                or native.provenance.get("build") != config["cub_build"]["manifest"]):
            raise ValueError("loaded CUB library differs from its qualified build provenance")
        report["cub_build"] = config["cub_build"]
        report["cub_callbacks"] = {symbol: callback_identity(getattr(native.lib, symbol), native.path)
                                   for symbol in ("ragged_cub_reduce", "ragged_cub_transform")}
        if file_identity(config["cub_library"]["path"]) != config["cub_library"]:
            raise ValueError("CUB native library changed")
    # Only load CUDA-linked AOT objects after this process has independently
    # verified Driver, Torch and selected device. Host mode never loads them.
    for name, module in config["modules"].items():
        item = module[mode + "_library"]
        if file_identity(item["path"]) != item:
            raise ValueError("AOT shared library changed")
        libraries[name] = SparseLibrary(item["path"], module["B"], module["width"])
        report["modules"][name] = {**libraries[name].provenance, "load_wall_ms": libraries[name].load_wall_ms}
        if mode == "gpu":
            report["modules"][name]["cuda_runtime_callback"] = callback_identity(
                libraries[name].library.mgpuLaunchKernel, Path(config["gpu_runtime"]["path"]).resolve())
    passed = True
    for name, record in config["inputs"].items():
        values, offsets, _ = load_case(record)
        library = libraries["long_tail" if name == "long_tail" else "semantics"]
        report["cases"][name] = {}
        if mode == "gpu":
            gpu_values = torch.from_numpy(values).to("cuda")
            gpu_offsets = torch.from_numpy(offsets).to("cuda")
            torch.cuda.synchronize()
            candidate = candidate_cuda_callable(library, gpu_values, gpu_offsets)
        for op in ("max", "softmax"):
            cells = {}
            report["cases"][name][op] = cells
            if mode == "gpu" and name == "long_tail":
                report["operations"][op] = cells
            methods = ("candidate", "cub") if mode == "gpu" else ("candidate",)
            for method in methods:
                cell = {"status": "failed", "oracle_passed": False, "input": record["identity"]}
                cells[method] = cell
                try:
                    if mode == "cpu":
                        actual, audit = library.call(op, values, offsets, audit=True)
                    elif method == "candidate":
                        # Separate structure qualification executes the same official AOT
                        # algorithm; the timed resident callable still rebuilds everything.
                        _, audit = library.call(op, values, offsets, audit=True)
                        actual_gpu = candidate(op)
                        torch.cuda.synchronize()
                        actual = actual_gpu.cpu().numpy()
                    else:
                        audit = None
                        actual_gpu = native.call("cub", gpu_values, gpu_offsets, op)
                        torch.cuda.synchronize()
                        actual = actual_gpu.cpu().numpy()
                    raw = output / f"{name}-{op}-{method}.npy"
                    np.save(raw, actual, allow_pickle=False)
                    cell.update(output=file_identity(raw), structure=audit)
                    assert_matches(actual, values, offsets, operation=op, input_dtype="f32")
                    cell.update(status="qualified", oracle_passed=True)
                    if mode == "gpu" and name == "long_tail":
                        cell.update(domain=DOMAIN, costs=list(CANDIDATE_COSTS if method == "candidate" else CUB_COSTS))
                        if config["cpu_oracle"][op] is True:
                            call = (lambda op=op: candidate(op)) if method == "candidate" else (lambda op=op: native.call("cub", gpu_values, gpu_offsets, op))
                            cell["timing"] = measure(call, torch.cuda.synchronize, config["warmup"], config["repeats"])
                            cell["status"] = "measured"
                        else:
                            cell["reason"] = "CPU raw-output qualification failed; no hot timing"
                except Exception as error:
                    cell.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
                    passed = False
                # Retain each operation even if a later operation fails.
                write_json(output / "execution.json", report)
    report["status"] = "complete" if passed else "failed"
    write_json(output / "execution.json", report)
    return 0 if passed else 1


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("cpu", "gpu"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    return execute(read_json(args.config), args.mode, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
