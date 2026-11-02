"""Plan or execute reproducible complete-call benchmark cases.

Each case calls bench.run once with all supported requested methods, so those
methods share its saved inputs and oracle. Input-size labels are byte ceilings,
not claims about the GPU L2. Fixed-B groups share B and the synthetic mean-length
parameter across distributions; realized N remains distribution-dependent.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import struct
import sys
from typing import Any

from bench.distributions import DISTRIBUTIONS, make_lengths
from bench.public_histogram import sample_histogram
from bench.run import METHODS as RUN_METHODS

ROOT = Path(__file__).resolve().parents[1]
PENDING_METHODS = {
    "sparse-gpu": "hybrid sparse GPU adapter and the sparse-vs-CUB scope decision are pending; no pure sparse GPU baseline is claimed",
}
METHODS = (*RUN_METHODS, *PENDING_METHODS)
NORMALIZATIONS = ("fixed_elements", "fixed_segments")
OPTIMIZATION_NAMES = ("classification", "online_fusion", "warp_mapping", "split_mapping", "shuffle", "pointwise_fusion",
                      "row_pipelining", "vectorize")
PRESETS = {
    "full": {"ops": ["sum", "max", "softmax"], "dtypes": ["f32", "bf16"], "ds": [1, 16, 128, 1024],
             "offsets_dtypes": ["i32", "i64"], "distributions": [*DISTRIBUTIONS, "public_histogram"],
             "normalizations": list(NORMALIZATIONS), "sizes": ["small", "large"]},
    "smoke": {"ops": ["sum", "softmax"], "dtypes": ["f32"], "ds": [16], "offsets_dtypes": ["i64"],
              "distributions": ["long_tail"], "normalizations": list(NORMALIZATIONS), "sizes": ["small"]},
}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--preset", choices=PRESETS, default="smoke")
    result.add_argument("--plan-only", action="store_true", help="write commands and planned cells without a GPU preflight")
    result.add_argument("--ops", nargs="+", choices=("sum", "max", "softmax"))
    result.add_argument("--dtypes", nargs="+", choices=("f32", "bf16"))
    result.add_argument("--ds", nargs="+", type=int, choices=(1, 16, 128, 1024))
    result.add_argument("--offsets-dtypes", nargs="+", choices=("i32", "i64"))
    result.add_argument("--distributions", nargs="+", choices=(*DISTRIBUTIONS, "public_histogram"))
    result.add_argument("--normalizations", nargs="+", choices=NORMALIZATIONS)
    result.add_argument("--sizes", nargs="+", choices=("small", "large"))
    result.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    result.add_argument("--small-bytes", type=int, default=1 << 20)
    result.add_argument("--large-bytes", type=int, default=128 << 20)
    result.add_argument("--mean-length", type=int, default=64, help="upper bound for the shared synthetic fixed-B scale")
    result.add_argument("--seed", type=int, default=711)
    result.add_argument("--histogram", type=Path, default=ROOT / "bench/data/uci-online-retail.json")
    result.add_argument("--compiler", default="build/bin/ragged-compile")
    result.add_argument("--native-library", default="build-baselines/libragged_baselines.so")
    result.add_argument("--python", default=sys.executable)
    result.add_argument("--warmup", type=int, default=5)
    result.add_argument("--repeats", type=int, default=30)
    result.add_argument("--scale", type=float, default=1.0)
    result.add_argument("--mask", action="store_true", help="dense f32 additive mask; requires exclusively softmax ops")
    result.add_argument("--disable-optimization", action="append", choices=OPTIMIZATION_NAMES, default=[])
    result.add_argument("--feature-tile", type=int, choices=(1, 2, 4, 8, 16, 32))
    result.add_argument("--length-hints", action="store_true")
    result.add_argument("--timeout", type=float, default=3600, help="per-case subprocess timeout in seconds; no automatic retry")
    return result


def _length_total(distribution: str, b: int, mean: int, args) -> int:
    lengths = (sample_histogram(args.histogram, b, seed=args.seed, total_elements=None)
               if distribution == "public_histogram" else
               make_lengths(distribution, b, seed=args.seed, mean_length=mean))
    return sum(map(int, lengths))


def _sizing(dtype: str, d: int, offsets_dtype: str, budget: int, normalization: str,
            distributions: list[str], args) -> dict[str, dict]:
    value_row_bytes = d * (4 if dtype == "f32" else 2)
    mask_row_bytes = 4 * d if args.mask else 0
    row_bytes, offset_bytes = value_row_bytes + mask_row_bytes, (4 if offsets_dtype == "i32" else 8)
    if normalization == "fixed_elements":
        n = max(1, (budget - offset_bytes) * args.mean_length // (row_bytes * args.mean_length + offset_bytes))
        b, mean = max(1, n // args.mean_length), args.mean_length
        requested_b = b
        totals = {distribution: n for distribution in distributions}
    else:
        target_n = max(1, (budget - 2 * offset_bytes) // row_bytes)
        # The native burst has a 16384x weight even at B=1. Keep a common
        # scale, reduce the common B if needed, and explicitly block impossible
        # cells instead of allocating a fixed huge N at large D.
        mean = max(1, min(args.mean_length, target_n // 32768))
        requested_b = max(1, target_n // args.mean_length)
        b = max(1, min(requested_b, 100000, target_n // (mean * 200)))
        while True:
            totals = {distribution: _length_total(distribution, b, mean, args) for distribution in distributions}
            if b == 1 or all(n * row_bytes + (b + 1) * offset_bytes <= budget for n in totals.values()):
                break
            b = max(1, b // 2)
    result = {}
    for distribution, n in totals.items():
        value_bytes, mask_bytes, offsets_bytes = n * value_row_bytes, n * mask_row_bytes, (b + 1) * offset_bytes
        total_bytes = value_bytes + mask_bytes + offsets_bytes
        reason = None
        if total_bytes > budget:
            reason = "size_budget: minimum generated input cannot fit this input-byte ceiling"
        if n > (2**31 - 1 if offsets_dtype == "i32" else 2**63 - 1):
            reason = "offsets_range: realized N does not fit the requested offsets dtype"
        adjustments = []
        if mean != args.mean_length:
            adjustments.append("common synthetic mean reduced to limit the native burst input")
        if b != requested_b:
            adjustments.append("common B reduced using the byte ceiling, planning cap and sampled lengths")
        result[distribution] = {"segments": b, "elements": n, "requested_b": requested_b, "resolved_b": b,
            "total_elements": n if normalization == "fixed_elements" else None,
            "mean_length": mean, "requested_mean_length": args.mean_length,
            "input_byte_budget": budget, "input_bytes": total_bytes, "values_bytes": value_bytes,
            "mask_bytes": mask_bytes, "offsets_bytes": offsets_bytes,
            "size_budget_reason": reason, "sizing_adjustments": adjustments}
    return result


def _command(args, case: dict, directory: Path) -> list[str]:
    methods = [method for method in args.methods if method in RUN_METHODS]
    if not methods:
        return []
    command = [args.python, "-m", "bench.run", "--op", case["op"], "--dtype", case["dtype"],
        "--d", str(case["D"]), "--offsets-dtype", case["offsets_dtype"],
        "--segments", str(case["segments"]), "--mean-length", str(case["mean_length"]),
        "--seed", str(case["seed"]), "--compiler", str(Path(args.compiler).resolve()),
        "--native-library", str(Path(args.native_library).resolve()), "--warmup", str(args.warmup),
        "--repeats", str(args.repeats), "--scale", str(args.scale), "--output", str(directory), "--methods", *methods]
    if args.mask:
        command.append("--mask")
    if args.length_hints:
        command.append("--length-hints")
    if args.feature_tile is not None:
        command.extend(("--feature-tile", str(args.feature_tile)))
    for name in args.disable_optimization:
        command.extend(("--disable-optimization", name))
    if case["distribution"] == "public_histogram":
        command.extend(("--histogram", str(args.histogram)))
    else:
        command.extend(("--distribution", case["distribution"]))
    command.extend(("--fixed-segments",) if case["normalization"] == "fixed_segments"
                   else ("--total-elements", str(case["elements"])))
    return command


def plan(args) -> dict[str, Any]:
    for field in ("small_bytes", "large_bytes", "mean_length", "repeats"):
        if type(getattr(args, field)) is not int or not 1 <= getattr(args, field) <= 2**63 - 1:
            raise ValueError(f"{field} must be a positive signed i64 integer")
    if (args.small_bytes >= args.large_bytes or not 0 <= args.seed <= 2**63-1 or args.warmup < 0
            or not 0 < args.timeout < float("inf")):
        raise ValueError("require small_bytes < large_bytes, seed/warmup >= 0 and a finite positive timeout")
    dimensions = {name: list(getattr(args, name) or values) for name, values in PRESETS[args.preset].items()}
    dimensions["methods"] = list(args.methods)
    if any(len(values) != len(set(values)) for values in dimensions.values()):
        raise ValueError("matrix dimensions and methods must not contain duplicates")
    if not math.isfinite(args.scale) or abs(args.scale) > 3.4028234663852886e38:
        raise ValueError("scale must be finite f32")
    if len(args.disable_optimization) != len(set(args.disable_optimization)):
        raise ValueError("optimization switches must be unique")
    if args.mask and dimensions["ops"] != ["softmax"]:
        raise ValueError("mask requires all selected ops to be softmax")
    args.output, args.histogram = args.output.resolve(), args.histogram.resolve()
    histogram_sha256 = None
    if "public_histogram" in dimensions["distributions"]:
        sample_histogram(args.histogram, 0, seed=args.seed, total_elements=0)
        histogram_sha256 = hashlib.sha256(args.histogram.read_bytes()).hexdigest()
    report = {"schema": 1, "kind": "ragged-benchmark-matrix", "status": "planned", "scope": "planned",
        "preset": args.preset, "dimensions": dimensions, "requested_methods": list(args.methods),
        "matrix_harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "benchmark_harness_sha256": hashlib.sha256((ROOT / "bench/run.py").read_bytes()).hexdigest(),
        "size_policy": {"small_bytes": args.small_bytes, "large_bytes": args.large_bytes,
            "scope": "values plus offsets plus optional resident f32 mask input bytes only; output, oracle and workspaces are additional",
            "l2_relation": "unmeasured", "fixed_segments": "B and mean_length shared across selected distributions per dtype/D/offsets/size group; N varies",
            "fixed_segments_max_b": 100000}, "preflight": None, "rows": []}
    for dtype, d, offsets_dtype, size, normalization in itertools.product(
            dimensions["dtypes"], dimensions["ds"], dimensions["offsets_dtypes"],
            dimensions["sizes"], dimensions["normalizations"]):
        sizing = _sizing(dtype, d, offsets_dtype, getattr(args, size + "_bytes"), normalization,
                         dimensions["distributions"], args)
        for op, distribution in itertools.product(dimensions["ops"], dimensions["distributions"]):
            case = {"op": op, "dtype": dtype, "D": d, "offsets_dtype": offsets_dtype,
                "distribution": distribution, "normalization": normalization, "size": size, "seed": args.seed,
                "histogram_sha256": histogram_sha256 if distribution == "public_histogram" else None,
                "scale": struct.unpack("f", struct.pack("f", args.scale))[0], "requested_scale": args.scale,
                "mask": args.mask, "disabled_optimizations": list(args.disable_optimization),
                "feature_tile": args.feature_tile, "length_hints": args.length_hints,
                **sizing[distribution]}
            digest = hashlib.sha256(json.dumps(case, sort_keys=True).encode()).hexdigest()[:16]
            case_id = f"{op}-{dtype}-d{d}-{offsets_dtype}-{distribution}-{normalization}-{size}-{digest}"
            directory = Path("cases") / case_id
            reason = case["size_budget_reason"]
            methods = [{"method": method, "status": "unmeasured", "reason": PENDING_METHODS[method]}
                       if method in PENDING_METHODS else
                       {"method": method, "status": "blocked" if reason else "planned", "reason": reason}
                       for method in args.methods]
            report["rows"].append({"case_id": case_id, "case": case, "status": "blocked" if reason else "planned",
                "reason": reason, "directory": directory.as_posix(), "results_path": (directory / "results.json").as_posix(),
                "command": _command(args, case, args.output / directory), "exit_code": None, "methods": methods,
                "stdout_path": f"logs/{case_id}.stdout.txt", "stderr_path": f"logs/{case_id}.stderr.txt",
                "observed_case": None, "results_sha256": None})
    report["case_count"] = len(report["rows"])
    report["method_cell_count"] = len(report["rows"]) * len(args.methods)
    return report


def gpu_preflight() -> dict:
    try:
        torch = importlib.import_module("torch")
        if not torch.cuda.is_available():
            return {"status": "blocked", "reason": "no CUDA device; no child benchmarks or GPU measurements were run"}
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        device = {"name": props.name, "capability": [props.major, props.minor],
                  "total_memory_bytes": props.total_memory, "multiprocessors": props.multi_processor_count}
        if device["capability"] != [8, 9]:
            return {"status": "blocked", "reason": "benchmark matrix requires sm_89", "device": device}
        from bench.device import cuda_driver_info
        return {"status": "ready", "device": device, "torch": torch.__version__, "torch_cuda": torch.version.cuda,
                "cuda_driver": cuda_driver_info(torch.cuda.current_device())}
    except Exception as exc:
        return {"status": "blocked", "reason": f"GPU preflight failed: {type(exc).__name__}: {exc}"}


def _save(args, report):
    report["method_status_counts"] = dict(Counter(cell["status"] for row in report["rows"] for cell in row["methods"]))
    path = args.output / "matrix.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _unmeasured(row, status, reason):
    row.update(status=status, reason=reason)
    for cell in row["methods"]:
        if cell["method"] not in PENDING_METHODS:
            cell.update(status="blocked" if status == "blocked" else "unmeasured", reason=reason)


def _unique_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate results field: {key}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"nonfinite results value: {value}")


def _finite_json_float(value):
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"nonfinite preflight value: {value}")
    return parsed


def _requested_python_preflight(args) -> dict:
    # Do not resolve symlinks: different venvs may share a Python binary but
    # import different Torch builds. Match the interpreter used by bench.run.
    if os.path.isabs(args.python) and os.path.abspath(args.python) == os.path.abspath(sys.executable):
        return gpu_preflight()
    command = [args.python, "-c",
               "import json; from bench.matrix import gpu_preflight; "
               "print(json.dumps(gpu_preflight(), allow_nan=False))"]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "python"), str(ROOT), environment.get("PYTHONPATH", "")])
    probe = {"command": command, "exit_code": None, "stdout": "", "stderr": ""}
    try:
        completed = subprocess.run(command, cwd=ROOT, env=environment,
                                   capture_output=True, text=True,
                                   timeout=min(args.timeout, 60), check=False)
        probe.update(exit_code=completed.returncode, stdout=completed.stdout, stderr=completed.stderr)
        if completed.returncode != 0:
            raise ValueError(f"requested Python preflight exited {completed.returncode}")
        result = json.loads(completed.stdout, object_pairs_hook=_unique_fields,
                            parse_constant=_invalid_constant, parse_float=_finite_json_float)
        if (not isinstance(result, dict) or result.get("status") not in ("ready", "blocked")
                or (result["status"] == "blocked" and not isinstance(result.get("reason"), str))):
            raise ValueError("requested Python preflight returned an invalid status or reason")
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        for name in ("stdout", "stderr"):
            value = getattr(exc, name, None)
            if value is not None:
                probe[name] = value.decode(errors="replace") if isinstance(value, bytes) else value
        probe["execution_error"] = f"{type(exc).__name__}: {exc}"
        result = {"status": "blocked", "reason": probe["execution_error"]}
    return result | {"python_preflight_probe": probe}


def _read_result(args, row):
    path = args.output / row["results_path"]
    try:
        payload = path.read_bytes()
        row["results_sha256"] = hashlib.sha256(payload).hexdigest()
        result = json.loads(payload, object_pairs_hook=_unique_fields, parse_constant=_invalid_constant)
        row["observed_case"] = result.get("case")
        cells = result.get("methods")
        if not isinstance(cells, list):
            raise ValueError("child results lack a methods array")
        by_method = {}
        for cell in cells:
            if (not isinstance(cell, dict) or cell.get("method") not in RUN_METHODS
                    or cell["method"] in by_method or cell["method"] not in args.methods
                    or cell.get("status") not in ("measured", "failed", "not_applicable", "unmeasured")
                    or (cell.get("status") == "unmeasured" and
                        (not isinstance(cell.get("reason"), str) or not cell["reason"].strip()))):
                raise ValueError("child method results are duplicate, unexpected or malformed")
            by_method[cell["method"]] = cell
        for index, cell in enumerate(row["methods"]):
            if cell["method"] in by_method:
                row["methods"][index] = by_method[cell["method"]]
            elif cell["method"] not in PENDING_METHODS:
                row["methods"][index] = {"method": cell["method"], "status": "unmeasured",
                    "reason": "requested child method has no result"}
        status = result.get("status")
        missing = [method for method in args.methods if method in RUN_METHODS and method not in by_method]
        child_unmeasured = any(cell["status"] == "unmeasured" for cell in by_method.values())
        if status == "blocked":
            _unmeasured(row, "blocked", result.get("reason", "child benchmark blocked"))
        elif (status, row["exit_code"]) not in (("complete", 0), ("partial", 2), ("no_applicable_methods", 2)):
            row.update(status="failed", reason=f"child benchmark status={status!r}, exit_code={row['exit_code']}")
        elif missing:
            row.update(status="failed", reason="child omitted requested methods: " + ", ".join(missing))
        elif (status == "partial") != child_unmeasured:
            row.update(status="failed", reason="child completion status contradicts optional unmeasured cells")
        elif any(cell["status"] == "failed" for cell in row["methods"]):
            row.update(status="failed", reason="at least one method failed")
        elif any(cell["status"] == "unmeasured" for cell in row["methods"]):
            row.update(status="partial", reason="some requested methods remain unmeasured")
        else:
            row.update(status=status, reason=None)
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        _unmeasured(row, "failed", f"missing or invalid results.json: {exc}")


def run_matrix(args, *, preflight_fn=None, runner=None) -> tuple[dict, int]:
    report = plan(args)
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("output must be absent or empty; matrix runs never overwrite or retry cold results")
    args.output.mkdir(parents=True, exist_ok=True)
    _save(args, report)
    if args.plan_only:
        return report, 0
    report["scope"] = "execution_attempted"
    report["preflight"] = preflight_fn() if preflight_fn is not None else _requested_python_preflight(args)
    if report["preflight"]["status"] != "ready":
        for row in report["rows"]:
            _unmeasured(row, "blocked", row["reason"] or report["preflight"]["reason"])
        report["status"] = "blocked"
        _save(args, report)
        return report, 2
    report["status"] = "running"
    execute = runner or subprocess.run
    for row in report["rows"]:
        if row["status"] == "blocked":
            continue
        if not row["command"]:
            _unmeasured(row, "partial", "all requested benchmark adapters are pending")
            continue
        environment = dict(os.environ)
        caches = args.output / "isolated-caches" / row["case_id"]
        overrides = {"TORCHINDUCTOR_CACHE_DIR": str(caches / "inductor"), "TRITON_CACHE_DIR": str(caches / "triton"),
                     "CUDA_CACHE_PATH": str(caches / "cuda"), "TORCH_EXTENSIONS_DIR": str(caches / "extensions")}
        environment.update(overrides)
        environment["PYTHONPATH"] = os.pathsep.join([str(ROOT / "python"), str(ROOT), environment.get("PYTHONPATH", "")])
        row["cache_environment"] = overrides
        # Logs live outside the still-absent child result directory so bench.run
        # can enforce its own independent cold-directory check.
        (args.output / "logs").mkdir(exist_ok=True)
        row["status"] = "running"
        _save(args, report)
        try:
            completed = execute(row["command"], cwd=ROOT, env=environment, capture_output=True,
                                text=True, timeout=args.timeout, check=False)
            row["exit_code"] = completed.returncode
            stdout, stderr = completed.stdout, completed.stderr
        except (subprocess.TimeoutExpired, OSError) as exc:
            stdout, stderr = getattr(exc, "stdout", "") or "", getattr(exc, "stderr", "") or ""
            row["execution_error"] = f"{type(exc).__name__}: {exc}"
        for field, text in (("stdout_path", stdout), ("stderr_path", stderr)):
            (args.output / row[field]).write_text(text.decode(errors="replace") if isinstance(text, bytes) else text)
        _read_result(args, row)
        _save(args, report)
    statuses = Counter(cell["status"] for row in report["rows"] for cell in row["methods"])
    report["method_status_counts"] = dict(statuses)
    if any(row["status"] == "failed" for row in report["rows"]):
        report["status"], code = "failed", 1
    elif statuses.get("blocked") or statuses.get("unmeasured"):
        report["status"], code = "partial", 2
    elif statuses.get("measured"):
        report["status"], code = "complete", 0
    else:
        report["status"], code = "no_applicable_methods", 2
    _save(args, report)
    return report, code


def main(argv=None) -> int:
    arguments = parser()
    args = arguments.parse_args(argv)
    try:
        report, code = run_matrix(args)
    except (OSError, ValueError) as exc:
        arguments.error(str(exc))
    print(f"{report['status']}: {args.output / 'matrix.json'} ({report['case_count']} cases)")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
