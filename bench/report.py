"""Aggregate retained benchmark evidence without inventing missing measurements."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import statistics
import textwrap
import zipfile
import numpy as np


class ReportError(ValueError):
    pass


def _json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ReportError(f"duplicate JSON field: {key}")
            result[key] = value
        return result
    try:
        return json.loads(path.read_text(), object_pairs_hook=unique,
                          parse_constant=lambda value: (_ for _ in ()).throw(ReportError(f"invalid JSON number: {value}")))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReportError(f"cannot read {path}: {exc}") from exc


def _number(value, description):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ReportError(f"{description} must be finite and nonnegative")
    return value


def _integer(value, description):
    if type(value) is not int or not 0 <= value <= 2**63 - 1:
        raise ReportError(f"{description} must be a nonnegative signed i64 integer")
    return value


def _timing(value, description):
    fields = {"count", "median", "min", "max", "p90", "samples"}
    if not isinstance(value, dict) or set(value) != fields:
        raise ReportError(f"{description} lacks a complete timing summary")
    samples = value["samples"]
    if not isinstance(samples, list) or not samples or type(value["count"]) is not int or value["count"] != len(samples):
        raise ReportError(f"{description} samples/count mismatch")
    for sample in samples:
        _number(sample, description)
    ordered = sorted(samples)
    position = (len(samples) - 1) * 0.9
    low, high = math.floor(position), math.ceil(position)
    p90 = ordered[low] + (ordered[high] - ordered[low]) * (position - low)
    expected = {"median": statistics.median(samples), "min": min(samples), "max": max(samples), "p90": p90}
    for field, number in expected.items():
        actual = _number(value[field], description)
        if not math.isclose(actual, number, rel_tol=1e-12, abs_tol=1e-12):
            raise ReportError(f"{description} {field} differs from retained samples")
    return value["median"]


def _measured(cell):
    oracle = cell.get("oracle")
    required = {"finite_values", "nan_values", "max_abs_error", "max_fraction_of_tolerance"}
    if not isinstance(oracle, dict) or not required <= set(oracle):
        raise ReportError("measured cell lacks independent oracle evidence")
    for field in ("finite_values", "nan_values"):
        _integer(oracle[field], "oracle " + field)
    _number(oracle["max_abs_error"], "oracle max_abs_error")
    if _number(oracle["max_fraction_of_tolerance"], "oracle tolerance fraction") > 1:
        raise ReportError("measured cell exceeds the fixed oracle tolerance")
    wall = _timing(cell.get("complete_call_wall_until_complete_ms"), "complete-call wall time")
    event = _timing(cell.get("complete_call_cuda_event_ms"), "complete-call CUDA-event time")
    if (cell["complete_call_wall_until_complete_ms"]["count"]
            != cell["complete_call_cuda_event_ms"]["count"]):
        raise ReportError("wall and CUDA-event sample counts differ")
    extra = _integer(cell.get("peak_additional_allocated_bytes"), "additional allocated memory")
    raw = ("allocator_baseline_allocated_bytes", "allocator_peak_allocated_bytes",
           "allocator_baseline_reserved_bytes", "allocator_peak_reserved_bytes")
    if any(field in cell for field in raw):
        counters = [_integer(cell.get(field), field) for field in raw]
        baseline, peak, baseline_reserved, peak_reserved = counters
        if (peak - baseline != extra or peak < baseline or peak_reserved < baseline_reserved
                or baseline_reserved < baseline or peak_reserved < peak):
            raise ReportError("allocator counters and additional allocated memory disagree")
    for field in ("timing_scope", "allocation_scope"):
        if not isinstance(cell.get(field), str) or not cell[field]:
            raise ReportError("measured cell lacks " + field)
    return {"wall_ms": wall, "event_ms": event, "peak_additional_allocated_bytes": extra,
            **{field: cell.get(field) for field in raw}, "oracle": oracle,
            "cold_compile_wall_ms": cell.get("cold_compile_wall_ms"),
            "hot_cache_wall_ms": cell.get("hot_cache_wall_ms"),
            "compile_and_first_call_wall_ms": cell.get("compile_and_first_call_wall_ms"),
            "execution_report": cell.get("last_execution_report"),
            "implementation": cell.get("baseline_strategy", cell.get("mapping")),
            "baseline_strategy": cell.get("baseline_strategy"),
            "provenance": {key: cell[key] for key in ("source_sha256", "toolchain", "library_sha256",
                "compiler_sha256", "cache_key", "spec") if key in cell},
            "timing_scope": cell["timing_scope"], "allocation_scope": cell["allocation_scope"],
            "memory_scope": cell.get("memory_scope", "additional PyTorch allocator memory only")}


def _hash(value, name):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ReportError(f"missing or invalid {name} hash")


def _input_artifact(path: Path, case: dict) -> str:
    """Bind reported input identities to the inputs the harness retained."""
    try:
        with np.load(path, allow_pickle=False) as saved:
            expected = {"values", "offsets"} | ({"mask"} if case.get("mask_sha256") else set())
            if set(saved.files) != expected:
                raise ReportError("saved input array inventory differs from the case")
            values, offsets = saved["values"], saved["offsets"]
            n, b, d = case["elements"], case["segments"], case["D"]
            if values.shape != (n, d) or values.dtype != np.float32:
                raise ReportError("saved values shape/dtype differs from case")
            if offsets.shape != (b + 1,) or offsets.dtype != np.dtype("int32" if case["offsets_dtype"] == "i32" else "int64"):
                raise ReportError("saved offsets shape/dtype differs from case")
            if (offsets[0] != 0 or offsets[-1] != n or np.any(offsets < 0)
                    or np.any(offsets > n) or np.any(offsets[:-1] > offsets[1:])):
                raise ReportError("saved offsets violate the validated benchmark contract")
            if hashlib.sha256(values.astype("<f4", copy=False).tobytes()).hexdigest() != case["values_sha256"]:
                raise ReportError("saved values hash differs from case")
            if hashlib.sha256(offsets.tobytes()).hexdigest() != case["offsets_sha256"]:
                raise ReportError("saved offsets hash differs from case")
            lengths = np.diff(offsets.astype(np.int64))
            if hashlib.sha256(lengths.astype("<i8").tobytes()).hexdigest() != case.get("lengths_sha256"):
                raise ReportError("saved lengths hash differs from case")
            if case["dtype"] == "bf16" and np.any(values.view(np.uint32) & 0xFFFF):
                raise ReportError("saved bf16 inputs lack their original quantization")
            if "mask" in expected:
                mask = saved["mask"]
                if mask.shape != values.shape or mask.dtype != np.float32:
                    raise ReportError("saved mask shape/dtype differs from case")
                if hashlib.sha256(mask.tobytes()).hexdigest() != case["mask_sha256"]:
                    raise ReportError("saved mask hash differs from case")
            byte_counts = {"values_bytes": n * d * (2 if case["dtype"] == "bf16" else 4),
                           "offsets_bytes": offsets.nbytes, "mask_bytes": n * d * 4 if "mask" in expected else 0}
            byte_counts["input_bytes"] = sum(byte_counts.values())
            if any(case.get(field) != count for field, count in byte_counts.items()):
                raise ReportError("reported resident input bytes differ from saved shapes/dtypes")
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as exc:
        if isinstance(exc, ReportError):
            raise
        raise ReportError(f"cannot validate retained inputs: {exc}") from exc


def _identity(case, planned, source, matrix):
    if not isinstance(case, dict):
        raise ReportError("measured results lack input identity")
    keys = ("op", "dtype", "D", "offsets_dtype", "distribution", "normalization", "segments", "elements", "seed")
    for key in keys:
        if key not in case or key not in planned or case[key] != planned[key]:
            raise ReportError(f"observed {key} differs from the requested case")
    for key in ("D", "segments", "elements", "seed"):
        _integer(case[key], "case " + key)
    if case.get("mean_length_parameter") != planned.get("mean_length"):
        raise ReportError("observed mean-length parameter differs from the requested case")
    for name in ("values_sha256", "offsets_sha256"):
        _hash(case.get(name), name)
    _hash(case.get("lengths_sha256"), "lengths")
    if case.get("histogram_sha256") != planned.get("histogram_sha256"):
        raise ReportError("histogram source differs from the requested case")
    for key in ("histogram_sha256", "mask_sha256"):
        if case.get(key) is not None:
            _hash(case[key], key)
    scale = planned.get("scale", 1.0)
    if type(case.get("scale")) not in (int, float) or not math.isfinite(case["scale"]):
        raise ReportError("scale must be finite numeric input identity")
    if case.get("scale") != scale:
        raise ReportError("scale differs from the requested case")
    if bool(case.get("mask_sha256")) != planned.get("mask", False):
        raise ReportError("mask presence differs from the requested case")
    if case.get("disabled_optimizations", []) != planned.get("disabled_optimizations", []):
        raise ReportError("optimization controls differ from the requested case")
    if case.get("feature_tile") != planned.get("feature_tile"):
        raise ReportError("feature tile differs from the requested case")
    expected_hints = planned.get("length_hints", False)
    hints = case.get("length_bounds")
    if not isinstance(hints, dict) or bool(hints) != expected_hints:
        raise ReportError("length hints differ from the requested case")
    if expected_hints:
        expected = {"min_len": case.get("min_length") or 0, "max_len": case.get("max_length") or 0}
        if hints != expected:
            raise ReportError("length hints differ from actual saved workload bounds")
    device = source.get("device")
    if (not isinstance(device, dict) or device.get("capability") != [8, 9]
            or any(type(x) is not int for x in device["capability"])
            or not isinstance(device.get("name"), str) or not device["name"]):
        raise ReportError("measured results lack an sm_89 device identity")
    for key in ("total_memory_bytes", "multiprocessors"):
        if _integer(device.get(key), "device " + key) == 0:
            raise ReportError("measured device properties must be positive")
    if not source.get("torch_cuda") or not source.get("torch"):
        raise ReportError("measured results lack CUDA/PyTorch versions")
    _hash(source.get("harness_sha256"), "benchmark harness")
    if source["harness_sha256"] != matrix.get("benchmark_harness_sha256"):
        raise ReportError("benchmark harness differs from the matrix snapshot")
    preflight = matrix.get("preflight")
    if not isinstance(preflight, dict) or preflight.get("status") != "ready":
        raise ReportError("measured results require a successful device preflight")
    if device != preflight.get("device"):
        raise ReportError("measured device differs from the matrix preflight")
    if any(source[key] != preflight.get(key) for key in ("torch", "torch_cuda")):
        raise ReportError("measured framework differs from the matrix preflight")
    for key in ("device_uuid_hex", "driver_package_version", "cuda_supported_version", "l2_cache_size_bytes"):
        expected = (preflight.get("cuda_driver") or {}).get(key)
        observed = (source.get("cuda_driver") or {}).get(key)
        if expected is not None and expected != observed:
            raise ReportError("measured CUDA Driver/device identity differs from preflight: " + key)
    return {"case": case, "device": device, "cuda_driver": source.get("cuda_driver"),
            "torch": source["torch"], "torch_cuda": source["torch_cuda"],
            "harness_sha256": source["harness_sha256"]}


def build_report(matrix_path: Path, reference_method: str = "triton") -> dict:
    matrix_path = Path(matrix_path).resolve()
    matrix = _json(matrix_path)
    if not isinstance(matrix, dict) or matrix.get("schema") != 1 or matrix.get("kind") != "ragged-benchmark-matrix":
        raise ReportError("unsupported benchmark matrix schema")
    if matrix.get("status") not in ("planned", "blocked", "running", "complete", "partial", "failed", "no_applicable_methods"):
        raise ReportError("unknown matrix status")
    requested = matrix.get("requested_methods", matrix.get("dimensions", {}).get("methods"))
    if (not isinstance(requested, list) or not requested or any(not isinstance(m, str) or not m for m in requested)
            or len(set(requested)) != len(requested)
            or any(re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name) is None for name in requested)):
        raise ReportError("requested method inventory must be nonempty and unique")
    if reference_method not in requested:
        raise ReportError("reference method is absent from the requested matrix")
    cases = matrix.get("rows")
    if not isinstance(cases, list) or not cases:
        raise ReportError("matrix must retain a nonempty requested case inventory")
    if matrix.get("case_count") != len(cases) or matrix.get("method_cell_count") != len(cases) * len(requested):
        raise ReportError("declared matrix inventory counts differ from retained rows")
    axes = ("ops", "dtypes", "ds", "offsets_dtypes", "distributions", "normalizations", "sizes")
    case_keys = ("op", "dtype", "D", "offsets_dtype", "distribution", "normalization", "size")
    dimensions = matrix.get("dimensions", {})
    if not isinstance(dimensions, dict) or any(not isinstance(dimensions.get(axis), list) or not dimensions[axis]
                                             for axis in axes):
        raise ReportError("matrix dimensions are incomplete")
    if dimensions.get("methods") != requested:
        raise ReportError("requested method inventories disagree")
    for axis in axes:
        expected_type = int if axis == "ds" else str
        if any(type(value) is not expected_type or not value for value in dimensions[axis]):
            raise ReportError("invalid matrix dimension values")
    if any(len(values := dimensions[axis]) != len(set(values)) for axis in axes):
        raise ReportError("matrix dimension values must be unique")
    expected_grid = set(itertools.product(*(dimensions[axis] for axis in axes)))
    observed_grid = [tuple(row.get("case", {}).get(key) for key in case_keys) for row in cases if isinstance(row, dict)]
    if len(observed_grid) != len(expected_grid) or set(observed_grid) != expected_grid:
        raise ReportError("requested dimension grid has missing, duplicate or unexpected cases")
    result = {"schema": 1, "kind": "ragged-benchmark-report", "source_matrix": str(matrix_path),
              "source_matrix_sha256": hashlib.sha256(matrix_path.read_bytes()).hexdigest(),
              "source_status": matrix.get("status"), "scope": matrix.get("scope"),
              "reference_method": reference_method, "comparison_scope": "same saved case and device; complete-call median",
              "environment": matrix.get("preflight"),
              "rows": []}
    case_ids = set()
    for row in cases:
        if not isinstance(row, dict) or not isinstance(row.get("case"), dict):
            raise ReportError("invalid requested case row")
        case_id = row.get("case_id")
        if (not isinstance(case_id, str) or re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", case_id) is None
                or case_id in case_ids):
            raise ReportError("case IDs must be unique path components")
        case_ids.add(case_id)
        copied = row.get("methods")
        if (not isinstance(copied, list) or len(copied) != len(requested)
                or [cell.get("method") for cell in copied if isinstance(cell, dict)] != requested):
            raise ReportError("case method inventory differs from requested methods")
        expected_path = f"cases/{case_id}/results.json"
        if row.get("results_path") != expected_path or row.get("directory") != f"cases/{case_id}":
            raise ReportError("case results path differs from its inventory identity")
        path = (matrix_path.parent / expected_path).resolve()
        if not path.is_relative_to(matrix_path.parent):
            raise ReportError("case result escapes the matrix directory")
        actual, identity, source_hash = {}, None, None
        if path.is_file():
            source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            if row.get("results_sha256") != source_hash:
                raise ReportError("child results hash differs from the matrix record")
            source = _json(path)
            if not isinstance(source, dict) or source.get("schema") != 1 or not isinstance(source.get("methods"), list):
                raise ReportError("invalid child benchmark schema")
            if row.get("observed_case") != source.get("case"):
                raise ReportError("observed case differs from retained child results")
            for cell in source["methods"]:
                if (not isinstance(cell, dict) or cell.get("method") not in requested
                        or cell["method"] in actual or cell.get("status") not in ("measured", "failed", "not_applicable", "unmeasured")
                        or (cell.get("status") == "unmeasured" and
                            (not isinstance(cell.get("reason"), str) or not cell["reason"].strip()))):
                    raise ReportError("duplicate, unexpected or invalid child method result")
                actual[cell["method"]] = cell
            if any(cell["status"] == "measured" for cell in actual.values()):
                identity = _identity(source.get("case"), row["case"], source, matrix)
                command = source.get("command")
                expected_command = row.get("command")
                if (not isinstance(command, list) or not command or not isinstance(expected_command, list)
                        or len(expected_command) < 4 or expected_command[1:3] != ["-m", "bench.run"]
                        or command[1:] != expected_command[3:]):
                    raise ReportError("child command differs from the requested invocation")
                inputs_path = path.with_name("case.npz").resolve()
                if not inputs_path.is_relative_to(matrix_path.parent):
                    raise ReportError("saved inputs escape the matrix directory")
                identity["case_npz_sha256"] = _input_artifact(inputs_path, source["case"])
            for cell in copied:
                name = cell["method"]
                if name not in actual and cell.get("status") == "measured":
                    raise ReportError("claimed measured method is missing from child results")
                if name in actual and cell.get("status") != actual[name]["status"]:
                    raise ReportError("copied method status differs from retained child result")
        elif any(cell.get("status") == "measured" for cell in copied):
            raise ReportError("claimed measurement has no retained child results")
        metrics = {name: _measured(cell) for name, cell in actual.items() if cell["status"] == "measured"}
        reference = metrics.get(reference_method)
        for cell in copied:
            name = cell["method"]
            observed = actual.get(name)
            status = observed["status"] if observed else cell.get("status", "unmeasured")
            if status not in ("measured", "failed", "not_applicable", "unmeasured", "blocked", "planned"):
                raise ReportError("unknown cell status")
            reason = (observed or cell).get("reason") or (observed or cell).get("error") or row.get("reason")
            if observed is None and status in ("failed", "not_applicable"):
                status, reason = "unmeasured", "no retained child result: " + (reason or "missing evidence")
            if observed is None and row.get("status") not in ("planned", "blocked") and status == "planned":
                status, reason = "unmeasured", "requested method has no retained result"
            entry = {"case_id": case_id, "case": row["case"], "method": name, "status": status,
                     "reason": reason, "case_status": row.get("status"), "source_results_sha256": source_hash,
                     "identity": identity, "metrics": metrics.get(name),
                     "source_method": observed,
                     "speedup_wall": None, "speedup_event": None, "comparison_reason": None}
            if name in metrics:
                if reference is None:
                    entry["comparison_reason"] = "reference method is unmeasured"
                elif metrics[name]["timing_scope"] != reference["timing_scope"]:
                    raise ReportError("candidate and reference timing scopes differ")
                else:
                    for metric, output in (("wall_ms", "speedup_wall"), ("event_ms", "speedup_event")):
                        numerator, denominator = reference[metric], metrics[name][metric]
                        if numerator > 0 and denominator > 0:
                            ratio = numerator / denominator
                            if math.isfinite(ratio):
                                entry[output] = ratio
                        if entry[output] is None:
                            entry["comparison_reason"] = "zero or unrepresentable timing ratio"
            result["rows"].append(entry)
    counts = dict(Counter(row["status"] for row in result["rows"]))
    result.update(case_count=len(case_ids), method_cell_count=len(result["rows"]), method_status_counts=counts)
    if matrix.get("status") == "planned":
        if counts.get("measured"):
            raise ReportError("planned matrix cannot claim measured cells")
        result["status"] = "planned"
    elif matrix.get("status") == "blocked":
        if counts.get("measured"):
            raise ReportError("blocked matrix cannot claim measured cells")
        result["status"] = "blocked"
    elif counts.get("failed") or any(row["case_status"] == "failed" for row in result["rows"]):
        result["status"] = "failed"
    elif any(counts.get(status) for status in ("unmeasured", "blocked", "planned")):
        result["status"] = "partial"
    elif counts.get("measured"):
        result["status"] = "complete"
    else:
        result["status"] = "no_applicable_methods"
    return result


def report_exit_code(report):
    return 0 if report["status"] == "complete" else 1 if report["status"] == "failed" else 2


def _display(value):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_markdown(report: dict) -> str:
    lines = ["# Benchmark evidence report", "", f"Status: **{report['status']}**. Scope: {_display(report['scope'])}.", "",
             f"Requested cases: {report['case_count']}; method cells: {report['method_cell_count']}.",
             f"Cell statuses: {_display(report['method_status_counts'])}.", "",
             f"Device and framework observation: {_display(report.get('environment'))}.", "",
             f"Ratios use `{report['reference_method']}` from the same saved case. Values below 1 retain slower methods.",
             "Wall time includes completion; CUDA-event time and allocator counters remain separate metrics.",
             "Missing, blocked, failed and inapplicable cells have no performance estimate. Input-size labels do not establish L2 residency.", "",
             "| Case | Method | Strategy | Status | Wall ms | Event ms | Extra allocated bytes | Wall ratio | Reason |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in report["rows"]:
        metrics = row["metrics"] or {}
        values = (row["case_id"], row["method"], metrics.get("implementation"), row["status"], metrics.get("wall_ms"), metrics.get("event_ms"),
                  metrics.get("peak_additional_allocated_bytes"), row["speedup_wall"], row["reason"] or row["comparison_reason"])
        lines.append("| " + " | ".join(map(_display, values)) + " |")
    return "\n".join(lines) + "\n"


def write_report(report, directory: Path, *, plot=False):
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise ReportError("report output directory must be absent or empty")
    plotted = [row for row in report["rows"] if row["speedup_wall"] is not None and row["method"] != report["reference_method"]]
    if plot and plotted:
        import os
        os.environ.setdefault("MPLCONFIGDIR", str(directory.resolve() / ".matplotlib-cache"))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (directory / "report.md").write_text(render_markdown(report))
    fields = ("case_id", "op", "dtype", "D", "offsets_dtype", "distribution", "normalization", "size", "method",
              "status", "implementation", "wall_ms", "event_ms", "peak_additional_allocated_bytes", "allocator_baseline_allocated_bytes",
              "allocator_peak_allocated_bytes", "allocator_baseline_reserved_bytes", "allocator_peak_reserved_bytes",
              "cold_compile_wall_ms", "hot_cache_wall_ms", "compile_and_first_call_wall_ms", "speedup_wall", "speedup_event", "reason")
    with (directory / "report.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in report["rows"]:
            writer.writerow({field: row.get(field, row["case"].get(field, (row["metrics"] or {}).get(field))) for field in fields})
    if plot and plotted:
        # Paginated actual paired cells; missing cells remain in the full CSV.
        for method in dict.fromkeys(row["method"] for row in plotted):
            cells = [row for row in plotted if row["method"] == method]
            for page, begin in enumerate(range(0, len(cells), 64), 1):
                subset = cells[begin:begin + 64]
                fig, ax = plt.subplots(figsize=(12, max(4, len(subset) * 0.24 + 2)))
                y = list(range(len(subset)))
                values = [row["speedup_wall"] for row in subset]
                ax.barh(y, values, color=["#2563eb" if x >= 1 else "#b45309" for x in values])
                ax.set_yticks(y, labels=[row["case_id"] for row in subset], fontsize=6)
                ax.axvline(1, color="black", linewidth=0.8)
                ax.set_xlabel(f"{report['reference_method']} wall median / {method} wall median")
                env = report.get("environment") or {}
                device = env.get("device", {}).get("name", "unrecorded device")
                title = (f"{method}: same-case ratios ({report['status']} matrix, page {page})\n"
                         f"{device}; Torch {env.get('torch')}; CUDA {env.get('torch_cuda')}")
                ax.set_title(textwrap.fill(title, 100), fontsize=9)
                fig.text(0.02, 0.015, "Only comparable measured cells shown. Ratios <1 are slower; full inventory and input conditions are retained in report.csv/json.", fontsize=6)
                fig.tight_layout(rect=(0, 0.04, 1, 1))
                fig.savefig(directory / f"{method}-wall-ratios-{page}.png", dpi=160)
                plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--reference-method", default="triton")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", action="store_true", help="plot actual comparable cells; requires matplotlib")
    args = parser.parse_args()
    try:
        report = build_report(args.matrix, args.reference_method)
        write_report(report, args.output, plot=args.plot)
    except (ReportError, ImportError) as exc:
        parser.exit(1, f"report rejected: {exc}\n")
    print(f"{report['status']}: {args.output / 'report.json'}")
    return report_exit_code(report)


if __name__ == "__main__":
    raise SystemExit(main())
