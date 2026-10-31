"""Independent report ingestion faults, using synthetic host-only JSON fixtures.

The numbers below are test data, never GPU measurements or project evidence.
No compiler, Torch, CUDA runtime or benchmark subprocess runs in this module.
"""
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np

from bench import matrix
from bench.distributions import describe, make_lengths, offsets_from_lengths


def _report_api():
    from bench.report import ReportError, build_report
    return ReportError, build_report


def _summary(value):
    return {"count": 3, "samples": [value, value, value], "median": value,
            "min": value, "max": value, "p90": value}


def _cell(method, milliseconds, finite_values):
    return {"method": method, "status": "measured",
            "oracle": {"finite_values": finite_values, "nan_values": 0,
                       "max_abs_error": 0.0, "max_fraction_of_tolerance": 0.0},
            "complete_call_wall_until_complete_ms": _summary(milliseconds),
            "complete_call_cuda_event_ms": _summary(milliseconds / 2),
            "peak_additional_allocated_bytes": 4096,
            "allocator_baseline_allocated_bytes": 8192, "allocator_peak_allocated_bytes": 12288,
            "allocator_baseline_reserved_bytes": 16384, "allocator_peak_reserved_bytes": 16384,
            "allocation_scope": "output, temporaries, workspace, NJT construction and dtype conversions",
            "timing_scope": "all kernels and host enqueue gaps; inputs already on device; no CUDA graph",
            "warmup_calls": 1}


def _fixture(tmp_path, *, ops=("sum",), pending=False, dtype="f32", mask=False):
    methods = ["triton", "ragged-auto"] + (["sparse-gpu"] if pending else [])
    args = matrix.parser().parse_args([
        "--output", str(tmp_path), "--ops", *ops, "--dtypes", dtype, "--ds", "16",
        "--offsets-dtypes", "i64", "--distributions", "uniform",
        "--normalizations", "fixed_elements", "--sizes", "small", "--methods", *methods,
        "--warmup", "1", "--repeats", "3"])
    args.mask = mask
    manifest = matrix.plan(args)
    manifest.update(status="partial" if pending else "complete", scope="execution_attempted")
    device = {"name": "synthetic report contract fixture (not device evidence)",
              "capability": [8, 9], "total_memory_bytes": 24 << 30, "multiprocessors": 128}
    manifest["preflight"] = {"status": "ready", "device": device,
                              "torch": "2.14.1+cu132", "torch_cuda": "13.2"}
    children = []
    for row in manifest["rows"]:
        planned = row["case"]
        case = {key: planned[key] for key in (
            "op", "dtype", "D", "offsets_dtype", "distribution", "normalization",
            "segments", "elements", "seed", "histogram_sha256")}
        lengths = make_lengths("uniform", planned["segments"], seed=planned["seed"],
                               total_elements=planned["elements"], mean_length=planned["mean_length"])
        offsets = offsets_from_lengths(lengths, "i64")
        values = np.zeros((planned["elements"], planned["D"]), dtype=np.float32)
        mask_values = np.full(values.shape, np.float32(0.125)) if mask else None
        case.update(describe(lengths))
        case.update(mean_length_parameter=planned["mean_length"],
                    values_sha256=hashlib.sha256(values.tobytes()).hexdigest(),
                    offsets_sha256=hashlib.sha256(offsets.tobytes()).hexdigest(),
                    scale=1.0, mask_sha256=hashlib.sha256(mask_values.tobytes()).hexdigest() if mask else None,
                    disabled_optimizations=[], feature_tile=None,
                    length_bounds={})
        case.update(values_bytes=values.size * (2 if dtype == "bf16" else 4),
                    offsets_bytes=offsets.nbytes, mask_bytes=mask_values.nbytes if mask else 0)
        case["input_bytes"] = case["values_bytes"] + case["offsets_bytes"] + case["mask_bytes"]
        destination = tmp_path / row["directory"]
        destination.mkdir(parents=True, exist_ok=True)
        np.savez(destination / "case.npz", values=values, offsets=offsets,
                 **({"mask": mask_values} if mask else {}))
        finite_values = (planned["elements"] if planned["op"] == "softmax" else planned["segments"]) * planned["D"]
        child = {"schema": 1, "status": "complete", "case": case, "device": deepcopy(device),
                 "torch": "2.14.1+cu132", "torch_cuda": "13.2", "python": "3.12.0",
                 "command": [str(Path("bench/run.py").resolve()), *row["command"][3:]],
                 "harness_sha256": manifest["benchmark_harness_sha256"],
                 "methods": [_cell("triton", 2.0, finite_values), _cell("ragged-auto", 4.0, finite_values)],
                 "measured_method_count": 2}
        row.update(status="partial" if pending else "complete", exit_code=0,
                   observed_case=deepcopy(case), reason=None)
        children.append(child)
    _save(tmp_path, manifest, children)
    return manifest, children


def _save(tmp_path, manifest, children, *, update_observed=True):
    for row, child in zip(manifest["rows"], children):
        destination = tmp_path / row["results_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = (json.dumps(child, allow_nan=True, sort_keys=True) + "\n").encode()
        destination.write_bytes(payload)
        row["results_sha256"] = hashlib.sha256(payload).hexdigest()
        if update_observed:
            row["observed_case"] = deepcopy(child.get("case"))
        pending = [cell for cell in row["methods"] if cell["method"] in matrix.PENDING_METHODS]
        row["methods"] = deepcopy(child["methods"]) + pending
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps(manifest, allow_nan=True, sort_keys=True) + "\n")
    return path


def _build(tmp_path):
    _, build_report = _report_api()
    return build_report(tmp_path / "matrix.json", reference_method="triton")


def test_slower_method_is_preserved_and_compared_using_both_complete_call_metrics(tmp_path):
    _fixture(tmp_path)
    report = _build(tmp_path)
    assert len(report["rows"]) == 2
    slower = next(row for row in report["rows"] if row["method"] == "ragged-auto")
    assert slower["status"] == "measured"
    assert slower["speedup_wall"] == slower["speedup_event"] == pytest.approx(0.5)


def test_pending_baseline_cell_is_preserved_without_a_speedup(tmp_path):
    from bench.report import report_exit_code
    _fixture(tmp_path, pending=True)
    report = _build(tmp_path)
    pending = next(row for row in report["rows"] if row["method"] == "sparse-gpu")
    assert pending["status"] == "unmeasured" and pending["reason"]
    assert pending.get("speedup_wall") is None and pending.get("speedup_event") is None
    assert report["status"] != "complete"
    assert report_exit_code(report) == 2


def test_result_bytes_must_match_the_recorded_hash(tmp_path):
    manifest, _ = _fixture(tmp_path)
    path = tmp_path / manifest["rows"][0]["results_path"]
    path.write_bytes(path.read_bytes() + b" ")
    error, _ = _report_api()
    with pytest.raises(error):
        _build(tmp_path)


@pytest.mark.parametrize("mutation", ["missing", "values", "offsets", "dtype", "unexpected_array"])
def test_measured_input_archive_must_exist_and_match_its_identity(tmp_path, mutation):
    manifest, _ = _fixture(tmp_path)
    path = tmp_path / manifest["rows"][0]["directory"] / "case.npz"
    with np.load(path, allow_pickle=False) as source:
        values, offsets = source["values"], source["offsets"]
    if mutation == "missing":
        path.unlink()
    else:
        if mutation == "values":
            values[0, 0] = 1.0
        elif mutation == "offsets":
            offsets[1] += 1
        elif mutation == "dtype":
            values = values.astype(np.float64)
        np.savez(path, values=values, offsets=offsets,
                 **({"unexpected": np.zeros(1)} if mutation == "unexpected_array" else {}))
    error, _ = _report_api()
    with pytest.raises(error):
        _build(tmp_path)


def test_child_command_must_match_the_invocation_that_was_planned(tmp_path):
    manifest, children = _fixture(tmp_path)
    command = children[0]["command"]
    command[command.index("--repeats") + 1] = "300"
    _save(tmp_path, manifest, children)
    error, _ = _report_api()
    with pytest.raises(error):
        _build(tmp_path)


def test_bf16_saved_inputs_cannot_silently_retain_f32_precision(tmp_path):
    manifest, children = _fixture(tmp_path, dtype="bf16")
    path = tmp_path / manifest["rows"][0]["directory"] / "case.npz"
    with np.load(path, allow_pickle=False) as source:
        values, offsets = source["values"], source["offsets"]
    values[0, 0] = np.float32(1.1)
    np.savez(path, values=values, offsets=offsets)
    children[0]["case"]["values_sha256"] = hashlib.sha256(values.tobytes()).hexdigest()
    _save(tmp_path, manifest, children)
    error, _ = _report_api()
    with pytest.raises(error):
        _build(tmp_path)


@pytest.mark.parametrize("mutation", ["missing", "values", "shape", "dtype"])
def test_mask_archive_cannot_differ_from_the_measured_f32_input(tmp_path, mutation):
    manifest, _ = _fixture(tmp_path, ops=("softmax",), mask=True)
    path = tmp_path / manifest["rows"][0]["directory"] / "case.npz"
    with np.load(path, allow_pickle=False) as source:
        values, offsets, mask = source["values"], source["offsets"], source["mask"]
    if mutation == "values":
        mask[0, 0] += 1.0
    elif mutation == "shape":
        mask = mask[:, :1]
    elif mutation == "dtype":
        mask = mask.astype(np.float64)
    np.savez(path, values=values, offsets=offsets, **({"mask": mask} if mutation != "missing" else {}))
    error, _ = _report_api()
    with pytest.raises(error):
        _build(tmp_path)


@pytest.mark.parametrize("field,value", [
    ("op", "max"), ("dtype", "bf16"), ("D", 128), ("offsets_dtype", "i32"),
    ("normalization", "fixed_segments"), ("distribution", "long_tail"),
    ("seed", 712), ("segments", 1), ("elements", 1), ("histogram_sha256", "4" * 64),
    ("mean_length_parameter", 32), ("scale", 0.5), ("mask_sha256", "5" * 64),
    ("disabled_optimizations", ["shuffle"]), ("feature_tile", 8),
    ("length_bounds", {"min_len": 1, "max_len": 64}),
])
def test_child_input_contract_must_match_the_planned_case(tmp_path, field, value):
    manifest, children = _fixture(tmp_path)
    children[0]["case"][field] = value
    _save(tmp_path, manifest, children)
    error, _ = _report_api()
    with pytest.raises(error):
        _build(tmp_path)


def test_observed_input_hash_is_bound_to_the_child_result(tmp_path):
    manifest, children = _fixture(tmp_path)
    children[0]["case"]["values_sha256"] = "9" * 64
    _save(tmp_path, manifest, children, update_observed=False)
    error, _ = _report_api()
    with pytest.raises(error):
        _build(tmp_path)


@pytest.mark.parametrize("field", ["name", "total_memory_bytes", "multiprocessors"])
def test_matrix_does_not_silently_combine_different_devices(tmp_path, field):
    manifest, children = _fixture(tmp_path, ops=("sum", "max"))
    children[1]["device"][field] = "different device" if field == "name" else 1
    _save(tmp_path, manifest, children)
    error, _ = _report_api()
    with pytest.raises(error):
        _build(tmp_path)


@pytest.mark.parametrize("mutation", ["missing", "over_tolerance", "negative_error", "boolean_count"])
def test_measured_label_requires_valid_oracle_evidence(tmp_path, mutation):
    manifest, children = _fixture(tmp_path)
    cell = children[0]["methods"][1]
    if mutation == "missing":
        del cell["oracle"]
    elif mutation == "over_tolerance":
        cell["oracle"]["max_fraction_of_tolerance"] = 1.01
    elif mutation == "negative_error":
        cell["oracle"]["max_abs_error"] = -1.0
    else:
        cell["oracle"]["finite_values"] = True
    _save(tmp_path, manifest, children)
    error, _ = _report_api()
    with pytest.raises(error):
        _build(tmp_path)


@pytest.mark.parametrize("mutation", ["empty", "nan", "negative", "boolean", "count", "median", "p90"])
def test_report_recomputes_sample_summaries_and_rejects_non_measurements(tmp_path, mutation):
    manifest, children = _fixture(tmp_path)
    summary = children[0]["methods"][1]["complete_call_wall_until_complete_ms"]
    if mutation in ("count", "median", "p90"):
        summary[mutation] = 99
    elif mutation == "empty":
        summary["samples"] = []
    else:
        summary["samples"][0] = {"nan": float("nan"), "negative": -1.0, "boolean": True}[mutation]
    _save(tmp_path, manifest, children)
    error, _ = _report_api()
    with pytest.raises(error):
        _build(tmp_path)


def test_zero_duration_is_retained_but_cannot_be_a_speedup_denominator(tmp_path):
    manifest, children = _fixture(tmp_path)
    for key in ("complete_call_wall_until_complete_ms", "complete_call_cuda_event_ms"):
        children[0]["methods"][1][key] = _summary(0.0)
    _save(tmp_path, manifest, children)
    slower = next(row for row in _build(tmp_path)["rows"] if row["method"] == "ragged-auto")
    assert slower.get("speedup_wall") is None and slower.get("speedup_event") is None


def test_duplicate_child_method_is_rejected_even_with_a_matching_file_hash(tmp_path):
    manifest, children = _fixture(tmp_path)
    children[0]["methods"].append(deepcopy(children[0]["methods"][0]))
    _save(tmp_path, manifest, children)
    error, _ = _report_api()
    with pytest.raises(error):
        _build(tmp_path)


def test_duplicate_planned_case_is_rejected(tmp_path):
    manifest, children = _fixture(tmp_path)
    manifest["rows"].append(deepcopy(manifest["rows"][0]))
    manifest["case_count"] += 1
    manifest["method_cell_count"] *= 2
    _save(tmp_path, manifest, children)
    error, _ = _report_api()
    with pytest.raises(error):
        _build(tmp_path)


@pytest.mark.parametrize("field", ["case_count", "method_cell_count"])
def test_declared_inventory_count_cannot_hide_missing_cells(tmp_path, field):
    manifest, children = _fixture(tmp_path)
    manifest[field] += 1
    _save(tmp_path, manifest, children)
    error, _ = _report_api()
    with pytest.raises(error):
        _build(tmp_path)


def test_consistent_counts_do_not_excuse_an_omitted_declared_dimension(tmp_path):
    manifest, children = _fixture(tmp_path)
    manifest["dimensions"]["ops"].append("max")
    _save(tmp_path, manifest, children)
    error, _ = _report_api()
    with pytest.raises(error):
        _build(tmp_path)


def test_requested_method_inventory_must_match_declared_dimensions(tmp_path):
    manifest, children = _fixture(tmp_path)
    manifest["dimensions"]["methods"] = ["triton"]
    _save(tmp_path, manifest, children)
    error, _ = _report_api()
    with pytest.raises(error):
        _build(tmp_path)


def test_missing_file_cannot_retain_a_claimed_measured_result(tmp_path):
    manifest, _ = _fixture(tmp_path)
    (tmp_path / manifest["rows"][0]["results_path"]).unlink()
    error, _ = _report_api()
    with pytest.raises(error):
        _build(tmp_path)


def test_missing_child_method_cannot_retain_a_copied_measured_label(tmp_path):
    manifest, children = _fixture(tmp_path)
    inventory = deepcopy(manifest["rows"][0]["methods"])
    children[0]["methods"].pop()
    children[0]["measured_method_count"] = 1
    _save(tmp_path, manifest, children)
    manifest["rows"][0]["methods"] = inventory
    (tmp_path / "matrix.json").write_text(json.dumps(manifest))
    error, _ = _report_api()
    with pytest.raises(error):
        _build(tmp_path)


def test_blocked_preflight_cannot_become_a_complete_measured_report(tmp_path):
    manifest, children = _fixture(tmp_path)
    manifest.update(status="blocked", preflight={"status": "blocked", "reason": "synthetic no-device fixture"})
    _save(tmp_path, manifest, children)
    error, _ = _report_api()
    with pytest.raises(error):
        _build(tmp_path)


@pytest.mark.parametrize("mutation", ["peak", "reserved", "missing", "boolean"])
def test_allocator_counters_must_reconcile_and_remain_distinct(tmp_path, mutation):
    manifest, children = _fixture(tmp_path)
    cell = children[0]["methods"][1]
    if mutation == "peak":
        cell["allocator_peak_allocated_bytes"] = 12287
    elif mutation == "reserved":
        cell["allocator_peak_reserved_bytes"] = 1
    elif mutation == "missing":
        del cell["allocator_baseline_reserved_bytes"]
    else:
        cell["allocator_baseline_reserved_bytes"] = True
    _save(tmp_path, manifest, children)
    error, _ = _report_api()
    with pytest.raises(error):
        _build(tmp_path)


def test_blocked_matrix_retains_all_cells_without_performance_numbers(tmp_path):
    from bench.report import report_exit_code, write_report
    manifest, _ = _fixture(tmp_path)
    manifest.update(status="blocked", preflight={"status": "blocked", "reason": "synthetic no-device fixture"})
    for row in manifest["rows"]:
        (tmp_path / row["results_path"]).unlink()
        row.update(status="blocked", reason="synthetic no-device fixture", observed_case=None,
                   results_sha256=None, exit_code=None)
        row["methods"] = [{"method": name, "status": "blocked", "reason": row["reason"]}
                          for name in manifest["requested_methods"]]
    (tmp_path / "matrix.json").write_text(json.dumps(manifest))
    report = _build(tmp_path)
    assert len(report["rows"]) == manifest["method_cell_count"]
    assert all(row["status"] == "blocked" for row in report["rows"])
    assert all(row.get("metrics") is None and row.get("speedup_wall") is None
               and row.get("speedup_event") is None for row in report["rows"])
    assert report["status"] != "complete"
    assert report_exit_code(report) == 2
    destination = tmp_path / "rendered"
    write_report(report, destination, plot=True)
    assert not list(destination.glob("*.png"))
    assert "**blocked**" in (destination / "report.md").read_text()
    with (destination / "report.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(report["rows"])
    assert all(row["status"] == "blocked" for row in rows)
    assert all(row[field] == "" for row in rows for field in
               ("wall_ms", "event_ms", "speedup_wall", "speedup_event", "peak_additional_allocated_bytes"))


def test_failed_oracle_cell_remains_a_failure_with_no_estimated_performance(tmp_path):
    from bench.report import report_exit_code
    manifest, children = _fixture(tmp_path)
    children[0]["methods"][1] = {"method": "ragged-auto", "status": "failed",
                                "error": "AssertionError: synthetic oracle mismatch"}
    children[0].update(status="failed", measured_method_count=1)
    manifest["status"] = "failed"
    manifest["rows"][0].update(status="failed", exit_code=1)
    _save(tmp_path, manifest, children)
    report = _build(tmp_path)
    failed = next(row for row in report["rows"] if row["method"] == "ragged-auto")
    assert failed["status"] == "failed" and "oracle mismatch" in failed["reason"]
    assert failed["metrics"] is None and failed["speedup_wall"] is None and failed["speedup_event"] is None
    assert report_exit_code(report) == 1


def test_report_export_preserves_existing_evidence(tmp_path):
    from bench.report import write_report
    _fixture(tmp_path)
    report = _build(tmp_path)
    destination = tmp_path / "rendered"
    write_report(report, destination)
    prior = {path.name: path.read_bytes() for path in destination.iterdir()}
    error, _ = _report_api()
    with pytest.raises(error):
        write_report(report, destination)
    assert prior == {path.name: path.read_bytes() for path in destination.iterdir()}


def test_actual_plot_and_markdown_preserve_slower_ratio_and_partial_scope(tmp_path):
    """Rendering is real; every numeric source is explicitly a test fixture."""
    from bench.report import write_report
    from PIL import Image
    manifest, children = _fixture(tmp_path, pending=True)
    strategy = "synthetic adapter strategy, not GPU evidence"
    children[0]["methods"][1].update(baseline_strategy=strategy, source_sha256="a" * 64)
    _save(tmp_path, manifest, children)
    report = _build(tmp_path)
    measured = next(row for row in report["rows"] if row["method"] == "ragged-auto")
    assert measured["metrics"]["baseline_strategy"] == strategy
    assert measured["metrics"]["provenance"]["source_sha256"] == "a" * 64
    destination = tmp_path / "synthetic-contract-render-only"
    write_report(report, destination, plot=True)
    images = list(destination.glob("*.png"))
    assert len(images) == 1 and images[0].name.startswith("ragged-auto-wall-ratios")
    with Image.open(images[0]) as image:
        assert image.width >= 100 and image.height >= 100
        image.verify()
    markdown = (destination / "report.md").read_text()
    assert "**partial**" in markdown and "synthetic report contract fixture" in markdown
    assert "sparse-gpu" in markdown and "unmeasured" in markdown
    assert strategy in markdown
    assert "Values below 1" in markdown and "do not establish L2 residency" in markdown
    with (destination / "report.csv").open(newline="") as stream:
        candidate = next(row for row in csv.DictReader(stream) if row["method"] == "ragged-auto")
    assert float(candidate["speedup_wall"]) == 0.5
    assert candidate["implementation"] == strategy


def test_ragged_complete_call_keeps_runtime_work_but_not_evidence_serialization_inside_timing(tmp_path, monkeypatch):
    """A host spy checks placement only; it supplies no synthetic CUDA timing."""
    import ragged
    from bench.run import make_call

    state = {"timed": False, "runs": 0, "allocations": 0, "serializations": 0}

    class Report:
        def to_dict(self):
            assert not state["timed"], "benchmark evidence serialization contaminated the complete-call interval"
            state["serializations"] += 1
            return {"synthetic_host_contract_only": True}

    class Program:
        cache_hit = False
        artifact = SimpleNamespace(key=SimpleNamespace(digest="fixture", compiler_sha256="0" * 64))

        def run(self, *args, **kwargs):
            state["runs"] += 1
            return Report()

    class Compiler:
        def __init__(self, *args):
            self.calls = 0

        def compile(self, *args):
            self.calls += 1
            return Program() if self.calls == 1 else SimpleNamespace(cache_hit=True)

        def toolchain(self):
            return {"synthetic_host_contract_only": True}

    def empty(*args, **kwargs):
        state["allocations"] += 1
        return object()

    monkeypatch.setattr(ragged, "Compiler", Compiler)
    args = SimpleNamespace(compiler="synthetic", op="sum", dtype="f32", d=16, offsets_dtype="i64", scale=1.0)
    values = SimpleNamespace(shape=(4, 16), dtype="synthetic-f32", device="synthetic-device")
    offsets = SimpleNamespace(numel=lambda: 3)
    call, _ = make_call("ragged-auto", args, SimpleNamespace(empty=empty), values, offsets, tmp_path)
    call()  # Every real benchmark performs an untimed oracle call first.
    state["timed"] = True
    for _ in range(4):
        call()
    assert state["runs"] == state["allocations"] == 5
