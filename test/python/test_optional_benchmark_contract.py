"""Independent optional-method orchestration contracts, with synthetic host spies.

All CUDA properties, timings and adapter results in this module are explicitly
synthetic test inputs. These tests do not qualify FBGEMM_GPU on CUDA or supply
GPU performance evidence. They exercise the real benchmark status/exit paths and
the retained child JSON/NPZ ingestion boundary.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from bench import matrix, run as benchmark
from bench.distributions import describe, make_lengths, offsets_from_lengths
from bench.report import ReportError, build_report, report_exit_code


SYNTHETIC_DEVICE = {
    "name": "synthetic optional benchmark contract fixture; not GPU evidence",
    "capability": [8, 9], "total_memory_bytes": 24 << 30, "multiprocessors": 128,
}


def _raise(error):
    raise error


def _summary(value):
    return {"count": 3, "samples": [value] * 3, "median": value,
            "min": value, "max": value, "p90": value}


def _synthetic_metrics():
    return {
        "complete_call_wall_until_complete_ms": _summary(4.0),
        "complete_call_cuda_event_ms": _summary(2.0),
        "peak_additional_allocated_bytes": 16,
        "allocator_baseline_allocated_bytes": 32, "allocator_peak_allocated_bytes": 48,
        "allocator_baseline_reserved_bytes": 64, "allocator_peak_reserved_bytes": 64,
        "timing_scope": "synthetic complete-call timing fixture; not GPU evidence",
        "allocation_scope": "synthetic allocation fixture; not GPU evidence",
    }


def _run_args(path, *, methods=("fbgemm",), op="softmax"):
    path.mkdir()
    return SimpleNamespace(
        output=path, methods=list(methods), op=op, dtype="f32", offsets_dtype="i64",
        d=16, distribution="uniform", segments=4, total_elements=8,
        mean_length=2, seed=711, histogram=None, scale=1.0, mask=False,
        length_hints=False, disable_optimization=[], feature_tile=None,
        warmup=0, repeats=3, compiler="unused-host-fixture", native_library="unused-host-fixture",
    )


@pytest.fixture
def synthetic_cuda(monkeypatch):
    """Redirect input placement to actual CPU Torch; never execute GPU kernels."""
    torch = pytest.importorskip("torch")
    original_to = torch.Tensor.to

    def cpu_only_to(tensor, *args, **kwargs):
        args = list(args)
        if args and args[0] == "cuda":
            args[0] = "cpu"
        if kwargs.get("device") == "cuda":
            kwargs["device"] = "cpu"
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", cpu_only_to)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: SimpleNamespace(
        name=SYNTHETIC_DEVICE["name"], major=8, minor=9,
        total_memory=SYNTHETIC_DEVICE["total_memory_bytes"],
        multi_processor_count=SYNTHETIC_DEVICE["multiprocessors"]))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(benchmark, "cuda_driver_info", lambda _: {"synthetic_host_fixture": True})
    measured = []

    def synthetic_measure(call, supplied_torch, *, method, **kwargs):
        assert supplied_torch is torch
        measured.append(method)
        return _synthetic_metrics()

    monkeypatch.setattr(benchmark, "measure", synthetic_measure)
    return torch, measured


def _cpu_softmax(torch, values, offsets, operation="softmax", **kwargs):
    assert operation == "softmax"
    chunks = [torch.softmax(values[int(start):int(end)].float(), dim=0)
              for start, end in zip(offsets[:-1], offsets[1:])]
    return torch.cat(chunks, dim=0).to(values.dtype)


def _optional_fault(monkeypatch, error):
    from baselines import fbgemm_reference
    monkeypatch.setattr(fbgemm_reference, "toolchain", lambda: _raise(error))
    return fbgemm_reference


def test_method_inventory_has_ten_unique_cells_and_optional_adapter_is_executable(tmp_path):
    assert len(benchmark.METHODS) == len(set(benchmark.METHODS)) == 9
    assert len(matrix.METHODS) == len(set(matrix.METHODS)) == 10
    assert "fbgemm" in benchmark.METHODS and "fbgemm" in matrix.RUN_METHODS
    assert "fbgemm" not in matrix.PENDING_METHODS
    assert set(matrix.PENDING_METHODS) == {"sparse-gpu"}
    args = _matrix_args(tmp_path, methods=matrix.METHODS)
    row = matrix.plan(args)["rows"][0]
    command_methods = row["command"][row["command"].index("--methods") + 1:]
    assert command_methods[:len(benchmark.METHODS)] == list(benchmark.METHODS)
    assert "sparse-gpu" not in command_methods


def test_run_duplicate_methods_reject_before_probing_torch(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["synthetic-host-cli", "--methods", "fbgemm", "fbgemm",
                                    "--output", str(tmp_path / "never-created")])
    monkeypatch.setattr(benchmark.importlib, "import_module", lambda _: _raise(
        AssertionError("duplicate methods must reject before any benchmark work")))
    with pytest.raises(SystemExit) as exc:
        benchmark.main()
    assert exc.value.code == 2
    assert not (tmp_path / "never-created").exists()


def test_matrix_duplicate_optional_methods_reject_before_output(tmp_path):
    args = _matrix_args(tmp_path, methods=("fbgemm", "fbgemm"))
    with pytest.raises(ValueError):
        matrix.run_matrix(args)
    assert not args.output.exists()


def test_absent_optional_distribution_is_unmeasured_partial_exit_two(
        tmp_path, monkeypatch, synthetic_cuda):
    from baselines.fbgemm_reference import OptionalDependencyUnavailable
    _optional_fault(monkeypatch, OptionalDependencyUnavailable("synthetic absent fbgemm-gpu distribution"))
    report, code = benchmark.run(_run_args(tmp_path / "run"))
    assert (report["status"], code, report["measured_method_count"]) == ("partial", 2, 0)
    cell, = report["methods"]
    assert cell["status"] == "unmeasured" and "absent" in cell["reason"]
    assert "oracle" not in cell and "complete_call_wall_until_complete_ms" not in cell
    assert synthetic_cuda[1] == []


def test_actual_optional_loader_absence_reaches_benchmark_unmeasured_status(
        tmp_path, monkeypatch, synthetic_cuda):
    from baselines import fbgemm_reference
    fbgemm_reference._load.cache_clear()
    def absent(name):
        assert name == "fbgemm-gpu"
        raise fbgemm_reference.metadata.PackageNotFoundError(name)
    monkeypatch.setattr(fbgemm_reference.metadata, "version", absent)
    try:
        report, code = benchmark.run(_run_args(tmp_path / "run"))
    finally:
        fbgemm_reference._load.cache_clear()
    assert (report["status"], code) == ("partial", 2)
    assert report["methods"][0]["status"] == "unmeasured"
    assert "not installed" in report["methods"][0]["reason"]
    assert synthetic_cuda[1] == []


def test_actual_optional_loader_wrong_release_reaches_failed_status(
        tmp_path, monkeypatch, synthetic_cuda):
    from baselines import fbgemm_reference
    fbgemm_reference._load.cache_clear()
    monkeypatch.setattr(fbgemm_reference.metadata, "version", lambda _: "9.9.0")
    try:
        report, code = benchmark.run(_run_args(tmp_path / "run"))
    finally:
        fbgemm_reference._load.cache_clear()
    assert (report["status"], code) == ("failed", 1)
    assert report["methods"][0]["status"] == "failed"
    assert "9.9.0" in report["methods"][0]["error"]
    assert synthetic_cuda[1] == []


@pytest.mark.parametrize("operation", ["sum", "max"])
def test_known_unsupported_operation_is_not_applicable_even_with_missing_optional_package(
        tmp_path, monkeypatch, synthetic_cuda, operation):
    from baselines.fbgemm_reference import OptionalDependencyUnavailable
    calls = []
    from baselines import fbgemm_reference
    def must_not_load():
        calls.append("load")
        raise OptionalDependencyUnavailable("synthetic absent package")
    monkeypatch.setattr(fbgemm_reference, "toolchain", must_not_load)
    report, code = benchmark.run(_run_args(tmp_path / "run", op=operation))
    assert (report["status"], code) == ("no_applicable_methods", 2)
    assert report["methods"][0]["status"] == "not_applicable"
    assert "softmax" in report["methods"][0]["reason"]
    assert calls == [] and synthetic_cuda[1] == []


@pytest.mark.parametrize("error", [
    RuntimeError("synthetic wrong FBGEMM_GPU version 9.9"),
    ImportError("synthetic installed package import failure"),
    AttributeError("synthetic missing jagged_softmax symbol"),
])
def test_installed_optional_backend_faults_are_failed_exit_one_not_absent(
        tmp_path, monkeypatch, synthetic_cuda, error):
    _optional_fault(monkeypatch, error)
    report, code = benchmark.run(_run_args(tmp_path / "run"))
    assert (report["status"], code) == ("failed", 1)
    cell, = report["methods"]
    assert cell["status"] == "failed" and str(error) in cell["error"]
    assert (tmp_path / "run" / "fbgemm.stderr.txt").is_file()
    assert synthetic_cuda[1] == []


@pytest.mark.parametrize("failure", ["oracle", "runtime"])
def test_optional_first_call_failure_preserves_preparation_evidence_and_prevents_timing(
        tmp_path, monkeypatch, synthetic_cuda, failure):
    from baselines import fbgemm_reference
    torch, measured = synthetic_cuda
    metadata = {"synthetic_host_fixture": True, "fbgemm_gpu": "synthetic release identity",
                "source_sha256": "a" * 64,
                "native_libraries": [{"path": "/synthetic/not-a-library.so", "sha256": "b" * 64}]}
    monkeypatch.setattr(fbgemm_reference, "toolchain", lambda: deepcopy(metadata))
    def faulty(values, *_args, **_kwargs):
        if failure == "runtime":
            raise RuntimeError("synthetic foreign kernel runtime failure")
        return torch.zeros_like(values)
    monkeypatch.setattr(fbgemm_reference, "call", faulty)
    report, code = benchmark.run(_run_args(tmp_path / "run"))
    assert (report["status"], code) == ("failed", 1)
    cell, = report["methods"]
    assert cell["status"] == "failed"
    for key, value in metadata.items():
        assert cell[key] == value
    assert cell["max_L"] == 8 and cell["max_L_source"] == "shape_N_upper_bound"
    assert cell["known_risks"] == list(fbgemm_reference.KNOWN_RISKS)
    assert cell["device_semantics"] == "unqualified"
    assert "no truncated segments" in cell["baseline_strategy"]
    assert measured == []


@pytest.mark.parametrize("dtype", ["f32", "bf16"])
def test_optional_complete_call_routes_f32_scale_mask_bounds_and_original_output_dtype(
        tmp_path, monkeypatch, dtype):
    torch = pytest.importorskip("torch")
    from baselines import fbgemm_reference
    from oracle.masked import assert_masked_softmax_matches
    values = torch.tensor([[1.0, -2.0], [2.0, -3.0], [-1.0, 4.0]],
                          dtype=torch.float32 if dtype == "f32" else torch.bfloat16)
    offsets = torch.tensor([0, 0, 1, 3], dtype=torch.int64)
    mask = torch.tensor([[0.125, -0.2], [0.125, -0.3], [-1.0, -float("inf")]])
    args = SimpleNamespace(op="softmax", scale=0.1234567, length_bounds={"min_len": 0, "max_len": 2})
    monkeypatch.setattr(fbgemm_reference, "toolchain", lambda: {"synthetic_host_fixture": True})
    captured = []
    def foreign_call(prepared, bounds, operation, *, max_len):
        captured.append((prepared.clone(), bounds, operation, max_len))
        return _cpu_softmax(torch, prepared, bounds, operation)
    monkeypatch.setattr(fbgemm_reference, "call", foreign_call)
    call, metadata = benchmark.make_call("fbgemm", args, torch, values, offsets, tmp_path, mask)
    assert captured == []
    for _ in range(2):
        actual = call()
        assert actual.dtype == values.dtype
        assert_masked_softmax_matches(actual.float().numpy(), values.float().numpy(), offsets.numpy(), mask.numpy(),
                                      scale=args.scale, input_dtype=dtype)
    assert len(captured) == 2
    for prepared, bounds, operation, max_len in captured:
        assert prepared.dtype == torch.float32 and bounds is offsets
        assert operation == "softmax" and max_len == 2
        expected = values.float() * np.float32(args.scale) + mask
        assert torch.equal(prepared, expected)
    assert metadata["max_L"] == 2 and metadata["max_L_source"] == "caller_verified_length_bound"


def test_measured_other_method_and_missing_optional_keep_same_inputs_and_partial_status(
        tmp_path, monkeypatch, synthetic_cuda):
    from baselines.fbgemm_reference import OptionalDependencyUnavailable
    _optional_fault(monkeypatch, OptionalDependencyUnavailable("synthetic absent optional backend"))
    original_make_call = benchmark.make_call
    torch, measured = synthetic_cuda
    captures = []
    def calls(method, args, supplied_torch, values, offsets, destination, mask=None):
        captures.append((method, values, offsets))
        if method == "triton":
            return lambda: _cpu_softmax(torch, values, offsets), {"synthetic_host_fixture": True}
        return original_make_call(method, args, supplied_torch, values, offsets, destination, mask)
    monkeypatch.setattr(benchmark, "make_call", calls)
    report, code = benchmark.run(_run_args(tmp_path / "run", methods=("triton", "fbgemm")))
    assert (report["status"], code, report["measured_method_count"]) == ("partial", 2, 1)
    assert [cell["status"] for cell in report["methods"]] == ["measured", "unmeasured"]
    assert measured == ["triton"]
    assert captures[0][1] is captures[1][1] and captures[0][2] is captures[1][2]
    with np.load(tmp_path / "run" / "case.npz", allow_pickle=False) as saved:
        assert np.array_equal(saved["values"], captures[0][1].numpy())
        assert np.array_equal(saved["offsets"], captures[0][2].numpy())


def test_failed_cell_takes_precedence_over_absent_optional_partial(
        tmp_path, monkeypatch, synthetic_cuda):
    from baselines.fbgemm_reference import OptionalDependencyUnavailable
    _optional_fault(monkeypatch, OptionalDependencyUnavailable("synthetic absent optional backend"))
    original_make_call = benchmark.make_call
    def calls(method, *args, **kwargs):
        if method == "triton":
            raise RuntimeError("synthetic unrelated method failure")
        return original_make_call(method, *args, **kwargs)
    monkeypatch.setattr(benchmark, "make_call", calls)
    report, code = benchmark.run(_run_args(tmp_path / "run", methods=("fbgemm", "triton")))
    assert (report["status"], code) == ("failed", 1)
    assert [cell["status"] for cell in report["methods"]] == ["unmeasured", "failed"]


def _matrix_args(tmp_path, *, methods=("triton", "ragged-auto", "fbgemm")):
    return matrix.parser().parse_args([
        "--output", str(tmp_path / "matrix"), "--ops", "softmax", "--dtypes", "f32", "--ds", "16",
        "--offsets-dtypes", "i64", "--distributions", "uniform", "--normalizations", "fixed_elements",
        "--sizes", "small", "--small-bytes", "2048", "--large-bytes", "4096", "--mean-length", "4",
        "--methods", *methods, "--warmup", "0", "--repeats", "3",
    ])


def _ready():
    return {"status": "ready", "device": deepcopy(SYNTHETIC_DEVICE),
            "torch": "synthetic-host-fixture-torch", "torch_cuda": "synthetic-host-fixture-cuda"}


def _measured(method, finite_values, milliseconds):
    metrics = _synthetic_metrics()
    metrics["complete_call_wall_until_complete_ms"] = _summary(milliseconds)
    metrics["complete_call_cuda_event_ms"] = _summary(milliseconds / 2)
    return {"method": method, "status": "measured", **metrics,
            "oracle": {"finite_values": finite_values, "nan_values": 0,
                       "max_abs_error": 0.0, "max_fraction_of_tolerance": 0.0}}


def _matrix_fixture(tmp_path, *, methods=("triton", "ragged-auto", "fbgemm"),
                    mutation=None, child_status="partial", child_code=2):
    args = _matrix_args(tmp_path, methods=methods)
    planned = matrix.plan(args)
    case = planned["rows"][0]["case"]
    finite_values = case["elements"] * case["D"]
    cells = [({"method": method, "status": "unmeasured",
               "reason": "synthetic missing optional distribution; genuine retained child result",
               "optional_dependency": {"name": "fbgemm-gpu", "state": "absent"}}
              if method == "fbgemm" else _measured(method, finite_values, 2.0 if method == "triton" else 4.0))
             for method in methods]
    if mutation == "omitted":
        cells = [cell for cell in cells if cell["method"] != "fbgemm"]
    elif mutation == "duplicate":
        cells.append(deepcopy(next(cell for cell in cells if cell["method"] == "fbgemm")))
    elif mutation == "all_measured":
        cells = [_measured(method, finite_values, 2.0 if method == "triton" else 4.0)
                 for method in methods]
    elif mutation == "empty_reason":
        next(cell for cell in cells if cell["method"] == "fbgemm")["reason"] = ""
    children = []
    def runner(command, **kwargs):
        destination = Path(command[command.index("--output") + 1])
        destination.mkdir(parents=True)
        lengths = make_lengths("uniform", case["segments"], seed=case["seed"],
                               total_elements=case["elements"], mean_length=case["mean_length"])
        offsets = offsets_from_lengths(lengths, "i64")
        values = np.zeros((case["elements"], case["D"]), dtype=np.float32)
        observed = {key: case[key] for key in ("op", "dtype", "D", "offsets_dtype", "distribution",
                    "normalization", "segments", "elements", "seed", "histogram_sha256")}
        observed.update(describe(lengths), mean_length_parameter=case["mean_length"],
                        values_sha256=hashlib.sha256(values.tobytes()).hexdigest(),
                        offsets_sha256=hashlib.sha256(offsets.tobytes()).hexdigest(),
                        scale=1.0, mask_sha256=None, disabled_optimizations=[], feature_tile=None,
                        length_bounds={}, values_bytes=values.nbytes, offsets_bytes=offsets.nbytes,
                        mask_bytes=0, input_bytes=values.nbytes + offsets.nbytes)
        np.savez(destination / "case.npz", values=values, offsets=offsets)
        child = {"schema": 1, "status": child_status, "case": observed, "methods": deepcopy(cells),
                 "device": deepcopy(SYNTHETIC_DEVICE), "torch": _ready()["torch"],
                 "torch_cuda": _ready()["torch_cuda"], "python": "synthetic-host-fixture-python",
                 "harness_sha256": planned["benchmark_harness_sha256"],
                 "command": [str(Path("bench/run.py").resolve()), *command[3:]],
                 "measured_method_count": sum(cell["status"] == "measured" for cell in cells)}
        (destination / "results.json").write_text(json.dumps(child, allow_nan=False) + "\n")
        children.append(child)
        return SimpleNamespace(returncode=child_code, stdout="synthetic host child stdout\n", stderr="")
    manifest, code = matrix.run_matrix(args, preflight_fn=_ready, runner=runner)
    return args.output, manifest, children[0], code


def test_matrix_preserves_actual_child_unmeasured_and_same_case_measurements(tmp_path):
    path, manifest, child, code = _matrix_fixture(tmp_path)
    row, = manifest["rows"]
    assert (manifest["status"], row["status"], code, row["exit_code"]) == ("partial", "partial", 2, 2)
    assert row["methods"] == child["methods"]
    assert row["observed_case"] == child["case"]
    assert row["results_sha256"] == hashlib.sha256((path / row["results_path"]).read_bytes()).hexdigest()
    assert row["methods"][-1]["optional_dependency"]["state"] == "absent"


def test_optional_only_actual_unmeasured_is_not_complete_in_matrix_or_report(tmp_path):
    path, manifest, child, code = _matrix_fixture(tmp_path, methods=("fbgemm",))
    assert code == 2 and manifest["status"] == "partial"
    report = build_report(path / "matrix.json", reference_method="fbgemm")
    assert report["status"] == "partial" and report_exit_code(report) == 2
    cell, = report["rows"]
    assert cell["source_method"] == child["methods"][0] and cell["metrics"] is None
    assert cell["speedup_wall"] is None and cell["speedup_event"] is None


@pytest.mark.parametrize("mutation", ["omitted", "duplicate"])
def test_missing_or_duplicate_optional_child_cell_is_failure_not_valid_absence(tmp_path, mutation):
    _, manifest, _, code = _matrix_fixture(tmp_path, mutation=mutation)
    assert code == 1 and manifest["status"] == manifest["rows"][0]["status"] == "failed"
    if mutation == "omitted":
        assert "omitted" in manifest["rows"][0]["reason"]


@pytest.mark.parametrize("child_status,child_code", [("partial", 0), ("complete", 2)])
def test_optional_partial_status_cannot_hide_wrong_child_exit_code(tmp_path, child_status, child_code):
    _, manifest, _, code = _matrix_fixture(tmp_path, child_status=child_status, child_code=child_code)
    assert code == 1 and manifest["status"] == "failed"


@pytest.mark.parametrize("mutation,child_status,child_code", [
    ("all_measured", "partial", 2), (None, "complete", 0),
])
def test_optional_child_status_must_agree_with_its_actual_unmeasured_inventory(
        tmp_path, mutation, child_status, child_code):
    _, manifest, _, code = _matrix_fixture(tmp_path, mutation=mutation,
                                          child_status=child_status, child_code=child_code)
    assert code == 1 and manifest["status"] == manifest["rows"][0]["status"] == "failed"


def test_actual_optional_unmeasured_requires_a_retained_nonempty_reason(tmp_path):
    path, manifest, _, code = _matrix_fixture(tmp_path, mutation="empty_reason")
    assert code == 1 and manifest["status"] == "failed"
    with pytest.raises(ReportError):
        build_report(path / "matrix.json", reference_method="triton")


def test_report_retains_actual_optional_unmeasured_without_disturbing_same_case_ratios(tmp_path):
    path, manifest, child, _ = _matrix_fixture(tmp_path)
    report = build_report(path / "matrix.json", reference_method="triton")
    assert (report["status"], report_exit_code(report)) == ("partial", 2)
    assert report["method_status_counts"] == {"measured": 2, "unmeasured": 1}
    cells = {cell["method"]: cell for cell in report["rows"]}
    assert cells["ragged-auto"]["speedup_wall"] == cells["ragged-auto"]["speedup_event"] == 0.5
    optional = cells["fbgemm"]
    assert optional["source_method"] == child["methods"][-1]
    assert optional["reason"] == child["methods"][-1]["reason"]
    assert optional["metrics"] is None and optional["speedup_wall"] is optional["speedup_event"] is None
    assert optional["identity"] == cells["triton"]["identity"] == cells["ragged-auto"]["identity"]
    assert optional["source_results_sha256"] == manifest["rows"][0]["results_sha256"]


def test_report_distinguishes_missing_optional_child_evidence_from_retained_absence(tmp_path):
    path, manifest, _, _ = _matrix_fixture(tmp_path, mutation="omitted")
    report = build_report(path / "matrix.json", reference_method="triton")
    optional = next(cell for cell in report["rows"] if cell["method"] == "fbgemm")
    assert (report["status"], report_exit_code(report)) == ("failed", 1)
    assert optional["source_method"] is None and optional["metrics"] is None
    assert "no result" in optional["reason"]
    assert manifest["rows"][0]["status"] == "failed"


def test_report_rejects_unknown_status_in_optional_child_result(tmp_path):
    path, manifest, child, _ = _matrix_fixture(tmp_path)
    child["methods"][-1]["status"] = "available_but_unqualified"
    row = manifest["rows"][0]
    payload = (json.dumps(child, allow_nan=False) + "\n").encode()
    (path / row["results_path"]).write_bytes(payload)
    row["results_sha256"] = hashlib.sha256(payload).hexdigest()
    row["methods"] = deepcopy(child["methods"])
    (path / "matrix.json").write_text(json.dumps(manifest, allow_nan=False) + "\n")
    with pytest.raises(ReportError):
        build_report(path / "matrix.json", reference_method="triton")
