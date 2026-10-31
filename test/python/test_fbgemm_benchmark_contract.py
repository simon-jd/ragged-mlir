"""Independent host routing tests, never FBGEMM or CUDA numerical evidence.

Real CPU Torch tensors inspect preprocessing. Foreign calls and GPU preflight
are explicit test doubles; every measurement entry point is forbidden.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from baselines import fbgemm_reference as adapter
from bench import matrix, report as aggregate, run as benchmark


def forbidden(*args, **kwargs):
    raise AssertionError("foreign execution, fallback or measurement is forbidden")


def provenance():
    return {"fbgemm_gpu": "1.8.0", "source_sha256": "synthetic-host-fixture",
            "native_libraries": [{"path": "synthetic-not-a-library.so", "sha256": "not-device-evidence"}],
            "known_risks": list(adapter.KNOWN_RISKS), "device_semantics": "unqualified"}


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("offset_dtype", ["int32", "int64"])
@pytest.mark.parametrize("hinted", [False, True])
def test_preprocessing_is_inside_every_call_and_bound_is_shared(tmp_path, monkeypatch, dtype, offset_dtype, hinted):
    torch = pytest.importorskip("torch")
    values = torch.tensor([[1.03125, -2.25], [1.0078125, 3.75], [4.25, -1.5]], dtype=getattr(torch, dtype))
    offsets = torch.tensor([0, 0, 2, 3], dtype=getattr(torch, offset_dtype))
    mask = torch.tensor([[0.0007, -0.0003], [0.0002, 0.0009], [-0.0005, 0.0001]], dtype=torch.float32)
    args = SimpleNamespace(op="softmax", scale=0.1234567,
                           length_bounds={"min_len": 0, "max_len": 2} if hinted else {})
    observed = []
    def invoke(work, actual_offsets, operation, *, max_len):
        observed.append((work.clone(), actual_offsets, operation, max_len))
        return work + torch.tensor(0.00023, dtype=torch.float32)
    monkeypatch.setattr(adapter, "toolchain", provenance)
    monkeypatch.setattr(adapter, "call", invoke)
    call, details = benchmark.make_call("fbgemm", args, torch, values, offsets, tmp_path, mask)
    assert observed == []  # No preprocessing or foreign execution in preparation.
    assert details["max_L"] == (2 if hinted else 3)
    assert details["max_L_source"] == ("caller_verified_length_bound" if hinted else "shape_N_upper_bound")
    for _ in range(2):
        # An updated mask must be observed on the next call, not captured as a
        # precomputed tensor during preparation or the first invocation.
        expected_work = values.float() * float(np.float32(args.scale)) + mask
        result = call()
        actual_work, actual_offsets, operation, max_len = observed[-1]
        assert actual_work.dtype == torch.float32
        assert torch.equal(actual_work, expected_work)
        assert actual_offsets is offsets and actual_offsets.dtype == getattr(torch, offset_dtype)
        assert operation == "softmax" and max_len == (2 if hinted else None)
        assert result.dtype == values.dtype
        assert torch.equal(result, (expected_work + 0.00023).to(values.dtype))
        mask.add_(0.00011)
    assert len(observed) == 2


@pytest.mark.parametrize("operation", ["sum", "max"])
def test_known_unsupported_operation_is_not_applicable_without_loading_package(tmp_path, monkeypatch, operation):
    monkeypatch.setattr(adapter, "toolchain", forbidden)
    with pytest.raises(benchmark.NotApplicable, match="no sum or max"):
        benchmark.make_call("fbgemm", SimpleNamespace(op=operation, scale=1.0), None, None, None, tmp_path)


@pytest.mark.parametrize("n,d,b,bound", [(adapter.MAX_LENGTH + 1, 1, 1, None),
                                       (2**21, 1024, 1, None), (1, 1, 2**31-1, None),
                                       (1, 1, 1, 0)])
def test_known_shape_limits_refuse_before_optional_dependency_load(tmp_path, monkeypatch, n, d, b, bound):
    monkeypatch.setattr(adapter, "toolchain", forbidden)
    values = SimpleNamespace(shape=(n, d))
    offsets = SimpleNamespace(numel=lambda: b+1)
    args = SimpleNamespace(op="softmax", scale=1.0, length_bounds={} if bound is None else {"max_len": bound})
    with pytest.raises(benchmark.NotApplicable, match="shape/bound limitation"):
        benchmark.make_call("fbgemm", args, None, values, offsets, tmp_path)


def host_pipeline(tmp_path, monkeypatch, *, operation="softmax"):
    """Exercise status persistence with CPU tensors and an explicit fake device.

    This tests the harness's control flow only; no timing function may run.
    """
    torch = pytest.importorskip("torch")
    class CPUTransfer:
        def __init__(self, tensor): self.tensor = tensor
        def to(self, device, dtype=None):
            assert device == "cuda"
            return self.tensor.to(dtype=dtype) if dtype else self.tensor
    fake = SimpleNamespace(
        __version__="synthetic-status-contract-not-Torch-CUDA", version=SimpleNamespace(cuda="synthetic"),
        float32=torch.float32, bfloat16=torch.bfloat16,
        from_numpy=lambda x: CPUTransfer(torch.from_numpy(x)),
        cuda=SimpleNamespace(is_available=lambda: True, current_device=lambda: 0, synchronize=lambda: None,
            get_device_properties=lambda ordinal: SimpleNamespace(major=8, minor=9,
                name="synthetic status contract only", total_memory=1024, multi_processor_count=1)))
    actual_import = benchmark.importlib.import_module
    monkeypatch.setattr(benchmark.importlib, "import_module", lambda name, *a, **k: fake if name == "torch" else actual_import(name, *a, **k))
    monkeypatch.setattr(benchmark, "cuda_driver_info", lambda ordinal: {"scope": "synthetic host fixture"})
    monkeypatch.setattr(benchmark, "measure", forbidden)
    monkeypatch.setattr(adapter, "toolchain", provenance)
    monkeypatch.setattr(adapter, "call", forbidden)
    args = SimpleNamespace(op=operation, dtype="f32", offsets_dtype="i64", distribution="uniform", segments=2,
        total_elements=4, d=2, mean_length=2, seed=711, scale=1.0, mask=False, length_hints=False,
        disable_optimization=[], feature_tile=None, methods=["fbgemm"], output=tmp_path, warmup=0, repeats=1)
    return torch, args


@pytest.mark.parametrize("failure", ["oracle", "runtime"])
def test_failed_numeric_or_runtime_case_retains_provenance_without_timing(tmp_path, monkeypatch, failure):
    torch, args = host_pipeline(tmp_path, monkeypatch)
    def call(work, offsets, operation, *, max_len):
        if failure == "runtime":
            raise RuntimeError("synthetic foreign execution failure")
        return torch.zeros_like(work)  # Deliberately fails the unchanged softmax oracle.
    monkeypatch.setattr(adapter, "call", call)
    result, code = benchmark.run(args)
    cell = result["methods"][0]
    assert code == 1 and result["status"] == cell["status"] == "failed"
    for key, value in provenance().items():
        assert cell[key] == value
    assert cell["max_L"] == 4 and cell["max_L_source"] == "shape_N_upper_bound"
    assert "baseline_strategy" in cell
    assert "complete_call_wall_until_complete_ms" not in cell and "complete_call_cuda_event_ms" not in cell
    assert (tmp_path / "fbgemm.stderr.txt").is_file()
    assert json.loads((tmp_path / "results.json").read_text())["methods"][0] == cell


@pytest.mark.parametrize("failure,expected_status,expected_code", [
    ("absent", "unmeasured", 2), ("broken_import", "failed", 1), ("unimplemented_op", "not_applicable", 2)])
def test_absence_setup_failure_and_known_inapplicability_remain_distinct(tmp_path, monkeypatch, failure, expected_status, expected_code):
    _, args = host_pipeline(tmp_path, monkeypatch, operation="sum" if failure == "unimplemented_op" else "softmax")
    def toolchain():
        if failure == "absent":
            raise adapter.OptionalDependencyUnavailable("synthetic missing optional package")
        if failure == "broken_import":
            raise ImportError("synthetic installed package undefined symbol")
        forbidden()
    monkeypatch.setattr(adapter, "toolchain", toolchain)
    result, code = benchmark.run(args)
    cell = result["methods"][0]
    assert code == expected_code and cell["status"] == expected_status
    assert result["status"] == {"absent":"partial", "broken_import":"failed", "unimplemented_op":"no_applicable_methods"}[failure]
    assert "oracle" not in cell and "complete_call_wall_until_complete_ms" not in cell


@pytest.mark.parametrize("status,exit_code,cell_status,accepted", [
    ("partial", 2, "unmeasured", True), ("complete", 0, "unmeasured", False),
    ("partial", 0, "unmeasured", False), ("partial", 2, "not_applicable", False)])
def test_matrix_retains_optional_missing_cell_without_reclassifying_it(tmp_path, status, exit_code, cell_status, accepted):
    args = matrix.parser().parse_args(["--output", str(tmp_path), "--ops", "softmax", "--ds", "16",
        "--dtypes", "f32", "--offsets-dtypes", "i64", "--distributions", "uniform",
        "--normalizations", "fixed_elements", "--sizes", "small", "--methods", "fbgemm"])
    assert "fbgemm" not in matrix.PENDING_METHODS and "fbgemm" in matrix.RUN_METHODS
    def runner(command, **kwargs):
        directory = Path(command[command.index("--output")+1])
        directory.mkdir(parents=True)
        child = {"schema":1, "status":status, "case": {"fixture":"no GPU evidence"},
                 "methods":[{"method":"fbgemm", "status":cell_status, "reason":"synthetic optional dependency"}]}
        (directory/"results.json").write_text(json.dumps(child))
        return SimpleNamespace(returncode=exit_code, stdout="synthetic child fixture", stderr="")
    result, code = matrix.run_matrix(args, preflight_fn=lambda: {"status":"ready", "scope":"synthetic host fixture"}, runner=runner)
    assert code == (2 if accepted else 1)
    assert result["rows"][0]["status"] == ("partial" if accepted else "failed")
    if accepted:
        aggregated = aggregate.build_report(tmp_path/"matrix.json", reference_method="fbgemm")
        assert aggregate.report_exit_code(aggregated) == 2
        cell = aggregated["rows"][0]
        assert cell["status"] == "unmeasured" and cell["reason"] == "synthetic optional dependency"
        assert cell["metrics"] is None and cell["speedup_wall"] is None and cell["speedup_event"] is None
