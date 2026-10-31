"""Independent sparse/CUB comparison contracts, using synthetic timing proofs.

All device-execution flags and timing samples constructed here are synthetic
test premises, never GPU measurements or scope-decision evidence. Small NumPy inputs
and scalar scheduling spies verify host contracts only.
"""
from copy import deepcopy
import hashlib
import math
from pathlib import Path
import json
import subprocess
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from scripts import sparse_comparison_support as sparse


def _timing(milliseconds=1.0):
    return {"samples_ms": [milliseconds] * 3, "count": 3, "median_ms": milliseconds,
            "warmup_calls": 1,
            "scope": "complete resident-input call through all-stream GPU completion; includes host work and transfers; no process/JIT startup"}


def _proof(max_ratio=1.0, softmax_ratio=1.0):
    identity = {key: hashlib.sha256(("synthetic-" + key).encode()).hexdigest()
                for key in ("values_sha256", "offsets_sha256", "coordinates_sha256")}
    identity.update(B=16384, N=1048576, D=1, dtype="f32", offsets_dtype="i64", stored_zero_count=4081)
    operations = {}
    for op, ratio in (("max", max_ratio), ("softmax", softmax_ratio)):
        operations[op] = {
            method: {"status": "measured", "oracle_passed": True, "input": deepcopy(identity),
                     "domain": sparse.DOMAIN, "costs": list(costs),
                     "timing": _timing(ratio if method == "candidate" else 1.0)}
            for method, costs in (("candidate", sparse.CANDIDATE_COSTS), ("cub", sparse.CUB_COSTS))}
    return {"synthetic_host_fixture": True, "actual_device_execution": True,
            "parent_artifact_validation": True, "backend": "upstream-sparse-gpu-hybrid",
            "all_complete_costs_included": True, "distribution_qualified": True, "warmup": 1, "repeats": 3,
            "input": identity, "cpu_oracle": {"max": True, "softmax": True},
            "gpu_semantic_oracle": {"max": True, "softmax": True}, "operations": operations}


@pytest.mark.parametrize("max_ratio,softmax_ratio,narrow", [
    (0.25, 0.5, True), (1.0, 1.0, True), (1.49, 1.499999, True),
    (1.5, 1.0, False), (1.0, 1.5, False), (2.0, 2.0, False),
])
def test_directed_ratio_treats_faster_candidate_as_good_enough_and_strict_boundary(
        max_ratio, softmax_ratio, narrow):
    result = sparse.scope_decision(_proof(max_ratio, softmax_ratio))
    assert result["status"] == "eligible"
    assert result["ratios_sparse_over_cub"] == {"max": max_ratio, "softmax": softmax_ratio}
    expected = "consider_sparse_mapping_extension" if narrow else "retain_separate_scope_for_this_workload"
    assert result["recommendation"] == expected
    assert result["threshold"] == 1.5 and "hybrid" in result["scope"]


@pytest.mark.parametrize("field,value", [
    ("actual_device_execution", False), ("actual_device_execution", 1),
    ("parent_artifact_validation", False), ("backend", "cpu"),
    ("backend", "pure-sparse-gpu"), ("all_complete_costs_included", False),
    ("distribution_qualified", False), ("distribution_qualified", 1),
    ("warmup", -1), ("warmup", True), ("repeats", 0), ("repeats", 3.0),
])
def test_host_only_unbound_artifacts_wrong_backend_or_missing_costs_never_get_ratio(field, value):
    proof = _proof()
    proof[field] = value
    result = sparse.scope_decision(proof)
    assert result["status"] == "pending" and result["reasons"]
    assert result["ratios_sparse_over_cub"] is None and result["recommendation"] is None


@pytest.mark.parametrize("op", ["max", "softmax"])
@pytest.mark.parametrize("method", ["candidate", "cub"])
@pytest.mark.parametrize("fault", ["unmeasured", "oracle", "input", "domain", "cost", "samples"])
def test_each_required_operation_and_method_must_qualify_independently(op, method, fault):
    proof = _proof()
    cell = proof["operations"][op][method]
    if fault == "unmeasured":
        cell["status"] = "unmeasured"
    elif fault == "oracle":
        cell["oracle_passed"] = False
    elif fault == "input":
        cell["input"]["values_sha256"] = "f" * 64
    elif fault == "domain":
        cell["domain"] = "host-resident input -> host-resident output"
    elif fault == "cost":
        cell["costs"].pop()
    else:
        cell["timing"]["samples_ms"].pop()
    result = sparse.scope_decision(proof)
    assert result["status"] == "pending" and result["reasons"]
    assert result["ratios_sparse_over_cub"] is None and result["recommendation"] is None


@pytest.mark.parametrize("op", ["max", "softmax"])
def test_missing_cpu_raw_oracle_keeps_whole_scope_decision_pending(op):
    proof = _proof()
    proof["cpu_oracle"][op] = False
    result = sparse.scope_decision(proof)
    assert result["status"] == "pending" and result["recommendation"] is None


@pytest.mark.parametrize("fault", ["empty", "zero", "negative", "nan", "infinite", "wrong-count", "wrong-median", "wrong-warmup"])
def test_incomplete_or_nonfinite_timing_is_not_performance_evidence(fault):
    timing = _timing()
    if fault == "empty":
        timing["samples_ms"] = []
    elif fault in ("zero", "negative", "nan", "infinite"):
        timing["samples_ms"][0] = {"zero": 0.0, "negative": -1.0, "nan": math.nan, "infinite": math.inf}[fault]
    elif fault == "wrong-count":
        timing["count"] = 2
    elif fault == "wrong-median":
        timing["median_ms"] = 0.1
    else:
        timing["warmup_calls"] = 0
    assert sparse.valid_timing(timing, 1, 3) is False


def test_hybrid_cost_inventory_retains_all_host_work_transfers_allocations_and_completion():
    required = {"input_D2H", "coordinate_construction", "CSR_assembly", "host_allocations",
                "upstream_GPU_allocations_and_transfers", "all_upstream_reduction_kernels",
                "host_exp_and_normalize_for_softmax", "packed_output_copy", "output_H2D", "all_stream_completion"}
    assert set(sparse.CANDIDATE_COSTS) == required
    assert "output_and_workspace_allocations" in sparse.CUB_COSTS
    assert "exp_and_normalize_kernels_for_softmax" in sparse.CUB_COSTS


def test_complete_call_measurement_encloses_call_and_completion_but_excludes_warmup(monkeypatch):
    events = []
    counter = iter(range(0, 6_000_000, 1_000_000))
    def clock():
        events.append("timer")
        return next(counter)
    def call():
        events.append("call")
        return object()
    def complete():
        events.append("complete")
    monkeypatch.setattr(sparse.time, "perf_counter_ns", clock)
    result = sparse.measure(call, complete, warmup=1, repeats=3)
    assert events == ["call", "complete"] + ["timer", "call", "complete", "timer"] * 3
    assert result["samples_ms"] == [1.0] * 3 and result["count"] == 3
    assert result["warmup_calls"] == 1 and "host work and transfers" in result["scope"]


def test_case_generation_is_deterministic_and_preserves_explicit_zeros_and_local_coordinates():
    values, offsets, coordinates = sparse.make_case(17, 513, 711)
    repeated = sparse.make_case(17, 513, 711)
    assert values.shape == (513, 1) and values.dtype == np.float32
    assert offsets.shape == (18,) and offsets.dtype == np.int64 and offsets[-1] == 513
    assert coordinates.shape == (513,) and coordinates.dtype == np.int64
    assert values[0, 0] == values[257, 0] == 0.0
    for index, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        np.testing.assert_array_equal(coordinates[start:end], np.arange(end - start, dtype=np.int64))
    for actual, expected in zip((values, offsets, coordinates), repeated):
        np.testing.assert_array_equal(actual, expected)
    identity = sparse.case_identity(values, offsets, coordinates)
    assert identity["stored_zero_count"] >= 2
    assert identity["values_sha256"] == hashlib.sha256(values.tobytes()).hexdigest()


@pytest.mark.parametrize("b,width", [(0, 1), (1, 0), (-1, 1), (True, 1), (1, 2**31), (1_000_001, 1)])
def test_static_csr_size_and_index_limit_rejection_happens_before_codegen(b, width):
    with pytest.raises(ValueError):
        sparse.render_module(b, width)


def test_rendered_source_uses_official_sparse_assemble_unary_reduce_and_borrowed_external_buffers():
    text = sparse.render_module(6, 2)
    assert "@B@" not in text and "@L@" not in text
    assert "tensor<6x2xf32, #CSR>" in text
    for token in ("sparse_tensor.assemble", "sparse_tensor.unary", "sparse_tensor.reduce", "llvm.emit_c_interface"):
        assert token in text
    assert "sparse softmax changed the explicit stored entry count" in text
    assert "memref<?xf32>" in text and "memref<?xi64>" in text


def test_nonhex_input_hash_cannot_support_eligible_scope_decision():
    proof = _proof()
    proof["input"]["values_sha256"] = "nothex!!" * 8
    for op in ("max", "softmax"):
        for method in ("candidate", "cub"):
            proof["operations"][op][method]["input"] = deepcopy(proof["input"])
    result = sparse.scope_decision(proof)
    assert result["status"] == "pending" and result["ratios_sparse_over_cub"] is None


def test_zero_requested_samples_and_empty_samples_reject_without_statistics_exception():
    timing = _timing()
    timing.update(samples_ms=[], count=0)
    assert sparse.valid_timing(timing, 1, 0) is False
    proof = _proof()
    proof["repeats"] = 0
    for cells in proof["operations"].values():
        for cell in cells.values():
            cell["timing"] = deepcopy(timing)
    result = sparse.scope_decision(proof)
    assert result["status"] == "pending" and result["ratios_sparse_over_cub"] is None


def test_kernel_only_timing_scope_cannot_be_labelled_complete_call_for_scope_decision():
    proof = _proof()
    for cells in proof["operations"].values():
        for cell in cells.values():
            cell["timing"]["scope"] = "kernel-only CUDA event duration; host/transfers excluded"
    result = sparse.scope_decision(proof)
    assert result["status"] == "pending" and result["ratios_sparse_over_cub"] is None


def test_serialized_cubin_decoder_accepts_mlir_ascii_and_hex_byte_representation(tmp_path):
    from scripts import run_sparse_comparison as runner
    # A synthetic byte blob checks literal decoding only. This is not a valid
    # linked CUDA program and does not qualify the upstream GPU code generator.
    blob = b"\x7fELF\x02\x01\x01\x00" + bytes(range(32))
    literal = "\\7FELF\\02\\01\\01\\00" + "".join(f"\\{value:02X}" for value in range(32))
    path = tmp_path / "synthetic-lowered.mlir"
    path.write_text('gpu.binary @synthetic [#gpu.object<bin = "' + literal + '">]\n')
    extracted = runner.extract_cubins(path, tmp_path)
    assert len(extracted) == 1
    assert Path(extracted[0]["path"]).read_bytes() == blob


@pytest.mark.parametrize("lengths,qualified", [
    ([1] * 98 + [33, 10001], True),  # Exactly 1% rare-long is admitted.
    ([1] * 197 + [33, 10001, 10002], True),
    ([1] * 97 + [33, 10001, 10002], False),
    ([31] * 50 + [33] * 49 + [10001], False),  # Strict majority boundary.
    ([31] * 51 + [33] * 48 + [10001], True),
    ([1] * 98 + [33, 10000], False),  # Must be strictly greater than 10k.
    ([1] * 99 + [10001], False),  # No CTA class.
    ([32] * 98 + [33, 10001], False),  # Warp class alone is not majority <32.
    ([1] * 15 + [33, 10001], False),  # Tiny oracle probe is not scope evidence.
])
def test_actual_offsets_control_distribution_eligibility_with_explicit_rare_and_majority_boundaries(lengths, qualified):
    offsets = np.concatenate([np.array([0], np.int64), np.cumsum(np.asarray(lengths, np.int64))])
    result = sparse.distribution_qualification(offsets)
    assert result["qualified"] is qualified
    assert result["B"] == len(lengths) and result["N"] == sum(lengths)
    assert result["short_lt32_count"] == sum(length < 32 for length in lengths)
    assert result["long_gt10000_count"] == sum(length > 10000 for length in lengths)
    assert "operational" in result["definition"]


def _execution_fixture(path, monkeypatch):
    """Synthetic CPU raw-output ingestion fixture; no library/AOT claim."""
    from scripts import run_sparse_comparison as runner
    from oracle import segment_max, segment_softmax
    values, offsets, coordinates = sparse.semantic_cases()["empty_nan_infinities_zero"]
    archive = path / "input.npz"
    np.savez(archive, values=values, offsets=offsets, coordinates=coordinates)
    inputs = {"long_tail": runner.file_identity(archive) | {"identity": sparse.case_identity(values, offsets, coordinates)}}
    source = {"sha256": "f" * 64, "files": {}, "synthetic_host_fixture": True}
    monkeypatch.setattr(runner, "source_identity", lambda: deepcopy(source))
    config = {"source": source, "inputs": inputs, "modules": {}}
    report = {"schema": 1, "mode": "cpu", "source": deepcopy(source), "inputs": deepcopy(inputs),
              "modules": {}, "actual_device_execution": False, "backend": "upstream-sparse-cpu-AOT",
              "cases": {"long_tail": {}}, "status": "complete", "synthetic_host_fixture": True}
    report["host_environment"] = {"python": runner.file_identity(sys.executable), "python_version": sys.version,
                                  "numpy_version": np.__version__, "numpy_module": runner.file_identity(np.__file__)}
    for op, function in (("max", segment_max), ("softmax", segment_softmax)):
        raw = path / f"long_tail-{op}-candidate.npy"
        with np.errstate(invalid="ignore", over="ignore"):
            np.save(raw, function(values, offsets).astype(np.float32), allow_pickle=False)
        cell = {"input": deepcopy(inputs["long_tail"]["identity"]), "output": runner.file_identity(raw),
                "status": "qualified", "oracle_passed": True}
        if op == "softmax":
            cell["structure"] = {"positions_sha256": sparse.array_hash(offsets),
                "coordinates_sha256": sparse.array_hash(coordinates), "stored_entry_count": len(values),
                "explicit_stored_zero_preserved": True}
        report["cases"]["long_tail"][op] = {"candidate": cell}
    result = path / "execution.json"
    result.write_text(json.dumps(report))
    return runner, config, report, result


def test_cpu_raw_arrays_are_recomputed_by_oracle_and_never_become_gpu_ratio(tmp_path, monkeypatch):
    runner, config, _, result = _execution_fixture(tmp_path, monkeypatch)
    qualified, retained = runner.validate_execution(result, config, "cpu")
    assert qualified == {"max": True, "softmax": True}
    assert retained["actual_device_execution"] is False
    decision = sparse.scope_decision(retained)
    assert decision["status"] == "pending" and decision["ratios_sparse_over_cub"] is None


@pytest.mark.parametrize("fault", ["short-output", "truncated-npy", "wrong-value", "wrong-dtype", "wrong-input-hash",
                                  "lost-explicit-zero", "measured-cpu", "cpu-as-gpu", "missing-output"])
def test_raw_output_eof_size_dtype_input_structure_and_domain_faults_cannot_qualify(tmp_path, monkeypatch, fault):
    runner, config, report, result = _execution_fixture(tmp_path, monkeypatch)
    cell = report["cases"]["long_tail"]["softmax"]["candidate"]
    raw = Path(cell["output"]["path"])
    if fault in ("short-output", "wrong-value", "wrong-dtype"):
        actual = np.load(raw, allow_pickle=False)
        if fault == "short-output":
            actual = actual[:-1]
        elif fault == "wrong-value":
            actual[0, 0] += 0.25
        else:
            actual = actual.astype(np.float64)
        np.save(raw, actual, allow_pickle=False)
        cell["output"] = runner.file_identity(raw)
    elif fault == "truncated-npy":
        raw.write_bytes(raw.read_bytes()[:-3])
        cell["output"] = runner.file_identity(raw)
    elif fault == "wrong-input-hash":
        cell["input"]["values_sha256"] = "0" * 64
    elif fault == "lost-explicit-zero":
        cell["structure"]["stored_entry_count"] -= 1
    elif fault == "measured-cpu":
        cell["status"] = "measured"
        cell["timing"] = _timing()
    elif fault == "cpu-as-gpu":
        report.update(mode="gpu", actual_device_execution=True)
    else:
        raw.unlink()
    result.write_text(json.dumps(report))
    with pytest.raises((runner.Failed, ValueError, OSError, AssertionError)):
        runner.validate_execution(result, config, "cpu")


def test_no_device_gpu_entrypoint_does_no_child_preparation_or_fake_measurements(tmp_path):
    from scripts import run_sparse_comparison as runner
    args = runner.parser().parse_args(["--output", str(tmp_path / "comparison"), "--mode", "gpu",
                                     "--llvm-root", str(tmp_path / "missing-sdk"), "--segments", "17", "--elements", "513"])
    observed = []
    def preflight(*, device):
        observed.append(device)
        return {"status": "blocked", "reason": "synthetic CUDA_ERROR_NO_DEVICE 100"}
    def forbidden(*args, **kwargs):
        pytest.fail("no device must not execute a preparation or measurement child")
    code = runner.run(args, preflight=preflight, runner=forbidden)
    report = json.loads((args.output / "comparison.json").read_text())
    assert code == 2 and report["status"] == "blocked" and observed == [0]
    assert report["stages"] == [] and sorted(path.name for path in args.output.iterdir()) == ["comparison.json"]
    assert report["scope_decision"]["ratios_sparse_over_cub"] is None
    assert report["scope_decision"]["recommendation"] is None


@pytest.mark.parametrize("option,value", [("--segments", "0"), ("--elements", str(2**31)), ("--seed", "-1"),
                                         ("--repeats", "0"), ("--timeout", "nan"), ("--max-host-bytes", "1")])
def test_invalid_size_index_or_sample_budget_rejects_before_output_and_preflight(tmp_path, option, value):
    from scripts import run_sparse_comparison as runner
    args = runner.parser().parse_args(["--output", str(tmp_path / "never-created"), option, value])
    with pytest.raises(ValueError):
        runner.run(args, preflight=lambda **kwargs: pytest.fail("invalid parameters must reject before hardware probe"))
    assert not args.output.exists()


@pytest.mark.parametrize("fault", ["nonzero", "timeout", "cuda-wrapper-error"])
def test_stage_process_failure_timeout_and_upstream_silent_error_preserve_failed_evidence(tmp_path, fault):
    from scripts import run_sparse_comparison as runner
    report = {"stages": []}
    def subprocess_fixture(argv, **kwargs):
        kwargs["stdout"].write(b"synthetic subprocess output; not GPU evidence\n")
        kwargs["stderr"].write(b"'cuLaunchKernel' failed with 'CUDA_ERROR_INVALID_VALUE'\n" if fault == "cuda-wrapper-error"
                               else b"synthetic stage failure/timeout\n")
        if fault == "timeout":
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
        return SimpleNamespace(returncode=9 if fault == "nonzero" else 0)
    recorder = runner.Recorder(tmp_path, report, {"synthetic_host_fixture": "true"}, 17, subprocess_fixture)
    name = "execute-gpu" if fault == "cuda-wrapper-error" else "synthetic-stage"
    with pytest.raises((runner.Failed, subprocess.TimeoutExpired)):
        recorder.run(name, ["synthetic-command"])
    stage, = report["stages"]
    assert stage["status"] == "failed" and stage["error"]
    assert "synthetic subprocess output" in Path(stage["stdout"]["path"]).read_text()
    assert stage["stdout"]["sha256"] == hashlib.sha256(Path(stage["stdout"]["path"]).read_bytes()).hexdigest()


@pytest.mark.parametrize("op", ["max", "softmax"])
def test_gpu_semantic_failure_does_not_reuse_cpu_qualification_for_scope_decision(op):
    proof = _proof()
    proof["gpu_semantic_oracle"][op] = False
    result = sparse.scope_decision(proof)
    assert result["status"] == "pending" and result["ratios_sparse_over_cub"] is None


def test_existing_output_and_source_parent_symlink_are_rejected_without_mutation(tmp_path, monkeypatch):
    from scripts import run_sparse_comparison as runner
    args = runner.parser().parse_args(["--output", str(tmp_path / "old-output")])
    args.output.mkdir()
    sentinel = args.output / "comparison.json"
    sentinel.write_text("synthetic previous evidence\n")
    with pytest.raises(ValueError):
        runner.run(args)
    assert sentinel.read_text() == "synthetic previous evidence\n"
    root = tmp_path / "synthetic-source"
    (root / "scripts").mkdir(parents=True)
    alias = tmp_path / "innocent-alias"
    alias.symlink_to(root / "scripts", target_is_directory=True)
    monkeypatch.setattr(runner, "ROOT", root)
    args.output = alias / "new-comparison"
    with pytest.raises(ValueError):
        runner.run(args)
    assert not (root / "scripts/new-comparison").exists()
