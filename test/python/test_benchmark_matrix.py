"""Host-only matrix orchestration contracts; fixtures are not GPU measurements."""
from collections import defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from bench import matrix


def arguments(tmp_path, *, methods=("ragged-naive", "ragged-auto"), plan_only=False):
    argv = ["--output", str(tmp_path / "output"), "--ops", "sum", "--dtypes", "f32", "--ds", "16",
            "--offsets-dtypes", "i64", "--distributions", "uniform", "--normalizations", "fixed_elements",
            "--sizes", "small", "--methods", *methods]
    if plan_only:
        argv.append("--plan-only")
    return matrix.parser().parse_args(argv)


def ready():
    return {"status": "ready", "device": {"name": "synthetic host test only", "capability": [8, 9]}}


def child_methods(command):
    result = []
    for value in command[command.index("--methods") + 1:]:
        if value.startswith("--"):
            break
        result.append(value)
    return result


def write_child(command, *, cells=None, status="complete", code=0):
    output = Path(command[command.index("--output") + 1])
    assert not output.exists()  # A cold child directory is independent of logs.
    output.mkdir(parents=True)
    observed = {"op": command[command.index("--op") + 1], "dtype": command[command.index("--dtype") + 1],
                "D": int(command[command.index("--d") + 1]), "values_sha256": "host-fixture-values",
                "offsets_sha256": "host-fixture-offsets", "scale": 1.0, "mask_sha256": None, "length_bounds": {}}
    report = {"status": status, "case": observed, "methods": cells if cells is not None else [
        {"method": method, "status": "measured", "oracle": {"passed": True},
         "complete_call_wall_until_complete_ms": {"median": 10 + index},
         "complete_call_cuda_event_ms": {"median": 7 + index}}
        for index, method in enumerate(child_methods(command))]}
    (output / "results.json").write_text(json.dumps(report))
    return SimpleNamespace(returncode=code, stdout="host fixture stdout\n", stderr="host fixture stderr\n")


def test_full_plan_covers_every_requested_axis_and_never_probes_gpu(tmp_path):
    args = matrix.parser().parse_args(["--preset", "full", "--plan-only", "--output", str(tmp_path / "matrix")])
    def forbidden(*args, **kwargs):
        raise AssertionError("plan-only must not probe GPU or execute child benchmarks")
    report, code = matrix.run_matrix(args, preflight_fn=forbidden, runner=forbidden)
    assert code == 0 and report["status"] == report["scope"] == "planned"
    assert report["case_count"] == 960 and report["method_cell_count"] == 9600
    expected = set(itertools.product(("sum", "max", "softmax"), ("f32", "bf16"), (1, 16, 128, 1024),
        ("i32", "i64"), (*matrix.DISTRIBUTIONS, "public_histogram"), matrix.NORMALIZATIONS, ("small", "large")))
    assert {(case["op"], case["dtype"], case["D"], case["offsets_dtype"], case["distribution"],
             case["normalization"], case["size"]) for case in (row["case"] for row in report["rows"])} == expected
    assert len({row["case_id"] for row in report["rows"]}) == 960
    assert not (args.output / "cases").exists()
    assert report["preflight"] is None and report["size_policy"]["l2_relation"] == "unmeasured"
    assert all(row["results_sha256"] is None and row["observed_case"] is None for row in report["rows"])
    assert all(cell["status"] in ("planned", "blocked", "unmeasured") for row in report["rows"] for cell in row["methods"])


def test_fixed_b_shares_scale_and_count_and_retains_impossible_size_cells(tmp_path):
    args = arguments(tmp_path, plan_only=True)
    args.ds, args.normalizations = [1024], ["fixed_segments"]
    args.distributions = [*matrix.DISTRIBUTIONS, "public_histogram"]
    report = matrix.plan(args)
    assert len({(row["case"]["resolved_b"], row["case"]["mean_length"]) for row in report["rows"]}) == 1
    assert len({row["case"]["elements"] for row in report["rows"]}) > 1
    burst = next(row for row in report["rows"] if row["case"]["distribution"] == "burst")
    assert burst["status"] == "blocked" and "size_budget" in burst["reason"]
    assert burst["case"]["resolved_b"] == burst["case"]["mean_length"] == 1
    assert burst["case"]["input_bytes"] > burst["case"]["input_byte_budget"]
    for row in report["rows"]:
        case = row["case"]
        assert case["requested_b"] >= case["resolved_b"]
        assert case["requested_mean_length"] >= case["mean_length"]
        assert case["sizing_adjustments"]
        assert case["total_elements"] is None and "--fixed-segments" in row["command"]
        if row["status"] != "blocked":
            assert case["input_bytes"] <= case["input_byte_budget"]


def test_fixed_n_scales_by_bytes_not_fixed_huge_rows_and_is_shared_across_distributions(tmp_path):
    args = arguments(tmp_path, plan_only=True)
    args.ds, args.sizes, args.distributions = [1, 1024], ["small", "large"], list(matrix.DISTRIBUTIONS)
    report = matrix.plan(args)
    groups = defaultdict(list)
    for row in report["rows"]:
        case = row["case"]
        groups[case["D"], case["size"]].append(case)
        assert case["input_bytes"] == case["elements"] * case["D"] * 4 + (case["segments"] + 1) * 8
        assert case["input_bytes"] <= case["input_byte_budget"]
        assert "--fixed-segments" not in row["command"]
        assert row["command"][row["command"].index("--total-elements") + 1] == str(case["elements"])
    assert all(len({case["elements"] for case in cases}) == 1 for cases in groups.values())
    assert groups[1024, "large"][0]["elements"] < groups[1, "large"][0]["elements"] // 500


def test_one_command_shares_case_inputs_all_methods_and_excludes_pending_adapters(tmp_path):
    args = arguments(tmp_path, methods=matrix.METHODS)
    args.distributions = ["public_histogram"]
    row = matrix.plan(args)["rows"][0]
    assert child_methods(row["command"]) == list(matrix.RUN_METHODS)
    assert "--histogram" in row["command"] and "--distribution" not in row["command"]
    assert row["case"]["histogram_sha256"] == hashlib.sha256(args.histogram.read_bytes()).hexdigest()
    for cell in row["methods"]:
        if cell["method"] in matrix.PENDING_METHODS:
            assert cell["status"] == "unmeasured" and "pending" in cell["reason"]
    assert row["directory"] == f"cases/{row['case_id']}"
    assert row["results_path"] == row["directory"] + "/results.json"


def test_blocked_preflight_persists_every_cell_without_running_children(tmp_path):
    args = arguments(tmp_path, methods=matrix.METHODS)
    def forbidden(*args, **kwargs):
        raise AssertionError("blocked preflight must not spawn benchmarks")
    report, code = matrix.run_matrix(args, preflight_fn=lambda: {"status": "blocked", "reason": "no host-test GPU"}, runner=forbidden)
    assert code == 2 and report["status"] == "blocked"
    assert report["scope"] == "execution_attempted"
    assert not (args.output / "cases").exists() and not (args.output / "logs").exists()
    assert report == json.loads((args.output / "matrix.json").read_text())
    assert {cell["status"] for cell in report["rows"][0]["methods"]} == {"blocked", "unmeasured"}
    assert report["rows"][0]["exit_code"] is None


def test_success_preserves_actual_results_hash_logs_and_pending_partial_status(tmp_path):
    args = arguments(tmp_path, methods=("ragged-naive", "ragged-auto", "sparse-gpu"))
    calls = []
    def runner(command, **kwargs):
        calls.append((command, kwargs))
        return write_child(command)
    report, code = matrix.run_matrix(args, preflight_fn=ready, runner=runner)
    row = report["rows"][0]
    assert code == 2 and report["status"] == row["status"] == "partial"
    assert len(calls) == 1 and child_methods(calls[0][0]) == ["ragged-naive", "ragged-auto"]
    assert row["exit_code"] == 0
    actual = (args.output / row["results_path"]).read_bytes()
    assert row["results_sha256"] == hashlib.sha256(actual).hexdigest()
    assert row["observed_case"] == json.loads(actual)["case"]
    assert row["methods"][:2] == json.loads(actual)["methods"]
    assert row["methods"][1]["complete_call_wall_until_complete_ms"]["median"] == 11  # Retain a slower cell.
    assert (args.output / row["stdout_path"]).read_text() == "host fixture stdout\n"
    assert (args.output / row["stderr_path"]).read_text() == "host fixture stderr\n"
    assert calls[0][1]["cwd"] == matrix.ROOT and calls[0][1]["check"] is False
    assert row["cache_environment"] == {key: calls[0][1]["env"][key] for key in row["cache_environment"]}


def test_failed_cell_does_not_hide_next_case_or_retry_and_caches_are_isolated(tmp_path):
    args = arguments(tmp_path)
    args.ops = ["sum", "max"]
    calls = []
    def runner(command, **kwargs):
        calls.append((command, kwargs))
        if len(calls) == 1:
            return write_child(command, status="failed", code=1, cells=[
                {"method": "ragged-naive", "status": "failed", "error": "host fixture oracle failure"},
                {"method": "ragged-auto", "status": "not_applicable", "reason": "host fixture incompatibility"}])
        return write_child(command)
    report, code = matrix.run_matrix(args, preflight_fn=ready, runner=runner)
    assert code == 1 and report["status"] == "failed" and len(calls) == 2
    assert [row["status"] for row in report["rows"]] == ["failed", "complete"]
    assert report["rows"][0]["methods"][1]["status"] == "not_applicable"
    assert calls[0][1]["env"]["TORCHINDUCTOR_CACHE_DIR"] != calls[1][1]["env"]["TORCHINDUCTOR_CACHE_DIR"]
    original = (args.output / "matrix.json").read_bytes()
    with pytest.raises(ValueError, match="never overwrite"):
        matrix.run_matrix(args, preflight_fn=ready, runner=runner)
    assert len(calls) == 2 and (args.output / "matrix.json").read_bytes() == original


@pytest.mark.parametrize("fault", ["missing", "malformed", "duplicate_field", "nonfinite", "omitted_method", "duplicate_method", "unexpected_method", "nonzero", "timeout"])
def test_incomplete_child_evidence_is_visible_and_never_timed_as_success(tmp_path, fault):
    args = arguments(tmp_path)
    def runner(command, **kwargs):
        if fault == "timeout":
            raise subprocess.TimeoutExpired(command, 1, output=b"partial stdout", stderr=b"partial stderr")
        if fault == "missing":
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if fault in ("malformed", "duplicate_field", "nonfinite"):
            output = Path(command[command.index("--output") + 1])
            output.mkdir(parents=True)
            (output / "results.json").write_text("not JSON" if fault == "malformed"
                else '{"status":"complete", "methods":[], "invalid_time":NaN}' if fault == "nonfinite"
                else '{"status":"failed", "status":"complete", "methods":[]}')
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        cells = [{"method": "ragged-naive", "status": "not_applicable", "reason": "host fixture"}]
        if fault == "duplicate_method": cells *= 2
        if fault == "unexpected_method": cells.append({"method": "cub", "status": "not_applicable"})
        if fault == "nonzero": cells.append({"method": "ragged-auto", "status": "not_applicable"})
        return write_child(command, cells=cells, code=9 if fault == "nonzero" else 0)
    report, code = matrix.run_matrix(args, preflight_fn=ready, runner=runner)
    assert code == 1 and report["status"] == report["rows"][0]["status"] == "failed"
    assert all(cell["status"] != "measured" for cell in report["rows"][0]["methods"])
    if fault == "timeout":
        assert "TimeoutExpired" in report["rows"][0]["execution_error"]
        assert (args.output / report["rows"][0]["stdout_path"]).read_text() == "partial stdout"


def test_pending_only_never_runs_implicit_default_methods(tmp_path):
    args = arguments(tmp_path, methods=("sparse-gpu",))
    def forbidden(*args, **kwargs): raise AssertionError("no supported method should launch")
    report, code = matrix.run_matrix(args, preflight_fn=ready, runner=forbidden)
    assert code == 2 and report["status"] == "partial"
    assert report["rows"][0]["command"] == []
    assert all(cell["status"] == "unmeasured" for cell in report["rows"][0]["methods"])


def test_ablation_flags_are_forwarded_and_bound_to_case_identity(tmp_path):
    args = arguments(tmp_path)
    args.ops = ["softmax"]
    original = matrix.plan(args)["rows"][0]
    args.scale, args.mask, args.length_hints, args.feature_tile = 0.1234567, True, True, 4
    args.disable_optimization = ["classification", "pointwise_fusion"]
    row = matrix.plan(args)["rows"][0]
    command, case = row["command"], row["case"]
    assert row["case_id"] != original["case_id"]
    assert command[command.index("--scale") + 1] == str(args.scale)
    assert "--mask" in command and "--length-hints" in command
    assert command[command.index("--feature-tile") + 1] == "4"
    assert [command[index + 1] for index, value in enumerate(command) if value == "--disable-optimization"] == args.disable_optimization
    assert case["disabled_optimizations"] == args.disable_optimization and case["length_hints"] is True
    assert case["mask"] is True and case["feature_tile"] == 4 and case["requested_scale"] == args.scale
    assert case["mask_bytes"] == case["elements"] * case["D"] * 4
    assert case["input_bytes"] == case["values_bytes"] + case["mask_bytes"] + case["offsets_bytes"]
    assert case["input_bytes"] <= case["input_byte_budget"]
    assert case["elements"] < original["case"]["elements"]


@pytest.mark.parametrize("dtype", ["f32", "bf16"])
def test_mask_bytes_are_f32_independent_of_values_dtype_and_fixed_b_stays_shared(tmp_path, dtype):
    args = arguments(tmp_path)
    args.ops, args.dtypes, args.mask = ["softmax"], [dtype], True
    args.normalizations = ["fixed_segments"]
    args.distributions = [*matrix.DISTRIBUTIONS, "public_histogram"]
    rows = matrix.plan(args)["rows"]
    assert len({(row["case"]["segments"], row["case"]["mean_length"]) for row in rows}) == 1
    for row in rows:
        case = row["case"]
        assert case["mask_bytes"] == case["elements"] * case["D"] * 4
        assert case["values_bytes"] == case["elements"] * case["D"] * (4 if dtype == "f32" else 2)
        assert case["input_bytes"] == case["values_bytes"] + case["mask_bytes"] + case["offsets_bytes"]
        assert row["status"] == "blocked" or case["input_bytes"] <= case["input_byte_budget"]


def test_scale_identity_is_recorded_as_f32(tmp_path):
    args = arguments(tmp_path)
    args.scale = 1.0 + 1e-8
    case = matrix.plan(args)["rows"][0]["case"]
    assert case["scale"] == 1.0 and case["requested_scale"] == args.scale


@pytest.mark.parametrize("field,value", [("scale", float("nan")), ("scale", float("inf")),
    ("scale", 3.5e38), ("mask", True), ("disable_optimization", ["shuffle", "shuffle"])])
def test_invalid_ablation_requests_reject_before_output(tmp_path, field, value):
    args = arguments(tmp_path)
    setattr(args, field, value)
    with pytest.raises(ValueError): matrix.run_matrix(args, preflight_fn=ready)
    assert not args.output.exists()


@pytest.mark.parametrize("field,value", [("methods", ["ragged-auto", "ragged-auto"]),
    ("small_bytes", 0), ("large_bytes", 1), ("mean_length", 0), ("seed", -1), ("timeout", float("nan"))])
def test_invalid_matrix_requests_reject_before_creating_output(tmp_path, field, value):
    args = arguments(tmp_path)
    setattr(args, field, value)
    with pytest.raises(ValueError): matrix.run_matrix(args, preflight_fn=ready)
    assert not args.output.exists()
