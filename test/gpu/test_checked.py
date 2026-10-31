"""Independent checked-offset device tests; missing sm_89 is never a pass.

The expected diagnostics use Python integers, so adversarial i64 endpoints
cannot overflow the reference. Invalid input must fail before touching output.
"""

from __future__ import annotations

import numpy as np
import pytest

from oracle import assert_matches
from ragged import Hints, ProgramSpec, ValidationError
from test_correctness import gpu_environment, prepared


CONDITIONS = (
    "first_nonzero", "final_mismatch", "negative", "above_n",
    "nonmonotone", "below_min_len", "above_max_len",
)


def expected_errors(offsets, n, hints):
    """Independent checked contract, including safe segment-length guards."""
    offsets = list(map(int, offsets))
    errors = {}
    if offsets[0] != 0:
        errors["first_nonzero"] = 0
    if offsets[-1] != n:
        errors["final_mismatch"] = len(offsets) - 1
    for name, predicate in (("negative", lambda x: x < 0),
                            ("above_n", lambda x: x > n)):
        indices = [i for i, value in enumerate(offsets) if predicate(value)]
        if indices:
            errors[name] = min(indices)
    for i, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        if end < start:
            errors.setdefault("nonmonotone", i)
        if 0 <= start <= end <= n:
            length = end - start
            if hints.min_len is not None and length < hints.min_len:
                errors.setdefault("below_min_len", i)
            if hints.max_len is not None and length > hints.max_len:
                errors.setdefault("above_max_len", i)
    return {name: errors[name] for name in CONDITIONS if name in errors}


def checked_spec(op, offset_dtype, mapping, *, dtype="f32", hints=None,
                 inplace=False, debug=False):
    if hints is None:
        hints = Hints(min_len=4097, max_len=8193) if mapping == "split" else Hints()
    return ProgramSpec(op, dtype, 3, offset_dtype, hints=hints,
                       strategy="naive" if mapping == "naive" else "auto",
                       alias="inplace" if inplace else "none", debug_visits=debug)


def assert_rejected(gpu_environment, spec, offsets, n, *, debug=False):
    torch, compile_spec, _, _ = gpu_environment
    expected = expected_errors(offsets, n, spec.hints)
    assert expected, "negative test must have an independent expected violation"
    dtype = torch.float32 if spec.dtype == "f32" else torch.bfloat16
    offset_type = torch.int32 if spec.offsets_dtype == "i32" else torch.int64
    values = torch.arange(n * spec.feature_dim, device="cuda", dtype=torch.float32)
    values = (values.reshape(n, spec.feature_dim) / 8).to(dtype=dtype)
    original = values.clone()
    device_offsets = torch.tensor(offsets, device="cuda", dtype=offset_type)
    rows = n if spec.op == "softmax" else len(offsets) - 1
    guard = torch.full((rows + 2, spec.feature_dim), -64, device="cuda", dtype=dtype)
    out = values if spec.alias == "inplace" else guard[1:-1]
    debug_buffers = {} if debug else None
    program = compile_spec(spec)
    with pytest.raises(ValidationError) as caught:
        program.run(values, device_offsets, out, mode="checked", debug_buffers=debug_buffers)
    error = caught.value
    assert error.errors == expected
    assert (error.condition, error.index) == next(iter(expected.items()))
    for condition, index in expected.items():
        assert f"{condition} at index {index}" in str(error)
    torch.cuda.current_stream().synchronize()
    assert torch.equal(values, original), "validation changed the input/in-place output"
    assert torch.all(guard == -64).item(), "main output or guard was written before rejection"
    if debug:
        assert debug_buffers == {}, "rejected call exposed main-program workspaces"


@pytest.mark.parametrize("op", ["sum", "max", "softmax"])
@pytest.mark.parametrize("offset_dtype", ["i32", "i64"])
@pytest.mark.parametrize("mapping", ["naive", "classified", "split"])
@pytest.mark.parametrize("offsets", [
    [1, 4, 8], [0, 4, 7], [0, -1, 8], [0, 9, 8], [0, 4, 3, 8],
], ids=["first", "final", "negative", "above-n", "nonmonotone"])
def test_checked_rejects_invalid_offsets_before_output(gpu_environment, op, offset_dtype,
                                                       mapping, offsets):
    assert_rejected(gpu_environment, checked_spec(op, offset_dtype, mapping), offsets, 8)


@pytest.mark.parametrize("op", ["sum", "max", "softmax"])
@pytest.mark.parametrize("offset_dtype", ["i32", "i64"])
@pytest.mark.parametrize("mapping", ["naive", "classified", "split"])
@pytest.mark.parametrize("claimed_checked", [False, True])
def test_checked_validates_both_length_hints_even_if_claimed_checked(
        gpu_environment, op, offset_dtype, mapping, claimed_checked):
    lower = 4097 if mapping == "split" else 2
    hints = Hints(min_len=lower, max_len=8193, offsets_checked=claimed_checked)
    offsets = [0, lower - 1, lower - 1 + 8194]
    spec = checked_spec(op, offset_dtype, mapping, hints=hints)
    assert expected_errors(offsets, offsets[-1], hints) == {
        "below_min_len": 0, "above_max_len": 1,
    }
    assert_rejected(gpu_environment, spec, offsets, offsets[-1])


@pytest.mark.parametrize("op", ["sum", "max", "softmax"])
@pytest.mark.parametrize("offset_dtype", ["i32", "i64"])
def test_checked_reports_all_conditions_and_earliest_indices(gpu_environment, op, offset_dtype):
    offsets = [1, -2, -3, 11, 10, 0, 1, 7, 6]
    hints = Hints(min_len=4, max_len=4, offsets_checked=True)
    assert expected_errors(offsets, 8, hints) == dict(zip(CONDITIONS, [0, 8, 1, 3, 0, 5, 6]))
    assert_rejected(gpu_environment, checked_spec(op, offset_dtype, "naive", hints=hints), offsets, 8)


@pytest.mark.parametrize("op", ["sum", "max", "softmax"])
@pytest.mark.parametrize("offset_dtype", ["i32", "i64"])
@pytest.mark.parametrize("mapping", ["naive", "classified", "split"])
def test_checked_extreme_endpoints_do_not_overflow_segment_lengths(
        gpu_environment, op, offset_dtype, mapping):
    bits = 32 if offset_dtype == "i32" else 64
    minimum, maximum = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    offsets = [0, minimum, maximum, 0, 8]
    # The invalid pair MIN -> MAX must not subtract and wrap before checking
    # bounds. Only the final 0 -> 8 segment is eligible for hint validation.
    hints = Hints(min_len=4097 if mapping == "split" else 0,
                  max_len=8 if mapping == "naive" else 8193, offsets_checked=True)
    assert_rejected(gpu_environment, checked_spec(op, offset_dtype, mapping, hints=hints), offsets, 8)


@pytest.mark.parametrize("op", ["sum", "max", "softmax"])
@pytest.mark.parametrize("offset_dtype", ["i32", "i64"])
@pytest.mark.parametrize("n,offsets", [(0, [1]), (0, [-1]), (2, [0]), (2, [2])])
def test_checked_zero_batch_still_checks_its_only_offset(gpu_environment, op, offset_dtype, n, offsets):
    assert_rejected(gpu_environment, checked_spec(op, offset_dtype, "classified"), offsets, n)


@pytest.mark.parametrize("offset_dtype", ["i32", "i64"])
@pytest.mark.parametrize("mapping", ["naive", "classified", "split"])
def test_checked_invalid_inplace_softmax_preserves_all_values(gpu_environment, offset_dtype, mapping):
    spec = checked_spec("softmax", offset_dtype, mapping, inplace=True)
    assert_rejected(gpu_environment, spec, [0, 4, 3, 8], 8)


def test_checked_failure_precedes_classification_workspace(gpu_environment):
    spec = checked_spec("softmax", "i64", "classified", debug=True)
    assert_rejected(gpu_environment, spec, [0, 4, 3, 8], 8, debug=True)


@pytest.mark.parametrize("op", ["sum", "max", "softmax"])
@pytest.mark.parametrize("offset_dtype", ["i32", "i64"])
def test_checked_atomic_min_retains_earliest_error_across_blocks(gpu_environment, op, offset_dtype):
    offsets = np.arange(4098, dtype=np.int64) * 2
    n = int(offsets[-1])
    offsets[130] = offsets[129] + 1  # Below-min at 129, above-max at 130.
    offsets[1026] = offsets[1025] + 1
    offsets[2051] = -1
    offsets[3077] = -2
    offsets[3001] = n + 1
    hints = Hints(min_len=2, max_len=2)
    spec = checked_spec(op, offset_dtype, "naive", hints=hints)
    assert expected_errors(offsets, n, hints) == {
        "negative": 2051, "above_n": 3001, "nonmonotone": 2050,
        "below_min_len": 129, "above_max_len": 130,
    }
    # Repeated independent launches also check init/check ordering and reset.
    for _ in range(3):
        assert_rejected(gpu_environment, spec, offsets, n)


def run_success(gpu_environment, spec, offsets):
    torch, compile_spec, _, _ = gpu_environment
    n = int(offsets[-1])
    values = prepared(np.random.default_rng(875).normal(size=(n, spec.feature_dim)), spec.dtype)
    dtype = torch.float32 if spec.dtype == "f32" else torch.bfloat16
    device_values = torch.from_numpy(values).to(device="cuda", dtype=dtype)
    offset_type = torch.int32 if spec.offsets_dtype == "i32" else torch.int64
    device_offsets = torch.tensor(offsets, device="cuda", dtype=offset_type)
    rows = n if spec.op == "softmax" else len(offsets) - 1
    guard = torch.full((rows + 2, spec.feature_dim), -64, device="cuda", dtype=dtype)
    output = device_values if spec.alias == "inplace" else guard[1:-1]
    report = compile_spec(spec).run(device_values, device_offsets, output, mode="checked")
    torch.cuda.current_stream().synchronize()
    assert report.mode == "checked"
    assert report.validation_workspace_bytes == 56
    assert report.validation_host_ms is not None and report.validation_host_ms >= 0
    validation = compile_spec(spec).artifact.metadata["validation"]
    assert report.kernel_names[:2] == tuple(kernel["name"] for kernel in validation["kernels"])
    if len(offsets) == 1:
        assert len(report.kernel_names) == 2 and report.classes == ()
        assert report.workspace_bytes == 56
    assert torch.all(guard[0] == -64).item() and torch.all(guard[-1] == -64).item()
    assert_matches(output.float().cpu().numpy(), values, np.asarray(offsets, dtype=np.int64),
                   operation=spec.op, input_dtype=spec.dtype)


@pytest.mark.parametrize("op", ["sum", "max", "softmax"])
@pytest.mark.parametrize("dtype", ["f32", "bf16"])
@pytest.mark.parametrize("offset_dtype", ["i32", "i64"])
@pytest.mark.parametrize("mapping", ["naive", "classified", "split"])
def test_checked_success_uses_the_same_numerical_oracle(gpu_environment, op, dtype, offset_dtype, mapping):
    offsets = [0, 4097, 12290] if mapping == "split" else [0, 0, 1, 34, 4131]
    run_success(gpu_environment, checked_spec(op, offset_dtype, mapping, dtype=dtype), offsets)


@pytest.mark.parametrize("op", ["sum", "max", "softmax"])
@pytest.mark.parametrize("offset_dtype", ["i32", "i64"])
@pytest.mark.parametrize("offsets", [[0], [0, 0], [0, 0, 0, 0]])
def test_checked_zero_elements_and_empty_segments_are_valid(gpu_environment, op, offset_dtype, offsets):
    run_success(gpu_environment, checked_spec(op, offset_dtype, "classified"), offsets)


@pytest.mark.parametrize("mapping", ["naive", "classified", "split"])
def test_checked_valid_inplace_softmax(gpu_environment, mapping):
    offsets = [0, 4097, 12290] if mapping == "split" else [0, 0, 1, 34, 4131]
    run_success(gpu_environment, checked_spec("softmax", "i64", mapping, inplace=True), offsets)


@pytest.mark.parametrize("op", ["sum", "max", "softmax"])
@pytest.mark.parametrize("offset_dtype", ["i32", "i64"])
def test_checked_status_is_reset_after_a_rejected_call(gpu_environment, op, offset_dtype):
    spec = checked_spec(op, offset_dtype, "classified")
    assert_rejected(gpu_environment, spec, [0, -1, 8], 8)
    run_success(gpu_environment, spec, [0, 4, 8])


@pytest.mark.parametrize("op", ["sum", "max", "softmax"])
@pytest.mark.parametrize("target_valid", [False, True])
def test_checked_validation_observes_explicit_stream_prior_writes(gpu_environment, op, target_valid):
    torch, compile_spec, _, _ = gpu_environment
    program = compile_spec(checked_spec(op, "i64", "classified"))
    current = torch.cuda.Stream()
    consumer = torch.cuda.Stream()
    with torch.cuda.stream(current):
        values = torch.arange(24, device="cuda", dtype=torch.float32).reshape(8, 3) / 8
        offsets = torch.tensor([0, 4, 8], dtype=torch.int64, device="cuda")
        output = torch.full((8 if op == "softmax" else 2, 3), -64.0, device="cuda")
        # Warm module loading and the small status allocator before constructing
        # the ordering pressure; they must not consume the queued-work window.
        program.run(values, offsets, output, stream=current, mode="checked")
        offsets[-1].fill_(7 if target_valid else 8)
        output.fill_(-64)
        consumer.wait_stream(current)
        with torch.cuda.stream(consumer):
            # Public CUDA computation provides queued work before the mutation.
            # The runtime must validate after it on consumer even though current
            # is a different stream when run() is called below.
            lhs = torch.ones((4096, 4096), device="cuda")
            rhs = torch.ones_like(lhs)
            delay = torch.empty_like(lhs)
            for _ in range(16):
                torch.mm(lhs, rhs, out=delay)
            offsets[-1].fill_(8 if target_valid else 7)
        assert torch.cuda.current_stream() == current
        if target_valid:
            report = program.run(values, offsets, output, stream=consumer, mode="checked")
            assert report.mode == "checked"
        else:
            with pytest.raises(ValidationError) as caught:
                program.run(values, offsets, output, stream=consumer, mode="checked")
            assert caught.value.errors == {"final_mismatch": 2}
        assert torch.cuda.current_stream() == current
        consumer.synchronize()
        if target_valid:
            assert_matches(output.cpu().numpy(), values.cpu().numpy(), np.array([0, 4, 8]),
                           operation=op, input_dtype="f32")
        else:
            assert torch.all(output == -64).item()
