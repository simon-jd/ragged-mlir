"""Specification self-tests. These do not prove compiler/GPU correctness."""

import ast
from pathlib import Path

import numpy as np
import pytest

from oracle import (
    assert_matches, quantize_bf16, segment_max, segment_reduce, segment_softmax,
    segment_sum, validate_offsets,
)


@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
def test_hand_calculated_reductions_and_softmax(index_dtype):
    values = np.array([[1, 2], [3, 4], [-2, 7]], dtype=np.float32)
    offsets = np.array([0, 2, 2, 3], dtype=index_dtype)
    np.testing.assert_array_equal(segment_sum(values, offsets), [[4, 6], [0, 0], [-2, 7]])
    np.testing.assert_array_equal(segment_max(values, offsets), [[3, 4], [-np.inf, -np.inf], [-2, 7]])
    low = 1 / (1 + np.exp(2))
    np.testing.assert_allclose(segment_softmax(values, offsets), [[low, low], [1-low, 1-low], [1, 1]])


@pytest.mark.parametrize("offsets", [[0], [0, 0], [0, 0, 0, 0]])
@pytest.mark.parametrize("d", [1, 3, 16, 128, 1024])
def test_empty_and_all_empty(offsets, d):
    values = np.empty((0, d), dtype=np.float32)
    offsets = np.array(offsets, dtype=np.int64)
    sums, maxima, softmax = segment_sum(values, offsets), segment_max(values, offsets), segment_softmax(values, offsets)
    assert sums.shape == maxima.shape == (len(offsets)-1, d)
    assert softmax.shape == (0, d)
    assert np.all(sums == 0) and not np.any(np.signbit(sums))
    assert np.all(np.isneginf(maxima))


def test_nonfinite_is_per_feature_and_nan_propagates():
    values = np.array([[np.nan, np.inf, -np.inf, -np.inf, 1],
                       [1, 1, -np.inf, 3, 2]], dtype=np.float32)
    offsets = np.array([0, 2], dtype=np.int32)
    np.testing.assert_array_equal(segment_sum(values, offsets), [[np.nan, np.inf, -np.inf, -np.inf, 3]])
    np.testing.assert_array_equal(segment_max(values, offsets), [[np.nan, np.inf, -np.inf, 3, 2]])
    actual = segment_softmax(values, offsets)
    assert np.all(np.isnan(actual[:, :3]))
    np.testing.assert_array_equal(actual[:, 3], [0, 1])
    np.testing.assert_allclose(actual[:, 4], [1/(1+np.e), np.e/(1+np.e)])


def test_opposite_infinities_sum_to_nan():
    values = np.array([[np.inf], [-np.inf]], dtype=np.float32)
    offsets = np.array([0, 2], dtype=np.int32)
    assert np.isnan(segment_sum(values, offsets)[0, 0])
    assert np.isposinf(segment_max(values, offsets)[0, 0])
    # A single +inf surrounded only by -inf must poison the entire feature,
    # including -inf positions. This catches an online-statistics empty-lane bug.
    assert np.all(np.isnan(segment_softmax(values, offsets)))


@pytest.mark.parametrize("raw,n,condition", [
    ([1, 3], 3, r"offsets\[0\]"),
    ([0, 3, 2, 3], 3, "nondecreasing"),
    ([0, 2], 3, r"offsets\[-1\]"),
    ([0, -1, 3], 3, r"\[0, N\]"),
    ([0, 4, 3], 3, r"\[0, N\]"),
    ([], 0, "length"),
])
@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_invalid_offsets(raw, n, condition, dtype):
    with pytest.raises(ValueError, match=condition):
        validate_offsets(np.array(raw, dtype=dtype), n)


@pytest.mark.parametrize("offsets", [np.array([0, 1.0]), np.array([0, 1], dtype=np.uint32), np.array([0, 1], dtype=np.int16)])
def test_offset_dtype(offsets):
    with pytest.raises(TypeError, match="i32 or i64"):
        validate_offsets(offsets, 1)


def test_offset_limits_without_giant_allocation():
    max32, max64 = np.iinfo(np.int32).max, np.iinfo(np.int64).max
    validate_offsets(np.array([0, max32-1, max32], dtype=np.int32), int(max32))
    validate_offsets(np.array([0, max32, max32+1], dtype=np.int64), int(max32)+1)
    validate_offsets(np.array([0, max64], dtype=np.int64), int(max64))
    with pytest.raises(ValueError, match="nondecreasing"):
        validate_offsets(np.array([0, max64, 1, max64], dtype=np.int64), int(max64))
    with pytest.raises(ValueError, match="i64 indexing"):
        validate_offsets(np.array([0], dtype=np.int64), int(max64)+1)


@pytest.mark.parametrize("values,condition", [
    (np.zeros(2, dtype=np.float32), "rank 2"),
    (np.zeros((1, 0), dtype=np.float32), "D must"),
    (np.zeros((1, 1025), dtype=np.float32), "D must"),
    (np.zeros((2, 3), dtype=np.float32).T, "contiguous"),
])
def test_invalid_values(values, condition):
    with pytest.raises(ValueError, match=condition):
        segment_sum(values, np.array([0, values.shape[0]], dtype=np.int32))


def test_reject_noncontiguous_offsets():
    with pytest.raises(ValueError, match="contiguous"):
        validate_offsets(np.array([0, 9, 2, 9], dtype=np.int32)[::2], 2)


def test_reject_invalid_kind():
    with pytest.raises(ValueError, match="sum or max"):
        segment_reduce(np.zeros((1, 1), dtype=np.float32), np.array([0, 1], dtype=np.int32), "mean")


def test_bf16_round_ties_even_and_nonfinite():
    # Both numbers are halfway between adjacent bf16 values near 1.
    values = np.array([1+2**-8, 1+3*2**-8, np.inf, -np.inf, np.nan, -0.0], dtype=np.float32)
    actual = quantize_bf16(values)
    np.testing.assert_array_equal(actual[:4], [1, 1+2**-6, np.inf, -np.inf])
    assert np.isnan(actual[4]) and np.signbit(actual[5])
    tiny_payload = np.array([0x7F800001], dtype=np.uint32).view(np.float32)
    assert np.isnan(quantize_bf16(tiny_payload)[0])


@pytest.mark.parametrize("d", [1, 3, 16, 127, 128, 1024])
@pytest.mark.parametrize("dtype", ["f32", "bf16"])
def test_finite_softmax_normalization_extremes(d, dtype):
    rng = np.random.default_rng(8723)
    values = rng.normal(size=(259, d)).astype(np.float32) * 1000
    if dtype == "bf16":
        values = quantize_bf16(values)
    offsets = np.array([0, 0, 1, 258, 259], dtype=np.int64)
    actual = segment_softmax(values, offsets)
    assert np.all(np.isfinite(actual)) and np.all(actual >= 0) and np.all(actual <= 1)
    for start, end in zip(offsets[:-1], offsets[1:]):
        if start != end:
            np.testing.assert_allclose(actual[start:end].sum(axis=0), 1, atol=1e-14)


@pytest.mark.parametrize("operation", [segment_sum, segment_max, segment_softmax])
def test_segment_permutation_and_empty_insertion(operation):
    rng = np.random.default_rng(9289)
    values = rng.normal(size=(12, 5)).astype(np.float32)
    offsets = np.array([0, 2, 7, 7, 12], dtype=np.int64)
    pieces = [values[a:b] for a, b in zip(offsets[:-1], offsets[1:])]
    permutation = [3, 0, 2, 1]
    permuted_values = np.concatenate([pieces[i] for i in permutation])
    permuted_offsets = np.array([0, *np.cumsum([len(pieces[i]) for i in permutation])], dtype=np.int64)
    original = operation(values, offsets)
    expected = np.concatenate([original[offsets[i]:offsets[i+1]] for i in permutation]) if operation == segment_softmax else original[permutation]
    np.testing.assert_array_equal(operation(permuted_values, permuted_offsets), expected)
    expanded = np.array([0, 0, 2, 7, 7, 12, 12], dtype=np.int64)
    actual = operation(values, expanded)
    np.testing.assert_array_equal(actual if operation == segment_softmax else actual[1:5], original)


def test_split_segment_sum_and_softmax_shift_invariance():
    rng = np.random.default_rng(773)
    values = rng.normal(size=(129, 7))
    original = np.array([0, 129], dtype=np.int64)
    split = np.array([0, 31, 129], dtype=np.int64)
    np.testing.assert_allclose(segment_sum(values, original)[0], segment_sum(values, split).sum(axis=0), atol=1e-13)
    np.testing.assert_allclose(segment_softmax(values, original), segment_softmax(values+13, original), rtol=1e-13, atol=1e-15)


def test_tolerance_accepts_rounding_but_rejects_wrong_results_and_nan():
    values = np.array([[1], [2]], dtype=np.float32)
    offsets = np.array([0, 2], dtype=np.int32)
    stats = assert_matches(np.array([[3.000001]]), values, offsets, operation="sum", input_dtype="f32")
    assert stats["finite_values"] == 1 and stats["max_abs_error"] > 0
    with pytest.raises(AssertionError, match="exceed tolerance"):
        assert_matches(np.array([[4]]), values, offsets, operation="sum", input_dtype="f32")
    with pytest.raises(AssertionError, match="NaN"):
        assert_matches(np.array([[np.nan]]), values, offsets, operation="sum", input_dtype="f32")
    with pytest.raises(AssertionError, match="shape"):
        assert_matches(np.array([3]), values, offsets, operation="sum", input_dtype="f32")


def test_tolerance_is_cancellation_safe():
    values = np.array([[1e8], [1], [-1e8]], dtype=np.float32)
    assert_matches(np.array([[0]]), values, np.array([0, 3], dtype=np.int32), operation="sum", input_dtype="f32")


def test_output_dtype_overflow_is_distinguished_from_bad_infinities():
    values = np.array([[3e38], [3e38]], dtype=np.float32)
    offsets = np.array([0, 2], dtype=np.int32)
    assert_matches(np.array([[np.inf]]), values, offsets, operation="sum", input_dtype="f32")
    with pytest.raises(AssertionError, match="positive infinity"):
        assert_matches(np.array([[np.inf]]), np.ones((2, 1), dtype=np.float32), offsets, operation="sum", input_dtype="f32")


def test_single_very_long_segment():
    values = np.tile(np.array([[1, -1, 0.5]], dtype=np.float32), (131073, 1))
    offsets = np.array([0, 131073], dtype=np.int64)
    np.testing.assert_array_equal(segment_sum(values, offsets), [[131073, -131073, 65536.5]])
    np.testing.assert_array_equal(segment_max(values, offsets), [[1, -1, 0.5]])
    np.testing.assert_allclose(segment_softmax(values, offsets), np.full(values.shape, 1/131073), rtol=1e-14)


@pytest.mark.parametrize("dtype", ["f32", "bf16"])
def test_softmax_absolute_tolerance_cannot_accept_all_zero_long_segment(dtype):
    values = np.zeros((1_000_000, 1), dtype=np.float32)
    with pytest.raises(AssertionError, match="sum to one"):
        assert_matches(np.zeros_like(values), values, np.array([0, len(values)], dtype=np.int64), operation="softmax", input_dtype=dtype)


def test_oracle_does_not_import_compiler_or_torch():
    for source in (Path(__file__).parents[2] / ".." / "oracle").resolve().glob("*.py"):
        for node in ast.walk(ast.parse(source.read_text())):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            assert all(name.split(".")[0] not in {"ragged", "torch", "triton"} for name in names)
