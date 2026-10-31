import numpy as np
import pytest

from oracle.minimize import minimize_case


def test_reducer_keeps_trigger_removes_other_segments_rows_features():
    values = np.zeros((20, 7), dtype=np.float32)
    values[12, 3] = 42
    offsets = np.array([0, 5, 5, 13, 20], dtype=np.int32)
    reduced, reduced_offsets = minimize_case(values, offsets, lambda v, _: bool(np.any(v == 42)))
    np.testing.assert_array_equal(reduced, [[42]])
    np.testing.assert_array_equal(reduced_offsets, [0, 1])
    assert reduced.dtype == np.float32 and reduced_offsets.dtype == np.int32


def test_reducer_preserves_multirow_trigger_and_nonfinite():
    values = np.array([[3], [np.nan], [4], [np.inf], [9]], dtype=np.float32)
    offsets = np.array([0, 5], dtype=np.int64)
    reduced, reduced_offsets = minimize_case(values, offsets, lambda v, _: bool(np.isnan(v).any() and np.isposinf(v).any()))
    np.testing.assert_array_equal(reduced, [[np.nan], [np.inf]])
    np.testing.assert_array_equal(reduced_offsets, [0, 2])


def test_reducer_refuses_nonfailure():
    with pytest.raises(ValueError, match="does not reproduce"):
        minimize_case(np.ones((1, 1)), np.array([0, 1], dtype=np.int64), lambda *_: False)
