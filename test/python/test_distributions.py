import numpy as np
import pytest

from bench.distributions import DISTRIBUTIONS, describe, make_lengths, offsets_from_lengths


@pytest.mark.parametrize("distribution", DISTRIBUTIONS)
@pytest.mark.parametrize("total", [0, 1, 1025, 2**53 + 19, 2**63 - 1])
def test_fixed_total_preserved_without_float_rounding(distribution, total):
    lengths = make_lengths(distribution, 77, total_elements=total, seed=11)
    assert sum(map(int, lengths)) == total
    assert np.all(lengths >= 0)
    assert offsets_from_lengths(lengths)[-1] == total
    np.testing.assert_array_equal(lengths, make_lengths(distribution, 77, total_elements=total, seed=11))
    assert sum(describe(lengths)["histogram_counts"]) == 77


def test_empty_and_overflow_validation():
    assert offsets_from_lengths(make_lengths("burst", 0)).tolist() == [0]
    with pytest.raises(ValueError, match="zero segments"):
        make_lengths("uniform", 0, total_elements=1)
    with pytest.raises(ValueError, match="fit i32"):
        offsets_from_lengths(np.array([2**31], dtype=np.int64), "i32")
    with pytest.raises(ValueError, match="nonnegative"):
        offsets_from_lengths(np.array([-1], dtype=np.int64))
    with pytest.raises(ValueError, match="i64"):
        make_lengths("burst", 32, mean_length=2**63 - 1)


def test_long_tail_and_burst_are_not_uniform():
    for distribution in ("long_tail", "bimodal", "burst", "exponential"):
        lengths = make_lengths(distribution, 16384, seed=42, total_elements=1_000_000)
        assert lengths.max() > 8 * np.median(lengths)
