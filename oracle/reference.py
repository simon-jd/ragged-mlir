"""Numerical specification for packed values[N, D] plus offsets[B+1].

All arithmetic is NumPy float64, independent of mapping, lowering, and runtime.
Inputs must already reflect their source dtype: use ``quantize_bf16`` for bf16
represented by NumPy float32 arrays. Outputs stay float64 so comparisons include
the compiler's accumulation and output rounding error.
"""

from __future__ import annotations

import numpy as np


TOLERANCES = {"f32": (2e-5, 2e-6), "bf16": (2e-2, 2e-3)}


def validate_offsets(offsets: np.ndarray, n: int) -> np.ndarray:
    """Validate offsets without allocating an N-row tensor (including i32 limits)."""
    offsets = np.asarray(offsets)
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 0:
        raise ValueError("N must be a nonnegative integer")
    if n > np.iinfo(np.int64).max:
        raise ValueError("N exceeds the i64 indexing range")
    if offsets.ndim != 1 or offsets.size == 0:
        raise ValueError("offsets must have rank 1 and length B+1 >= 1")
    if offsets.dtype not in (np.dtype("int32"), np.dtype("int64")):
        raise TypeError("offsets dtype must be i32 or i64")
    if not offsets.flags.c_contiguous:
        raise ValueError("offsets must be contiguous")
    if int(offsets[0]) != 0:
        raise ValueError("offsets[0] must equal 0")
    if np.any(offsets < 0) or np.any(offsets > n):
        raise ValueError("offsets must lie in [0, N]")
    # Compare adjacent values rather than subtracting: subtraction can overflow.
    if np.any(offsets[1:] < offsets[:-1]):
        raise ValueError("offsets must be nondecreasing")
    if int(offsets[-1]) != n:
        raise ValueError("offsets[-1] must equal N")
    return offsets


def _inputs(values: np.ndarray, offsets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values)
    if values.ndim != 2:
        raise ValueError("values must have rank 2 [N, D]")
    if not 1 <= values.shape[1] <= 1024:
        raise ValueError("D must be in [1, 1024]")
    if values.dtype.kind != "f":
        raise TypeError("values must have a real floating dtype")
    if not values.flags.c_contiguous:
        raise ValueError("values must be row-major contiguous")
    return values.astype(np.float64), validate_offsets(offsets, values.shape[0])


def quantize_bf16(values: np.ndarray) -> np.ndarray:
    """Round float32 to bf16, ties to even; return exact float32 representations.

    Preserve NaNs explicitly so a tiny NaN payload cannot round to infinity.
    This helper is test-data preparation, not part of the numeric oracle.
    """
    values = np.asarray(values, dtype=np.float32)
    bits = values.view(np.uint32)
    nan = np.isnan(values)
    with np.errstate(over="ignore"):
        rounded = (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)
    rounded = np.where(nan, np.uint32(0x7FC00000), rounded).astype(np.uint32)
    return rounded.view(np.float32)


def segment_reduce(values: np.ndarray, offsets: np.ndarray, kind: str) -> np.ndarray:
    if kind not in ("sum", "max"):
        raise ValueError("reduction kind must be sum or max")
    values, offsets = _inputs(values, offsets)
    out = np.empty((len(offsets) - 1, values.shape[1]), dtype=np.float64)
    with np.errstate(invalid="ignore", over="ignore"):
        for segment, (begin, end) in enumerate(zip(offsets[:-1], offsets[1:])):
            part = values[int(begin):int(end)]
            if kind == "sum":
                out[segment] = np.sum(part, axis=0, dtype=np.float64)
            else:
                # np.max propagates NaNs; fmax/nanmax would silently erase them.
                out[segment] = np.max(part, axis=0, initial=-np.inf)
    return out


def segment_sum(values: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    return segment_reduce(values, offsets, "sum")


def segment_max(values: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    return segment_reduce(values, offsets, "max")


def segment_softmax(values: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Stable per-feature softmax; NaN/+inf/all -inf columns become all NaN."""
    values, offsets = _inputs(values, offsets)
    out = np.empty_like(values)
    with np.errstate(invalid="ignore", over="ignore", divide="ignore", under="ignore"):
        for begin, end in zip(offsets[:-1], offsets[1:]):
            begin, end = int(begin), int(end)
            if begin == end:
                continue
            part = values[begin:end]
            weights = np.exp(part - np.max(part, axis=0))
            out[begin:end] = weights / np.sum(weights, axis=0, dtype=np.float64)
    return out


def assert_matches(
    actual: np.ndarray,
    values: np.ndarray,
    offsets: np.ndarray,
    *,
    operation: str,
    input_dtype: str,
) -> dict[str, float | int]:
    """Check the common accuracy contract and return finite error statistics.

    For sum, a cancellation-safe absolute scale is sum(abs(input)); for max and
    softmax it is abs(reference). NaN masks and signed infinities must match
    exactly. No mapping or optimization may choose a looser tolerance.
    """
    functions = {"sum": segment_sum, "max": segment_max, "softmax": segment_softmax}
    if operation not in functions:
        raise ValueError("operation must be sum, max, or softmax")
    if input_dtype not in TOLERANCES:
        raise ValueError("input_dtype must be f32 or bf16")
    reference = functions[operation](values, offsets)
    # Outputs use the input dtype. An otherwise finite float64 answer may exceed
    # its representable range; that final conversion must produce signed inf.
    with np.errstate(over="ignore", invalid="ignore"):
        rounded = reference.astype(np.float32)
        if input_dtype == "bf16":
            rounded = quantize_bf16(rounded)
    expected = np.where(np.isinf(rounded), rounded.astype(np.float64), reference)
    actual = np.asarray(actual, dtype=np.float64)
    if actual.shape != expected.shape:
        raise AssertionError(f"shape mismatch: {actual.shape} != {expected.shape}")
    if not np.array_equal(np.isnan(actual), np.isnan(expected)):
        raise AssertionError("NaN locations differ from oracle")
    if not np.array_equal(np.isposinf(actual), np.isposinf(expected)):
        raise AssertionError("positive infinity locations differ from oracle")
    if not np.array_equal(np.isneginf(actual), np.isneginf(expected)):
        raise AssertionError("negative infinity locations differ from oracle")
    scale = segment_sum(np.abs(values), offsets) if operation == "sum" else np.abs(expected)
    rtol, atol = TOLERANCES[input_dtype]
    finite = np.isfinite(expected)
    error = np.abs(actual[finite] - expected[finite])
    bound = atol + rtol * scale[finite]
    failures = error > bound
    if np.any(failures):
        raise AssertionError(
            f"{operation}/{input_dtype}: {int(np.sum(failures))} finite values exceed "
            f"tolerance; max abs error={float(np.max(error)):.9g}"
        )
    if operation == "softmax":
        # Per-element atol alone accepts an all-zero answer for a sufficiently
        # long segment. Check the defining probability invariant as well.
        if np.any(actual[finite] < 0) or np.any(actual[finite] > 1):
            raise AssertionError("finite softmax probabilities must lie in [0, 1]")
        for begin, end in zip(offsets[:-1], offsets[1:]):
            begin, end = int(begin), int(end)
            if begin == end:
                continue
            valid_columns = np.all(np.isfinite(expected[begin:end]), axis=0)
            total = np.sum(actual[begin:end, valid_columns], axis=0, dtype=np.float64)
            if np.any(np.abs(total - 1) > rtol + atol):
                raise AssertionError("finite softmax columns must sum to one within dtype tolerance")
    return {
        "finite_values": int(np.sum(finite)),
        "nan_values": int(np.sum(np.isnan(expected))),
        "max_abs_error": float(np.max(error, initial=0)),
        "max_fraction_of_tolerance": float(np.max(error / bound, initial=0)),
    }
