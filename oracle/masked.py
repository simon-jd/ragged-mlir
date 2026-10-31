"""Additive-mask input semantics independent of compiler and runtime code.

Inputs already reflect source dtype (bf16 inputs are exact f32 representations).
Only the specified multiply and mask addition round to f32; softmax reference
arithmetic and all existing acceptance bounds remain in reference.py.
"""

import numpy as np

from .reference import assert_matches


def masked_logits(values, mask, *, scale=1.0):
    values, mask = np.asarray(values), np.asarray(mask)
    if values.ndim != 2 or mask.shape != values.shape:
        raise ValueError("dense mask must have exactly values[N,D] shape")
    if mask.dtype != np.dtype("float32") or not mask.flags.c_contiguous:
        raise ValueError("mask must be contiguous float32")
    if not values.flags.c_contiguous or values.dtype.kind != "f":
        raise ValueError("values must be contiguous floating point")
    if isinstance(scale, (bool, np.bool_)) or not np.isfinite(scale) or abs(scale) > np.finfo(np.float32).max:
        raise ValueError("scale must be finite f32")
    with np.errstate(over="ignore", invalid="ignore"):
        scaled = np.multiply(values.astype(np.float32), np.float32(scale), dtype=np.float32)
        return np.add(scaled, mask, dtype=np.float32)


def assert_masked_softmax_matches(actual, values, offsets, mask, *, scale=1.0, input_dtype="f32"):
    return assert_matches(actual, masked_logits(values, mask, scale=scale), offsets,
                          operation="softmax", input_dtype=input_dtype)
