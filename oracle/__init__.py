"""Independent NumPy float64 specification; never imports the compiler/runtime."""

from .reference import (
    TOLERANCES,
    assert_matches,
    quantize_bf16,
    segment_max,
    segment_reduce,
    segment_softmax,
    segment_sum,
    validate_offsets,
)

__all__ = [
    "TOLERANCES", "assert_matches", "quantize_bf16", "segment_max",
    "segment_reduce", "segment_softmax", "segment_sum", "validate_offsets",
]
