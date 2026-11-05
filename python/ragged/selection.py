"""Choosing between device classification and one block per segment.

An adaptive program holds both compiled schedules. Before each call it
predicts the complete-call time of each from cheap statistics of the input:
the row count N, the segment count B, the longest segment L and the feature
width D (and the element size), and runs the faster one. The prediction for
a schedule is linear in six terms (microseconds),

    t = c0 + c1 * M + c2 * S + c3 * B + c4 * M * log2(skew) + c5 * X

with M = N*D elements (in units of 2^18, a MiB of f32), S = L * span / 128
the elements each thread of the one block that owns the longest segment
reads (span = d_tile * d_vector features per tile), skew = L / (N/B) and X
the MiB of the call's working set (the input rows plus the output: B rows
for sum and max, N rows for softmax) beyond the L2 capacity: a fixed cost (host work and launches), the per-element
folds, the critical path of block per segment, per-segment work, the
imbalance that skewed lengths add, and the bytes that back-to-back calls
cannot keep in the L2. Coefficients are fitted per operation family (sum
and max; softmax) and schedule by ``bench.calibrate`` and stored in
``cost_model.json``; classification runs only when its prediction is lower
than block per segment's by the stored margin. The model stores the L2
capacity of the GPU it was calibrated on (``l2_mib``); the runtime passes the
L2 of the device each call runs on instead, which is the same on that GPU.

Version 1 models (before 2026-11) used the row bytes, N*D*element size in
MiB, as the sixth term; for f32 that is M again. They still load, so that
``bench.calibrate evaluate`` can replay their choices.

L comes from the caller (``run(max_len=...)``, the metadata a jagged tensor
such as NJT carries). Without it no statistic of the lengths is known on the
host, and the program keeps device classification, whose time does not
depend on the longest segment.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

from .errors import CacheMismatch

MODEL_PATH = Path(__file__).with_name("cost_model.json")
SCHEDULES = ("classified", "naive")
FAMILIES = ("reduce", "softmax")
TERMS = 6
# The sixth term of each model version.
LAST_TERM = {1: "row_mib", 2: "l2_excess_mib"}


def family(op: str) -> str:
    return "softmax" if op == "softmax" else "reduce"


def features(n: int, b: int, d: int, max_len: int, span: int, element_bytes: int = 4, op: str = "sum",
             l2_mib: float = 72.0, version: int = 2) -> tuple[float, ...]:
    """The six terms of the prediction: 2^18 elements, thousands of thread
    elements and of segments, skew-weighted elements, and MiB (of the
    working set beyond an L2 of ``l2_mib``; of the rows for version 1)."""
    elements = n * d / 2**18
    skew = max_len * b / n if n else 1.0
    rows = n * d * element_bytes / 2**20
    if version == 1:
        last = rows
    else:
        output_rows = n if op == "softmax" else b
        last = max(0.0, rows + output_rows * d * element_bytes / 2**20 - l2_mib)
    return (1.0, elements, max_len * span / 128 / 1000.0, b / 1000.0,
            elements * math.log2(max(1.0, skew)), last)


@dataclass(frozen=True)
class CostModel:
    coefficients: dict[str, dict[str, tuple[float, ...]]]
    margin: float
    calibration: dict[str, Any]
    version: int = 2
    l2_mib: float = 72.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CostModel":
        try:
            version = data["version"]
            if version not in LAST_TERM:
                raise CacheMismatch("unsupported cost model version")
            coefficients = {
                name: {schedule: tuple(float(value) for value in data["coefficients"][name][schedule])
                       for schedule in SCHEDULES}
                for name in FAMILIES}
            margin = float(data["margin"])
            l2_mib = float(data["l2_mib"]) if version >= 2 else 0.0
        except (KeyError, TypeError, ValueError) as exc:
            raise CacheMismatch(f"invalid cost model: {exc}") from exc
        if (any(len(values) != TERMS or any(value < 0 for value in values)
                for schedules in coefficients.values() for values in schedules.values())
                or not 0 <= margin < 1 or (version >= 2 and not l2_mib > 0)):
            raise CacheMismatch(f"cost model coefficients must be {TERMS} nonnegative terms, margin in [0, 1) "
                                "and the L2 capacity positive")
        return cls(coefficients, margin, dict(data.get("calibration", {})), version, l2_mib)

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "CostModel":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def predict(self, schedule: str, op: str, n: int, b: int, d: int, max_len: int, span: int,
                element_bytes: int = 4, l2_mib: float | None = None) -> float:
        """Predicted microseconds; ``l2_mib`` replaces the calibration GPU's L2 capacity."""
        terms = features(n, b, d, max_len, span, element_bytes, op,
                         self.l2_mib if l2_mib is None else l2_mib, self.version)
        return sum(c * x for c, x in zip(self.coefficients[family(op)][schedule], terms))

    def choose(self, op: str, n: int, b: int, d: int, max_len: int | None,
               span: int, element_bytes: int = 4, l2_mib: float | None = None) -> tuple[str, dict[str, Any]]:
        """The schedule to run and the inputs and predictions behind it;
        ``l2_mib`` is the L2 capacity of the device that runs the call
        (default: the GPU the model was calibrated on)."""
        if max_len is None:
            return "classified", {"max_len": None, "reason": "no length statistics on the host"}
        if l2_mib is not None and not (isinstance(l2_mib, (int, float)) and l2_mib > 0):
            raise ValueError("l2_mib must be a positive capacity in MiB")
        classified = self.predict("classified", op, n, b, d, max_len, span, element_bytes, l2_mib)
        naive = self.predict("naive", op, n, b, d, max_len, span, element_bytes, l2_mib)
        selected = "classified" if classified < naive * (1 - self.margin) else "naive"
        return selected, {"max_len": max_len, "l2_mib": self.l2_mib if l2_mib is None else l2_mib,
                          "predicted_us": {"classified": round(classified, 2), "naive": round(naive, 2)}}


_model: CostModel | None = None


def default_model() -> CostModel:
    global _model
    if _model is None:
        _model = CostModel.load()
    return _model
