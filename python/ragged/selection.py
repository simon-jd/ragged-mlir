"""Choosing between device classification and one block per segment.

An adaptive program holds both compiled schedules. Before each call it
predicts the complete-call time of each from cheap statistics of the input:
the row count N, the segment count B, the longest segment L and the feature
width D (and the element size), and runs the faster one. The prediction for
a schedule is linear in six terms (microseconds),

    t = c0 + c1 * M + c2 * S + c3 * B + c4 * M * log2(skew) + c5 * MiB

with M = N*D elements (in units of 2^18, a MiB of f32), S = L * span / 128
the elements each thread of the one block that owns the longest segment
reads (span = d_tile * d_vector features per tile), skew = L / (N/B) and MiB
the bytes of the rows: a fixed cost (launches and host work), the per-element
folds, the critical path of block per segment, per-segment work, the
imbalance that skewed lengths add, and moving the bytes (which for f32 is a
second copy of M, so the fit decides how bf16 differs). Coefficients are fitted per operation
family (sum and max; softmax) and schedule by ``bench.calibrate`` on
training seeds and stored in ``cost_model.json``; classification runs only
when its prediction is lower than block per segment's by the stored margin.

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


def family(op: str) -> str:
    return "softmax" if op == "softmax" else "reduce"


def features(n: int, b: int, d: int, max_len: int, span: int, element_bytes: int = 4) -> tuple[float, ...]:
    """The six terms of the prediction: 2^18 elements, thousands of thread
    elements and of segments, skew-weighted elements and MiB."""
    elements = n * d / 2**18
    skew = max_len * b / n if n else 1.0
    return (1.0, elements, max_len * span / 128 / 1000.0, b / 1000.0,
            elements * math.log2(max(1.0, skew)), n * d * element_bytes / 2**20)


@dataclass(frozen=True)
class CostModel:
    coefficients: dict[str, dict[str, tuple[float, ...]]]
    margin: float
    calibration: dict[str, Any]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CostModel":
        try:
            if data["version"] != 1:
                raise CacheMismatch("unsupported cost model version")
            coefficients = {
                name: {schedule: tuple(float(value) for value in data["coefficients"][name][schedule])
                       for schedule in SCHEDULES}
                for name in FAMILIES}
            margin = float(data["margin"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CacheMismatch(f"invalid cost model: {exc}") from exc
        if (any(len(values) != TERMS or any(value < 0 for value in values)
                for schedules in coefficients.values() for values in schedules.values())
                or not 0 <= margin < 1):
            raise CacheMismatch(f"cost model coefficients must be {TERMS} nonnegative terms and margin in [0, 1)")
        return cls(coefficients, margin, dict(data.get("calibration", {})))

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "CostModel":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def predict(self, schedule: str, op: str, n: int, b: int, d: int, max_len: int, span: int,
                element_bytes: int = 4) -> float:
        terms = features(n, b, d, max_len, span, element_bytes)
        return sum(c * x for c, x in zip(self.coefficients[family(op)][schedule], terms))

    def choose(self, op: str, n: int, b: int, d: int, max_len: int | None,
               span: int, element_bytes: int = 4) -> tuple[str, dict[str, Any]]:
        """The schedule to run and the inputs and predictions behind it."""
        if max_len is None:
            return "classified", {"max_len": None, "reason": "no length statistics on the host"}
        classified = self.predict("classified", op, n, b, d, max_len, span, element_bytes)
        naive = self.predict("naive", op, n, b, d, max_len, span, element_bytes)
        selected = "classified" if classified < naive * (1 - self.margin) else "naive"
        return selected, {"max_len": max_len, "predicted_us": {"classified": round(classified, 2),
                                                               "naive": round(naive, 2)}}


_model: CostModel | None = None


def default_model() -> CostModel:
    global _model
    if _model is None:
        _model = CostModel.load()
    return _model
