"""Choosing between device classification and one block per segment.

An adaptive program holds both compiled schedules. Before each call it
predicts the complete-call time of each from cheap statistics of the input:
the row count N, the segment count B, the longest segment L, the feature
width D and the element size, and runs the faster one. The prediction for a
schedule is linear in four terms,

    t = c0 + c1 * N*D*bytes + c2 * L*D*bytes + c3 * B      (microseconds)

a fixed cost (launches and host work), streaming the rows, the longest
segment's critical path, and per-segment work. The skew L/(N/B) enters
through the ratio of the second and third terms. Coefficients are fitted per
operation family (sum and max; softmax) and schedule by ``bench.calibrate``
on training seeds and stored in ``cost_model.json``; classification runs only
when its prediction is lower than block per segment's by the stored margin.

L comes from the caller (``run(max_len=...)``, the metadata a jagged tensor
such as NJT carries). Without it no statistic of the lengths is known on the
host, and the program keeps device classification, whose time does not
depend on the longest segment.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .errors import CacheMismatch

MODEL_PATH = Path(__file__).with_name("cost_model.json")
SCHEDULES = ("classified", "naive")
FAMILIES = ("reduce", "softmax")
TERMS = 4


def family(op: str) -> str:
    return "softmax" if op == "softmax" else "reduce"


def features(n: int, b: int, d: int, element_bytes: int, max_len: int) -> tuple[float, ...]:
    """The four terms of the prediction; bytes in MiB, segments in thousands."""
    return (1.0, n * d * element_bytes / 2**20, max_len * d * element_bytes / 2**20, b / 1000.0)


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
            raise CacheMismatch("cost model coefficients must be four nonnegative terms and margin in [0, 1)")
        return cls(coefficients, margin, dict(data.get("calibration", {})))

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "CostModel":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def predict(self, schedule: str, op: str, n: int, b: int, d: int, element_bytes: int,
                max_len: int) -> float:
        terms = features(n, b, d, element_bytes, max_len)
        return sum(c * x for c, x in zip(self.coefficients[family(op)][schedule], terms))

    def choose(self, op: str, n: int, b: int, d: int, element_bytes: int,
               max_len: int | None) -> tuple[str, dict[str, Any]]:
        """The schedule to run and the inputs and predictions behind it."""
        if max_len is None:
            return "classified", {"max_len": None, "reason": "no length statistics on the host"}
        classified = self.predict("classified", op, n, b, d, element_bytes, max_len)
        naive = self.predict("naive", op, n, b, d, element_bytes, max_len)
        selected = "classified" if classified < naive * (1 - self.margin) else "naive"
        return selected, {"max_len": max_len, "predicted_us": {"classified": round(classified, 2),
                                                               "naive": round(naive, 2)}}


_model: CostModel | None = None


def default_model() -> CostModel:
    global _model
    if _model is None:
        _model = CostModel.load()
    return _model
