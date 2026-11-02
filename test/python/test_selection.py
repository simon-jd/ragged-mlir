"""Per-call schedule choice: the cost model and the runtime's dispatch to it."""
from __future__ import annotations

from dataclasses import replace
import json

import pytest

from ragged import CacheMismatch, ValidationError
from ragged.selection import MODEL_PATH, CostModel, features
import ragged.runtime as runtime
from test_runtime_classified import host_program


def model(classified, naive, margin=0.0):
    return CostModel.from_dict({"version": 1, "margin": margin,
                                "coefficients": {name: {"classified": list(classified) + [0] * (6 - len(classified)), "naive": list(naive) + [0] * (6 - len(naive))}
                                                 for name in ("reduce", "softmax")}})


def test_shipped_model_is_valid_and_records_its_calibration():
    shipped = CostModel.load(MODEL_PATH)
    assert set(shipped.coefficients) == {"reduce", "softmax"}
    assert json.loads(MODEL_PATH.read_text())["calibration"]


def test_features_count_elements_thread_elements_segments_and_skew():
    # 2^20 rows of D=4 in 1024 segments (mean 1024 rows), longest 2^14 rows
    # (skew 16), 32 features per tile.
    assert features(2**20, 1024, 4, 2**14, 32) == (1.0, 16.0, 4.096, 1.024, 64.0, 16.0)
    assert features(2**20, 1024, 4, 2**14, 32, 2)[5] == 8.0
    assert features(64, 1, 1, 64, 1)[4] == 0   # no skew


@pytest.mark.parametrize("bad", [{"version": 2}, {"version": 1, "margin": 0.1, "coefficients": {}},
                                 {"version": 1, "margin": 1.0, "coefficients": {
                                     name: {"classified": [0] * 6, "naive": [0] * 6} for name in ("reduce", "softmax")}},
                                 {"version": 1, "margin": 0.0, "coefficients": {
                                     name: {"classified": [-1, 0, 0, 0, 0, 0], "naive": [0] * 6} for name in ("reduce", "softmax")}}])
def test_invalid_models_are_rejected(bad):
    with pytest.raises(CacheMismatch):
        CostModel.from_dict(bad)


def test_long_segments_choose_classification_and_short_ones_block_per_segment():
    # Block per segment is cheaper to launch but pays for the longest segment.
    chosen = model(classified=[20, 1, 0, 0], naive=[10, 1, 4, 0])
    assert chosen.choose("sum", 2**20, 16384, 1, 64, 1)[0] == "naive"
    selected, info = chosen.choose("sum", 2**20, 16384, 1, 2**20, 1)
    assert selected == "classified" and info["predicted_us"]["naive"] > info["predicted_us"]["classified"]


def test_margin_keeps_block_per_segment_when_classification_is_barely_faster():
    assert model([10, 0, 0, 0], [10.5, 0, 0, 0]).choose("max", 1, 1, 1, 1, 1)[0] == "classified"
    assert model([10, 0, 0, 0], [10.5, 0, 0, 0], margin=0.1).choose("max", 1, 1, 1, 1, 1)[0] == "naive"


def test_unknown_longest_segment_keeps_classification():
    assert model([100, 0, 0, 0], [1, 0, 0, 0]).choose("softmax", 1, 1, 1, None, 1)[0] == "classified"


class Companion:
    def __init__(self):
        self.calls = []

    def _run(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return "naive-report"


@pytest.fixture
def adaptive(monkeypatch):
    program = host_program(debug=False)
    program.spec = replace(program.spec, strategy="auto")
    program.naive = Companion()
    calls = []
    monkeypatch.setattr(program, "_run", lambda *args, **kwargs: calls.append(kwargs) or "classified-report")
    return program, calls


class Shape:
    def __init__(self, *shape):
        self.shape = shape

    def numel(self):
        return self.shape[0]


def test_run_dispatches_on_the_model_choice(adaptive, monkeypatch):
    program, calls = adaptive
    values, offsets = Shape(100, 33), Shape(5)
    monkeypatch.setattr(runtime, "default_model", lambda: model([10, 0, 0, 0], [5, 0, 0, 0]))
    assert program.run(values, offsets, None, max_len=40) == "naive-report"
    (args, kwargs), = program.naive.calls
    assert kwargs["selected"] == "naive" and kwargs["selection"]["max_len"] == 40
    monkeypatch.setattr(runtime, "default_model", lambda: model([5, 0, 0, 0], [10, 0, 0, 0]))
    assert program.run(values, offsets, None, max_len=40) == "naive-report"   # chosen once per shape
    program._choices.clear()
    assert program.run(values, offsets, None, max_len=40) == "classified-report"
    assert calls[-1]["selected"] == "classified"


def test_forced_schedules_and_unknown_lengths(adaptive, monkeypatch):
    program, calls = adaptive
    monkeypatch.setattr(runtime, "default_model", lambda: model([10, 0, 0, 0], [5, 0, 0, 0]))
    values, offsets = Shape(100, 33), Shape(5)
    assert program.run(values, offsets, None, select="classified", max_len=3) == "classified-report"
    assert calls[-1]["selection"] == {"forced": True}
    assert program.run(values, offsets, None, select="naive") == "naive-report"
    assert program.run(values, offsets, None) == "classified-report"
    assert program.run(values, offsets, None, max_len=3) == "naive-report"


@pytest.mark.parametrize("options", [{"select": "fastest"}, {"max_len": -1}, {"max_len": 2.5}])
def test_selection_arguments_are_validated(adaptive, options):
    program, _ = adaptive
    with pytest.raises(ValidationError):
        program.run(Shape(1, 33), Shape(2), None, **options)


def test_naive_selection_requires_a_companion():
    program = host_program(debug=False)
    with pytest.raises(ValidationError, match="companion"):
        program.run(None, None, None, select="naive")
