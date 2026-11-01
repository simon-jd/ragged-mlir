"""The README chart's speedups come from summary.json medians."""
import pytest

from bench.chart import draw, speedups


def _case(op, d, distribution, methods, group="main", dtype="f32"):
    return {"group": group, "op": op, "dtype": dtype, "D": d, "distribution": distribution,
            "methods": {name: {mode: {"median_us": value} for mode, value in modes.items()}
                        for name, modes in methods.items()}}


SUMMARY = {"cases": [
    _case("max", 1, "uniform", {"ragged-auto": {"warm": 50.0, "cold": 60.0}, "torch-segment": {"warm": 15.0, "cold": 30.0}}),
    _case("sum", 128, "burst", {"ragged-auto": {"warm": 100.0}, "ragged-naive": {"warm": 10.0},
                                 "triton": {"warm": 200.0}, "cub": {}}),
    _case("sum", 1, "long_tail", {"ragged-auto": {"warm": 40.0, "cold": 50.0}, "cub": {"warm": 80.0, "cold": 75.0},
                                   "triton": {"warm": 120.0, "cold": 100.0}}),
    _case("sum", 1, "long_tail", {"ragged-auto": {"warm": 1.0}}, group="ablation/serial-classes"),
]}


def test_speedup_is_the_fastest_comparison_over_ragged_in_table_order():
    rows = speedups(SUMMARY)
    assert [label for label, _ in rows] == ["sum D=1 long tail", "sum D=128 bursty", "max D=1 uniform"]
    assert rows[0][1] == {"warm": (2.0, "cub"), "cold": (1.5, "cub")}
    # ragged-naive is this compiler, not a comparison method.
    assert rows[1][1] == {"warm": (2.0, "triton")}
    assert rows[2][1]["warm"] == (0.3, "torch-segment") and rows[2][1]["cold"] == (0.5, "torch-segment")


def test_chart_renders(tmp_path):
    pytest.importorskip("matplotlib")
    draw(SUMMARY, tmp_path / "speedup.svg")
    assert (tmp_path / "speedup.svg").read_text().startswith("<?xml")
