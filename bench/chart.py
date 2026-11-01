"""Draw the README chart: ragged's speedup over the fastest comparison, per case.

Speedup is the fastest comparison method's median time divided by ragged's
(ragged-auto), for warm and cold L2; below 1 means a comparison method is
faster. Needs matplotlib.

    python -m bench.chart results/bench/summary.json --output docs/speedup.svg
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LABELS = {"torch-segment": "torch", "triton": "Triton", "cub": "CUB", "cuda-naive": "CUDA",
          "njt-eager": "NJT", "njt-compile": "NJT compiled"}
DISTRIBUTIONS = {"uniform": "uniform", "long_tail": "long tail", "burst": "bursty"}


def speedups(summary):
    """(label, {mode: (speedup, fastest method)}) for every main case, in table order."""
    rows = []
    order = {op: index for index, op in enumerate(("sum", "max", "softmax"))}
    cases = sorted((case for case in summary["cases"] if case["group"] == "main"),
                   key=lambda case: (order[case["op"]], case["dtype"] != "f32", case["D"],
                                     list(DISTRIBUTIONS).index(case["distribution"])))
    for case in cases:
        label = f"{case['op']}{' bf16' if case['dtype'] == 'bf16' else ''} D={case['D']} {DISTRIBUTIONS[case['distribution']]}"
        modes = {}
        for mode in ("warm", "cold"):
            ragged = case["methods"].get("ragged-auto", {}).get(mode)
            others = [(record[mode]["median_us"], method) for method, record in case["methods"].items()
                      if method in LABELS and mode in record]
            if ragged and others:
                best, method = min(others)
                modes[mode] = (best / ragged["median_us"], method)
        rows.append((label, modes))
    return rows


def draw(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    rows = speedups(summary)
    figure, axis = plt.subplots(figsize=(8, 0.42 * len(rows) + 1.2))
    height = 0.38
    colors = {"warm": "#3b6ea5", "cold": "#e08a2c"}
    for index, (label, modes) in enumerate(rows):
        for offset, mode in ((-height / 2, "warm"), (height / 2, "cold")):
            if mode not in modes:
                continue
            value, method = modes[mode]
            axis.barh(index + offset, value, height=height, color=colors[mode],
                      label=f"{mode} L2" if index == 0 else None)
            axis.text(value * 1.04, index + offset, f"{value:.2f}× vs {LABELS[method]}", va="center", fontsize=7)
    axis.axvline(1.0, color="black", linewidth=0.8)
    axis.set_xscale("log")
    ticks = [0.25, 0.5, 1, 2]
    axis.set_xticks(ticks, [f"{tick:g}×" for tick in ticks])
    axis.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    axis.set_yticks(range(len(rows)), [label for label, _ in rows], fontsize=8)
    axis.invert_yaxis()
    axis.set_xlabel("ragged's speedup over the fastest comparison (log scale; below 1×, ragged is slower)", fontsize=9)
    axis.legend(loc="lower right", fontsize=8)
    axis.set_xlim(right=axis.get_xlim()[1] * 2.2)
    figure.tight_layout()
    figure.savefig(output, metadata={"Date": None})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    draw(json.loads(args.summary.read_text()), args.output)


if __name__ == "__main__":
    main()
