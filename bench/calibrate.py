"""Fit ragged.selection's cost model on training seeds; evaluate it on held-out ones.

RUNS holds bench.run results that measured both schedules, ragged-classified
and ragged-naive, for example

    python -m bench.suite --groups main --seeds 711 712 --output build/calibration \\
        --methods ragged-classified ragged-naive --lock /var/tmp/gpu.lock
    python -m bench.calibrate fit build/calibration --train 711 712 --output python/ragged/cost_model.json

Each (workload, seed) contributes the median over repetitions of each
process's median complete-call time with a warm L2. For each operation family
and schedule, nonnegative coefficients of the four terms of
ragged.selection.features are fitted to minimize the squared relative error
(every subset of terms is solved by least squares and the best nonnegative
solution kept). The margin is the smallest of 0, 1, ..., 20% for which, on the
training data, every input where classification is chosen measures no slower
than block per segment.

`evaluate` reads a suite's runs (which also time ragged-auto, the adaptive
program) and reports, per held-out workload and seed, both schedules, the one
the model chose and how the adaptive call compares with ragged-naive.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path
import statistics
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from ragged.selection import FAMILIES, SCHEDULES, TERMS, CostModel, family, features  # noqa: E402

METHOD = {"classified": "ragged-classified", "naive": "ragged-naive"}


def collect(runs: Path, l2: str = "warm", group: str = "main") -> dict:
    """{(case name, seed): {"case": case, "methods": {method: [run medians in us]}, "selected": [...]}}"""
    found: dict = {}
    for path in sorted((runs / group).glob("*/seed-*/rep-*/results.json")):
        report = json.loads(path.read_text())
        name, seed = path.parts[-4], int(path.parts[-3].removeprefix("seed-"))
        entry = found.setdefault((name, seed), {"case": report["case"], "methods": defaultdict(list),
                                                "selected": []})
        for cell in report["methods"]:
            if cell.get("status") != "measured":
                continue
            timing = cell["cold_l2"] if l2 == "cold" else cell
            entry["methods"][cell["method"]].append(timing["complete_call_cuda_event_ms"]["median"] * 1000)
            if cell["method"] == "ragged-auto" and cell.get("last_execution_report"):
                entry["selected"].append(cell["last_execution_report"].get("selected"))
    return found


def shape(case: dict) -> tuple[int, int, int, int, int]:
    """(N, B, D, element bytes, longest segment) of a recorded case."""
    return (case["elements"], case["segments"], case["D"], 2 if case["dtype"] == "bf16" else 4,
            case["max_length"])


def nnls(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    best, best_error = np.zeros(a.shape[1]), float(np.sum(b * b))
    for size in range(1, a.shape[1] + 1):
        for subset in combinations(range(a.shape[1]), size):
            solution, *_ = np.linalg.lstsq(a[:, subset], b, rcond=None)
            if np.any(solution < 0):
                continue
            x = np.zeros(a.shape[1])
            x[list(subset)] = solution
            error = float(np.sum((a @ x - b) ** 2))
            if error < best_error:
                best, best_error = x, error
    return best


def fit(found: dict, seeds: list[int]) -> tuple[dict, list[dict]]:
    coefficients: dict = {name: {} for name in FAMILIES}
    rows = []
    for name in FAMILIES:
        points = [(key, entry) for key, entry in sorted(found.items())
                  if key[1] in seeds and family(entry["case"]["op"]) == name
                  and all(entry["methods"].get(METHOD[s]) for s in SCHEDULES)]
        if len(points) < TERMS:
            raise SystemExit(f"{name}: {len(points)} training inputs, need at least {TERMS}")
        for schedule in SCHEDULES:
            x = np.array([features(*shape(entry["case"])) for _, entry in points])
            t = np.array([statistics.median(entry["methods"][METHOD[schedule]]) for _, entry in points])
            coefficients[name][schedule] = [round(float(c), 6) for c in nnls(x / t[:, None], np.ones(len(t)))]
        for key, entry in points:
            rows.append({"case": key[0], "seed": key[1], "family": name,
                         **{s: statistics.median(entry["methods"][METHOD[s]]) for s in SCHEDULES}})
    return coefficients, rows


def margin_for(model_data: dict, found: dict, seeds: list[int]) -> float:
    for percent in range(0, 21):
        model = CostModel.from_dict({**model_data, "margin": percent / 100})
        safe = True
        for (name, seed), entry in found.items():
            if seed not in seeds or not all(entry["methods"].get(METHOD[s]) for s in SCHEDULES):
                continue
            case = entry["case"]
            n, b, d, element, longest = shape(case)
            chosen, _ = model.choose(case["op"], n, b, d, element, longest)
            measured = {s: statistics.median(entry["methods"][METHOD[s]]) for s in SCHEDULES}
            safe &= chosen == "naive" or measured["classified"] <= measured["naive"]
        if safe:
            return percent / 100
    return 0.2


def run_fit(args: argparse.Namespace) -> None:
    found = collect(args.runs)
    coefficients, rows = fit(found, args.train)
    data = {"version": 1, "coefficients": coefficients, "margin": 0.0,
            "calibration": {"runs": str(args.runs), "train_seeds": args.train, "l2": "warm",
                            "statistic": "median over processes of each process's median complete-call time",
                            "terms": "t_us = c0 + c1*N*D*bytes/MiB + c2*L*D*bytes/MiB + c3*B/1000",
                            "inputs": len(rows)}}
    data["margin"] = margin_for(data, found, args.train)
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    model = CostModel.from_dict(data)
    print(f"margin {data['margin']:.2f}; coefficients {json.dumps(coefficients)}")
    print(f"{'case':30} {'seed':>4} {'classified':>10} {'pred':>7} {'naive':>8} {'pred':>7}  chosen")
    for row in rows:
        entry = found[(row["case"], row["seed"])]
        n, b, d, element, longest = shape(entry["case"])
        chosen, info = model.choose(entry["case"]["op"], n, b, d, element, longest)
        predicted = info["predicted_us"]
        print(f"{row['case']:30} {row['seed']:>4} {row['classified']:10.1f} {predicted['classified']:7.1f} "
              f"{row['naive']:8.1f} {predicted['naive']:7.1f}  {chosen}")


def run_evaluate(args: argparse.Namespace) -> None:
    model = CostModel.load(args.model)
    for l2 in ("warm", "cold"):
        found = collect(args.runs, l2)
        print(f"{l2} L2, seeds {' '.join(map(str, args.seeds))}: median complete-call us over repetitions")
        print(f"{'case':30} {'seed':>4} {'classified':>10} {'naive':>8} {'chosen':>10} {'adaptive':>9} "
              f"{'vs naive':>8} {'naive IQR':>10}")
        worst = []
        for (name, seed), entry in sorted(found.items()):
            if seed not in args.seeds or not entry["methods"].get("ragged-auto"):
                continue
            case = entry["case"]
            n, b, d, element, longest = shape(case)
            chosen, _ = model.choose(case["op"], n, b, d, element, longest)
            times = {method: statistics.median(values) for method, values in entry["methods"].items()}
            naive_runs = sorted(entry["methods"]["ragged-naive"])
            q1, q3 = np.percentile(naive_runs, [25, 75])
            ratio = times["ragged-auto"] / times["ragged-naive"]
            worst.append((ratio, name, seed))
            selected = set(entry["selected"]) or {chosen}
            print(f"{name:30} {seed:>4} {times.get('ragged-classified', float('nan')):10.1f} "
                  f"{times['ragged-naive']:8.1f} {'/'.join(sorted(selected)):>10} {times['ragged-auto']:9.1f} "
                  f"{100 * (ratio - 1):+7.1f}% {q1:5.1f}-{q3:<5.1f}")
        if worst:
            ratio, name, seed = max(worst)
            print(f"largest excess over ragged-naive: {100 * (ratio - 1):+.1f}% ({name}, seed {seed})\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    fitter = commands.add_parser("fit", help="fit coefficients and margin on training seeds")
    fitter.add_argument("runs", type=Path)
    fitter.add_argument("--train", type=int, nargs="+", default=[711, 712])
    fitter.add_argument("--output", type=Path, required=True)
    fitter.set_defaults(handler=run_fit)
    evaluator = commands.add_parser("evaluate", help="compare the adaptive program with both schedules")
    evaluator.add_argument("runs", type=Path)
    evaluator.add_argument("--seeds", type=int, nargs="+", default=[713])
    evaluator.add_argument("--model", type=Path, default=Path("python/ragged/cost_model.json"))
    evaluator.set_defaults(handler=run_evaluate)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
