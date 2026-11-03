"""Fit ragged.selection's cost model on measured runs; evaluate it on held-out ones.

RUNS holds bench.run results that measured both schedules, ragged-classified
and ragged-naive, in one or more suite groups, for example

    python -m bench.calibrate fit results/bench/runs --groups main heldout --train 711 712 713 \\
        --output python/ragged/cost_model.json

Each (group, workload, seed) contributes the median over repetitions of each
process's median complete-call time with a warm L2. For each operation family
and schedule, nonnegative coefficients of the six terms of
ragged.selection.features are fitted to minimize the squared relative error
(every subset of terms is solved by least squares and the best nonnegative
solution kept). The margin by which classification must be predicted faster
is chosen by leave-one-shape-out cross-validation: each (N, B, D) shape in
turn is left out, the coefficients are fitted on the others, and the margin
of 0, 1, ..., 10% that puts the most left-out inputs within 3% of the faster
schedule is kept (the smallest on a tie).

`evaluate` reads a suite's runs (which also time ragged-auto, the adaptive
program) and reports, per workload and seed, both schedules, the one the
model chooses, the adaptive call as measured (it ran the model installed
when the runs were made) and both against the faster schedule measured in
the same runs. `--model` replays another model's choices (version 1 models
included) on the same runs.
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
from ragged.optimizations import Optimizations  # noqa: E402
from ragged.selection import FAMILIES, SCHEDULES, TERMS, CostModel, family, features  # noqa: E402

METHOD = {"classified": "ragged-classified", "naive": "ragged-naive"}
NOISE_TIE = 0.03   # what the results call a tie
MARGINS = [percent / 100 for percent in range(11)]
L2_MIB = 72.0      # RTX 4090: 72 MiB of L2, the GPU every committed run measured


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


def shape(case: dict) -> tuple[int, int, int, int, int, int]:
    """(N, B, D, longest segment, features per tile, element bytes) of a recorded case."""
    options = Optimizations(**{name: False for name in case.get("disabled_optimizations", [])})
    tile, vector = options.geometry(case["D"], case["dtype"])
    return (case["elements"], case["segments"], case["D"], case["max_length"], tile * vector,
            2 if case["dtype"] == "bf16" else 4)


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


def measured(entry: dict) -> dict[str, float]:
    return {s: statistics.median(entry["methods"][METHOD[s]]) for s in SCHEDULES}


def points(found: dict, seeds: list[int]) -> list[tuple[tuple, dict]]:
    """Training inputs: entries of the given seeds that measured both schedules."""
    return [(key, entry) for key, entry in sorted(found.items())
            if key[-1] in seeds and all(entry["methods"].get(METHOD[s]) for s in SCHEDULES)]


def fit(inputs: list[tuple[tuple, dict]]) -> dict:
    coefficients: dict = {name: {} for name in FAMILIES}
    for name in FAMILIES:
        chosen = [entry for _, entry in inputs if family(entry["case"]["op"]) == name]
        if len(chosen) < TERMS:
            raise SystemExit(f"{name}: {len(chosen)} training inputs, need at least {TERMS}")
        x = np.array([features(*shape(entry["case"]), op=entry["case"]["op"], l2_mib=L2_MIB) for entry in chosen])
        for schedule in SCHEDULES:
            t = np.array([measured(entry)[schedule] for entry in chosen])
            coefficients[name][schedule] = [round(float(c), 6) for c in nnls(x / t[:, None], np.ones(len(t)))]
    return coefficients


def model_data(coefficients: dict, margin: float, calibration: dict) -> dict:
    return {"version": 2, "coefficients": coefficients, "margin": margin, "l2_mib": L2_MIB,
            "calibration": calibration}


def within(model: CostModel, inputs: list[tuple[tuple, dict]]) -> int:
    """How many inputs the model's choice puts within 3% of the faster schedule."""
    count = 0
    for _, entry in inputs:
        chosen, _ = model.choose(entry["case"]["op"], *shape(entry["case"]))
        times = measured(entry)
        count += times[chosen] <= min(times.values()) * (1 + NOISE_TIE)
    return count


def margin_by_shape(inputs: list[tuple[tuple, dict]]) -> tuple[float, dict[float, int]]:
    """The margin that leave-one-shape-out cross-validation puts the most inputs within 3% with."""
    shapes = sorted({shape(entry["case"])[:3] for _, entry in inputs})
    scores = dict.fromkeys(MARGINS, 0)
    for left_out in shapes:
        train = [item for item in inputs if shape(item[1]["case"])[:3] != left_out]
        test = [item for item in inputs if shape(item[1]["case"])[:3] == left_out]
        coefficients = fit(train)
        for margin in MARGINS:
            scores[margin] += within(CostModel.from_dict(model_data(coefficients, margin, {})), test)
    best = max(scores.values())
    return min(margin for margin, score in scores.items() if score == best), scores


def run_fit(args: argparse.Namespace) -> None:
    found = {(group, *key): entry for group in args.groups
             for key, entry in collect(args.runs, "warm", group).items()}
    inputs = points(found, args.train)
    coefficients = fit(inputs)
    margin, scores = margin_by_shape(inputs)
    shapes = sorted({shape(entry["case"])[:3] for _, entry in inputs})
    data = model_data(coefficients, margin, {
        "runs": str(args.runs), "groups": args.groups, "train_seeds": args.train, "l2": "warm",
        "statistic": "median over processes of each process's median complete-call time",
        "terms": "t_us = c0 + c1*M + c2*S + c3*B/1000 + c4*M*log2(skew) + c5*X, M = N*D/2^18, "
                 "S = L*d_tile*d_vector/128/1000, skew = L*B/N, X = MiB of input rows plus output "
                 "(B rows, N for softmax) beyond l2_mib",
        "margin": "leave-one-shape-out cross-validation over 0..10%",
        "inputs": len(inputs), "shapes": len(shapes)})
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    model = CostModel.from_dict(data)
    print(f"{len(inputs)} inputs in {len(shapes)} (N, B, D) shapes; coefficients {json.dumps(coefficients)}")
    print("left-out inputs within 3% per margin: " +
          ", ".join(f"{round(100 * m)}% {score}" for m, score in scores.items()) + f"; margin {margin:.2f}")
    print(f"within 3% of the faster schedule on the training inputs: {within(model, inputs)} of {len(inputs)}")
    print(f"{'group':8} {'case':32} {'seed':>4} {'classified':>10} {'pred':>7} {'naive':>8} {'pred':>7}  chosen")
    for (group, name, seed), entry in inputs:
        chosen, info = model.choose(entry["case"]["op"], *shape(entry["case"]))
        predicted, times = info["predicted_us"], measured(entry)
        print(f"{group:8} {name:32} {seed:>4} {times['classified']:10.1f} {predicted['classified']:7.1f} "
              f"{times['naive']:8.1f} {predicted['naive']:7.1f}  {chosen}")


def run_evaluate(args: argparse.Namespace) -> None:
    model = CostModel.load(args.model)
    for l2 in ("warm", "cold"):
        found = collect(args.runs, l2, args.group)
        print(f"{l2} L2, {args.group} runs, seeds {' '.join(map(str, args.seeds))}, model {args.model} "
              f"(version {model.version}): median complete-call us over repetitions")
        print(f"{'case':34} {'seed':>4} {'classified':>10} {'naive':>8} {'chooses':>10} {'vs best':>8} "
              f"{'adaptive ran':>12} {'adaptive':>9} {'vs best':>8}")
        outcomes, adaptive = defaultdict(int), defaultdict(int)
        for (name, seed), entry in sorted(found.items()):
            if seed not in args.seeds or not all(entry["methods"].get(METHOD[s]) for s in SCHEDULES):
                continue
            case = entry["case"]
            chosen, _ = model.choose(case["op"], *shape(case))
            times = measured(entry)
            best = min(times.values())
            regret = times[chosen] / best - 1
            outcomes["within 3%" if regret <= NOISE_TIE else "more than 3% slower"] += 1
            line = (f"{name:34} {seed:>4} {times['classified']:10.1f} {times['naive']:8.1f} {chosen:>10} "
                    f"{100 * regret:+7.1f}%")
            if entry["methods"].get("ragged-auto"):
                auto = statistics.median(entry["methods"]["ragged-auto"])
                ran = "/".join(sorted(set(entry["selected"])))
                adaptive["within 3%" if auto / best - 1 <= NOISE_TIE else "more than 3% slower"] += 1
                line += f" {ran:>12} {auto:9.1f} {100 * (auto / best - 1):+7.1f}%"
            print(line)
        total = sum(outcomes.values())
        print("the model's choice against the faster schedule: " +
              ", ".join(f"{count} of {total} {label}" for label, count in sorted(outcomes.items())))
        if adaptive:
            print("the adaptive call as measured (with the model of those runs) against the faster schedule: " +
                  ", ".join(f"{count} of {sum(adaptive.values())} {label}" for label, count in sorted(adaptive.items())))
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    fitter = commands.add_parser("fit", help="fit coefficients and margin on training runs")
    fitter.add_argument("runs", type=Path)
    fitter.add_argument("--groups", nargs="+", default=["main"], help="suite groups to train on")
    fitter.add_argument("--train", type=int, nargs="+", default=[711, 712, 713])
    fitter.add_argument("--output", type=Path, required=True)
    fitter.set_defaults(handler=run_fit)
    evaluator = commands.add_parser("evaluate", help="compare the model's choice and the adaptive program with both schedules")
    evaluator.add_argument("runs", type=Path)
    evaluator.add_argument("--seeds", type=int, nargs="+", default=[711, 712, 713])
    evaluator.add_argument("--model", type=Path, default=Path("python/ragged/cost_model.json"))
    evaluator.add_argument("--group", default="heldout2", help="suite group to evaluate")
    evaluator.set_defaults(handler=run_evaluate)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
