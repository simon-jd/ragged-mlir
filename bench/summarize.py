"""Generate summary.json from the raw results of a bench.suite run.

For every (group, case, method) and L2 state, each run (one seed and
repetition, one process) contributes the median of its timed calls. The
summary reports the median of those per-run medians and their interquartile
range (25th and 75th percentiles), in microseconds, with the number of runs
and the median of each seed's runs (seeds change the segment lengths, which
on the long-tail inputs changes the longest segment by an order of
magnitude). A method that did not measure in every run lists its statuses.
Runs with --cuda-graph add a "graph" state: replays of a CUDA graph captured
from one call, which leave out the host's share of the call.

    python -m bench.summarize RUNS --output results/bench/summary.json
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

TIMING = "complete_call_cuda_event_ms"
MODES = {"warm": lambda cell: cell.get(TIMING), "cold": lambda cell: (cell.get("cold_l2") or {}).get(TIMING),
         "graph": lambda cell: (cell.get("cuda_graph") or {}).get("replay_cuda_event_ms")}


TIE = 0.03   # within 3% of the fastest comparison is a tie


def versus_fastest(methods: dict, mode: str) -> dict | None:
    """ragged-auto against the fastest non-ragged method in one L2 state:
    a win below 0.97 times its time, a tie up to 1.03 times, else a loss."""
    ragged = (methods.get("ragged-auto") or {}).get(mode)
    others = [(record[mode]["median_us"], method) for method, record in methods.items()
              if not method.startswith("ragged") and mode in record]
    if not ragged or not others:
        return None
    fastest_us, fastest = min(others)
    ratio = ragged["median_us"] / fastest_us
    result = "win" if ratio < 1 - TIE else "tie" if ratio <= 1 + TIE else "loss"
    return {"fastest": fastest, "fastest_us": fastest_us, "ragged_us": ragged["median_us"],
            "ratio": round(ratio, 3), "result": result}


def _spread(values):
    values = np.asarray(values, dtype=np.float64)
    p25, median, p75 = np.percentile(values, [25, 50, 75])
    return {"median_us": round(float(median), 2), "p25_us": round(float(p25), 2),
            "p75_us": round(float(p75), 2), "runs": int(values.size)}


def _gpu_range(states, field):
    values = []
    for state in states:
        for gpu in (state or {}).get("gpus", []):
            try:
                values.append(float(gpu[field]))
            except (KeyError, TypeError, ValueError):
                pass
    return [min(values), max(values)] if values else None


def summarize(root: Path) -> dict:
    root = Path(root)
    runs = sorted(root.glob("**/seed-*/rep-*/results.json"))
    if not runs:
        raise ValueError(f"no results.json under {root}")
    cells = defaultdict(lambda: defaultdict(list))      # (group, case) -> method -> cells
    cases, statuses, commits, states, seeds = {}, defaultdict(lambda: defaultdict(int)), set(), [], defaultdict(dict)
    for path in runs:
        relative = path.relative_to(root).parts
        group, case = "/".join(relative[:-4]), relative[-4]
        seed = relative[-3].removeprefix("seed-")
        report = json.loads(path.read_text())
        key = (group, case)
        commits.add((report.get("source") or {}).get("git_commit"))
        states.extend((report.get("gpu_state") or {}).get(moment) for moment in ("before", "after"))
        info = report.get("case") or {}
        if info:
            cases.setdefault(key, {"op": info.get("op"), "dtype": info.get("dtype"), "D": info.get("D"),
                                   "distribution": info.get("distribution"), "rows": info.get("elements"),
                                   "segments": info.get("segments")})
            seeds[key][seed] = [info.get("min_length"), info.get("max_length")]
        for cell in report.get("methods", []):
            statuses[(key, cell["method"])][cell["status"]] += 1
            cells[key][cell["method"]].append({**cell, "_seed": seed})
    summary = []
    for key in sorted(cells):
        entry = {"group": key[0], "case": key[1], **cases.get(key, {}),
                 "lengths_by_seed": dict(sorted(seeds[key].items())), "methods": {}}
        for method, method_cells in sorted(cells[key].items()):
            record = {"statuses": dict(statuses[(key, method)])}
            measured = [cell for cell in method_cells if cell["status"] == "measured"]
            for mode, timing in MODES.items():
                medians = [timing(cell)["median"] * 1e3 for cell in measured if timing(cell)]
                if medians:
                    record[mode] = _spread(medians)
                    by_seed = defaultdict(list)
                    for cell in measured:
                        if timing(cell):
                            by_seed[cell["_seed"]].append(timing(cell)["median"] * 1e3)
                    record[mode]["by_seed_median_us"] = {seed: round(float(np.median(values)), 2)
                                                         for seed, values in sorted(by_seed.items())}
            reasons = sorted({cell.get("reason") or cell.get("error", "")[:200] for cell in method_cells
                              if cell["status"] != "measured"}
                             | {"cuda graph: " + cell["cuda_graph"].get("error", "")[:200] for cell in measured
                                if (cell.get("cuda_graph") or {}).get("status") == "failed"})
            if reasons:
                record["reasons"] = reasons
            if method == "ragged-auto" and measured:
                report = measured[-1].get("last_execution_report") or {}
                entry["classes"] = report.get("classes")
                entry["streams"] = report.get("streams")
                # The schedule the adaptive program chose, per seed.
                chosen = defaultdict(lambda: defaultdict(int))
                for cell in measured:
                    chosen[cell["_seed"]][(cell.get("last_execution_report") or {}).get("selected")] += 1
                record["selected_by_seed"] = {seed: dict(counts) for seed, counts in sorted(chosen.items())}
            if method == "triton" and measured:
                record["tuned_configs"] = sorted({json.dumps((cell.get("tuning") or {}).get("config"), sort_keys=True)
                                                  for cell in measured})
            entry["methods"][method] = record
        verdicts = {mode: versus_fastest(entry["methods"], mode) for mode in MODES}
        if any(verdicts.values()):
            entry["versus_fastest"] = {mode: verdict for mode, verdict in verdicts.items() if verdict}
        summary.append(entry)
    return {"schema": 2, "generated_by": "python -m bench.summarize", "runs": len(runs),
            "source_commits": sorted(commit for commit in commits if commit),
            "statistic": "median and interquartile range over runs of each run's median complete-call CUDA-event time",
            "versus_fastest": "ragged-auto / fastest non-ragged method per state: win < 0.97, tie <= 1.03, loss above",
            "gpu_state": {field: _gpu_range(states, field) for field in
                          ("clocks.sm", "clocks.mem", "temperature.gpu", "power.draw", "utilization.gpu")},
            "host_loadavg_1min": (lambda values: [min(values), max(values)] if values else None)(
                [state["host_loadavg"][0] for state in states if state and state.get("host_loadavg")]),
            "cases": summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(summarize(args.runs), indent=1) + "\n")


if __name__ == "__main__":
    main()
