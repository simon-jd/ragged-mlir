"""Run the benchmark suite behind results/bench: every case, seed and repetition.

Each (case, seed, repetition) is one ``bench.run`` process, so run-to-run
variation (allocation placement, clocks, other work on the machine) shows up
in the spread across repetitions. With ``--lock``, each process holds that
file lock (``flock``) while it runs and releases it between processes. A run
whose results.json is complete is skipped, so an interrupted suite resumes.
Inputs (case.npz) and compiler caches are deleted after each run; only
results.json (and any failure tracebacks) remain.

    python -m bench.suite --output runs/2026-10-31 --lock /var/tmp/gpu.lock
    python -m bench.summarize runs/2026-10-31 --output results/bench/summary.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

SHAPES = {1: ("16384", "1048576"), 128: ("4096", "262144")}   # segments, total rows
DISTRIBUTIONS = ("uniform", "long_tail", "burst")
HINTED = {("sum", "uniform"), ("sum", "long_tail"), ("softmax", "uniform")}
ALL_METHODS = ("ragged-naive", "ragged-auto", "torch-segment", "njt-eager", "njt-compile",
               "cuda-naive", "cub", "triton")


def cases():
    """(group, name, bench.run arguments) for every configuration of the suite."""
    main = [(op, "f32", d, dist) for op in ("sum", "max", "softmax") for d in (1, 128) for dist in DISTRIBUTIONS]
    main.append(("softmax", "bf16", 128, "long_tail"))
    out = []
    for op, dtype, d, dist in main:
        segments, rows = SHAPES[d]
        name = f"{op}-{dtype}-d{d}-{dist}"
        base = ["--op", op, "--dtype", dtype, "--d", str(d), "--distribution", dist,
                "--segments", segments, "--total-elements", rows]
        out.append(("main", name, base + ["--methods", *ALL_METHODS]))
        ablations = [] if dtype == "bf16" else [("serial-classes", ["--serial-classes"]),
                                                ("no-row-pipelining", ["--disable-optimization", "row_pipelining"])]
        if op == "softmax":
            ablations.append(("three-pass", ["--disable-optimization", "online_fusion"]))
        for label, flags in ablations:
            out.append((f"ablation/{label}", name, base + flags + ["--methods", "ragged-auto"]))
        if dtype == "f32" and (op, dist) in HINTED:
            methods = ["ragged-auto", "ragged-naive"] + (["torch-segment"] if op == "sum" else [])
            out.append(("hints", name, base + ["--length-hints", "--methods", *methods]))
    return out


def complete(path: Path) -> bool:
    try:
        return json.loads((path / "results.json").read_text()).get("status") not in (None, "running")
    except (OSError, ValueError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[711, 712, 713])
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--groups", nargs="+", default=["main", "hints", "ablation"])
    parser.add_argument("--only", help="run only cases whose name contains this text")
    parser.add_argument("--l2", default="both", choices=("warm", "cold", "both"))
    parser.add_argument("--lock", type=Path, help="hold this flock(1) lock around each benchmark process")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--compiler", default="build/bin/ragged-compile")
    parser.add_argument("--native-library", default="build-baselines/libragged_baselines.so")
    parser.add_argument("--timeout", type=int, default=1800, help="seconds per process, not counting the lock wait")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.repetitions < 1 or not args.seeds:
        parser.error("need at least one seed and one repetition")
    triton_cache = args.output / "triton-cache"
    selected = [case for case in cases() if case[0].split("/")[0] in args.groups
                and (args.only is None or args.only in case[1])]
    # Repetition-major order spreads each case's repetitions over the whole
    # suite instead of running them back to back.
    plan = [(group, name, flags, seed, repetition) for repetition in range(args.repetitions)
            for seed in args.seeds for group, name, flags in selected]
    log = args.output / "suite.log"
    if not args.dry_run:
        args.output.mkdir(parents=True, exist_ok=True)
    for index, (group, name, flags, seed, repetition) in enumerate(plan, 1):
        destination = args.output / group / name / f"seed-{seed}" / f"rep-{repetition}"
        command = ["timeout", str(args.timeout), args.python, "-m", "bench.run",
                   "--compiler", args.compiler, "--native-library", args.native_library,
                   *flags, "--seed", str(seed), "--l2", args.l2, "--triton-cache", str(triton_cache),
                   "--output", str(destination)]
        if args.lock is not None:
            command = ["flock", str(args.lock), *command]
        if args.dry_run:
            print(" ".join(command))
            continue
        if complete(destination):
            continue
        if destination.exists():
            shutil.rmtree(destination)   # an interrupted run of this suite
        started = time.time()
        process = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        code = process.returncode
        if code not in (0, 2) and destination.is_dir():
            (destination / "process.stderr.txt").write_text(process.stderr[-20000:])
        for transient in ("case.npz", "cache", "triton-cache"):
            target = destination / transient
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()
        with log.open("a") as stream:
            stream.write(json.dumps({"index": index, "of": len(plan), "group": group, "case": name, "seed": seed,
                                     "repetition": repetition, "exit": code,
                                     "seconds": round(time.time() - started, 1)}) + "\n")


if __name__ == "__main__":
    main()
