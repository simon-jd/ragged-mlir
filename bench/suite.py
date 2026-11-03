"""Run the benchmark suite behind results/bench: every case, seed and repetition.

Each (case, seed, repetition) is one ``bench.run`` process, so run-to-run
variation (allocation placement, clocks, other work on the machine) shows up
in the spread across repetitions. With ``--lock``, each process holds that
file lock (``flock``) while it runs and releases it between processes; with
``--lock-batch SECONDS`` as well, the suite holds the lock across consecutive
processes for up to that long before releasing it, so that on a shared GPU it
waits for the lock once per batch instead of once per process. A run
whose results.json is complete is skipped, so an interrupted suite resumes.
Inputs (case.npz) and compiler caches are deleted after each run; only
results.json (and any failure tracebacks) remain.

    python -m bench.suite --output runs/2026-10-31 --lock /var/tmp/gpu.lock
    python -m bench.summarize runs/2026-10-31 --output results/bench/summary.json
"""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

SHAPES = {1: ("16384", "1048576"), 128: ("4096", "262144")}   # segments, total rows
DISTRIBUTIONS = ("uniform", "long_tail", "burst")
HINTED = {("sum", "uniform"), ("sum", "long_tail"), ("softmax", "uniform")}
ALL_METHODS = ("ragged-naive", "ragged-classified", "ragged-auto", "torch-segment", "njt-eager", "njt-compile",
               "cuda-naive", "cub", "triton")
# The "compare" group measures the main workloads against every comparison,
# including those added after the main runs (CUB at D=128, cuSPARSE, FBGEMM),
# and times a CUDA-graph replay of each method beside its complete call.
# Eager NJT, never the fastest and up to 12 ms per call, is left out.
COMPARE_METHODS = ("ragged-auto", "ragged-naive", "torch-segment", "njt-compile", "cuda-naive", "cub", "cusparse",
                 "triton", "fbgemm")
# Shapes the cost model was not fitted on, fixed before they were measured:
# (D, B, N). The fitted shapes are D=1 with B=16,384 and D=128 with B=4,096,
# 64 rows per segment on average; these change D, or B at the same N.
HELDOUT_SHAPES = ((1, 4096, 1048576), (1, 65536, 1048576), (32, 8192, 524288), (64, 4096, 262144),
                  (128, 1024, 262144), (128, 16384, 262144), (256, 2048, 131072))
# A second held-out set, fixed after the first one was used to change the
# model and before the changed model was run: segment and row counts that
# no earlier group used, D from 1 to 512 (only D=1 seen before), 10 to 256
# rows on average, and length distributions no earlier group measured.
HELDOUT2_SHAPES = ((1, 200000, 3200000), (2, 32768, 2097152), (8, 131072, 1310720), (16, 12288, 786432),
                   (48, 6000, 393216), (96, 2500, 160000), (192, 768, 196608), (512, 512, 32768))
HELDOUT2_DISTRIBUTIONS = ("bimodal", "exponential")
SCHEDULE_METHODS = ("ragged-classified", "ragged-naive", "ragged-auto")


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
        # Ablations of the classified schedule (the one they change), and of
        # vectorized access for both schedules where D allows it.
        ablations = [] if dtype == "bf16" else [("serial-classes", ["--serial-classes"]),
                                                ("no-row-pipelining", ["--disable-optimization", "row_pipelining"])]
        if op == "softmax" and dtype == "f32":
            ablations.append(("three-pass", ["--disable-optimization", "online_fusion"]))
        for label, flags in ablations:
            out.append((f"ablation/{label}", name, base + flags + ["--methods", "ragged-classified"]))
        if d > 1:
            out.append(("ablation/no-vectorize", name, base + ["--disable-optimization", "vectorize",
                                                              "--methods", "ragged-classified", "ragged-naive"]))
        if dtype == "f32" and (op, dist) in HINTED:
            methods = ["ragged-auto", "ragged-naive"] + (["torch-segment"] if op == "sum" else [])
            out.append(("hints", name, base + ["--length-hints", "--methods", *methods]))
        out.append(("compare", name, base + ["--cuda-graph", "--methods", *COMPARE_METHODS]))
    for d, b, n in HELDOUT_SHAPES:
        for op in ("sum", "max", "softmax"):
            for dist in DISTRIBUTIONS:
                out.append(("heldout", f"{op}-f32-d{d}-b{b}-{dist}",
                            ["--op", op, "--dtype", "f32", "--d", str(d), "--distribution", dist,
                             "--segments", str(b), "--total-elements", str(n), "--methods", *SCHEDULE_METHODS]))
    for d, b, n in HELDOUT2_SHAPES:
        for op in ("sum", "max", "softmax"):
            for dist in HELDOUT2_DISTRIBUTIONS:
                out.append(("heldout2", f"{op}-f32-d{d}-b{b}-{dist}",
                            ["--op", op, "--dtype", "f32", "--d", str(d), "--distribution", dist,
                             "--segments", str(b), "--total-elements", str(n), "--methods", *SCHEDULE_METHODS]))
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
    parser.add_argument("--methods", nargs="+", help="measure only these methods in every selected case")
    parser.add_argument("--l2", default="both", choices=("warm", "cold", "both"))
    parser.add_argument("--lock", type=Path, help="hold this flock(1) lock around each benchmark process")
    parser.add_argument("--lock-batch", type=float, default=0,
                        help="with --lock, hold it across consecutive processes for up to this many seconds")
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
    if args.methods:
        selected = [(group, name, flags[:flags.index("--methods") + 1] + args.methods)
                    for group, name, flags in selected]
    # Repetition-major order spreads each case's repetitions over the whole
    # suite instead of running them back to back.
    plan = [(group, name, flags, seed, repetition) for repetition in range(args.repetitions)
            for seed in args.seeds for group, name, flags in selected]
    log = args.output / "suite.log"
    held = None   # (lock file, time it was acquired) with --lock-batch
    if not args.dry_run:
        args.output.mkdir(parents=True, exist_ok=True)
    for index, (group, name, flags, seed, repetition) in enumerate(plan, 1):
        destination = args.output / group / name / f"seed-{seed}" / f"rep-{repetition}"
        command = ["timeout", str(args.timeout), args.python, "-m", "bench.run",
                   "--compiler", args.compiler, "--native-library", args.native_library,
                   *flags, "--seed", str(seed), "--l2", args.l2, "--triton-cache", str(triton_cache),
                   "--output", str(destination)]
        if args.lock is not None and not args.lock_batch:
            command = ["flock", str(args.lock), *command]
        if args.dry_run:
            print(" ".join(command))
            continue
        if complete(destination):
            continue
        if destination.exists():
            shutil.rmtree(destination)   # an interrupted run of this suite
        if args.lock is not None and args.lock_batch:
            if held is not None and time.time() - held[1] > args.lock_batch:
                held[0].close()          # closing the descriptor releases the lock
                held = None
            if held is None:
                handle = open(args.lock, "a")
                fcntl.flock(handle, fcntl.LOCK_EX)
                held = (handle, time.time())
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
