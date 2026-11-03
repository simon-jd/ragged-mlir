"""Interleave ragged-naive and ragged-auto on the D=1 bursty sum of seeds 711-713 in one process.

usage: PYTHONPATH=python:. python results/bench/burst-gap/run.py OUTDIR
Each round times 30 calls of each program (after 5 warm-up calls) with CUDA
events, alternating the order; it also records each call's host enqueue time
(perf_counter around the call, no synchronization) and the kernel time from
torch.profiler.
"""
import json, statistics, sys, time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from bench.run import make_call, prepare_case

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
result = {}
for seed in (711, 712, 713):
    args = SimpleNamespace(op="sum", dtype="f32", d=1, distribution="burst", segments=16384,
                           total_elements=1048576, mean_length=64, seed=seed, offsets_dtype="i64",
                           compiler="build/bin/ragged-compile", length_bounds={}, disable_optimization=[],
                           feature_tile=None, scale=1.0)
    values, offsets, case = prepare_case(args)
    lengths = np.diff(offsets)
    args.sequence_bounds = {"min_len": int(lengths.min()), "max_len": int(lengths.max())}
    tensor = torch.from_numpy(values).cuda()
    bounds = torch.from_numpy(offsets).cuda()
    calls = {}
    for method in ("ragged-naive", "ragged-auto"):
        calls[method], _ = make_call(method, args, torch, tensor, bounds, out / f"{seed}-{method}")
    times = {m: [] for m in calls}
    host = {m: [] for m in calls}
    for round_ in range(10):
        order = list(calls) if round_ % 2 == 0 else list(calls)[::-1]
        for method in order:
            call = calls[method]
            for _ in range(5):
                call()
            torch.cuda.synchronize()
            samples, enqueue = [], []
            for _ in range(30):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                t0 = time.perf_counter_ns()
                start.record(); call(); end.record()
                enqueue.append((time.perf_counter_ns() - t0) / 1e3)
                end.synchronize()
                samples.append(start.elapsed_time(end) * 1e3)
            times[method].append(statistics.median(samples))
            host[method].append(statistics.median(enqueue))
    kernel = {}
    for method, call in calls.items():
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            for _ in range(50):
                call()
            torch.cuda.synchronize()
        durations = [e.device_time for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA
                     and "ragged" in e.name]
        kernel[method] = statistics.median(durations) if durations else None
    result[seed] = {m: {"event_us_per_round": [round(x, 2) for x in times[m]],
                        "event_us": round(statistics.median(times[m]), 2),
                        "host_enqueue_us": round(statistics.median(host[m]), 2),
                        "kernel_us": None if kernel[m] is None else round(kernel[m], 2)} for m in calls}
    print(seed, {m: (result[seed][m]["event_us"], result[seed][m]["host_enqueue_us"], result[seed][m]["kernel_us"])
                 for m in calls}, flush=True)
(out / "burst-gap.json").write_text(json.dumps(result, indent=1))
