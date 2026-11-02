"""Per-kernel GPU time and per-call GPU span of ragged on a saved benchmark case.

`bench.run` saves each workload's inputs as case.npz in its output directory.
Run the calls under Nsight Systems, export its kernel trace, then summarize:

    nsys profile --capture-range=cudaProfilerApi -o /tmp/ragged \\
        python -m bench.profile run OUTPUT/case.npz --op sum
    nsys stats --report cuda_gpu_trace --format csv -o /tmp/ragged /tmp/ragged.nsys-rep
    python -m bench.profile summarize /tmp/ragged_cuda_gpu_trace.csv

The span of a call runs from its first kernel's start to its last kernel's
end; with concurrent class chains it is shorter than the sum of kernel times.
`--strategy naive` profiles the block-per-segment mode, `--strategy classified`
forces device classification on an adaptive program, and `--l2 cold`
overwrites a buffer of four times the L2 size before every call, as the
benchmark's cold-L2 timing does.

`python -m bench.profile host OUTPUT/case.npz --op sum` measures the host time
of a call itself: the GPU is idle when each call starts, so a full launch
queue cannot make the host wait.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import statistics
import tempfile


def load(args: argparse.Namespace):
    import numpy as np
    import torch
    from ragged import ArtifactCache, Compiler, ProgramSpec

    case = np.load(args.case)
    values = torch.from_numpy(case["values"]).cuda()
    offsets = torch.from_numpy(case["offsets"]).cuda()
    spec = ProgramSpec(args.op, "f32", values.shape[1], "i64",
                       strategy="naive" if args.strategy == "naive" else "auto")
    program = Compiler(args.compiler).compile(spec, ArtifactCache(tempfile.mkdtemp()))
    rows = values.shape[0] if args.op == "softmax" else offsets.numel() - 1
    return torch, program, values, offsets, torch.empty((rows, values.shape[1]), device="cuda")


def run(args: argparse.Namespace) -> None:
    torch, program, values, offsets, out = load(args)
    options = {"concurrent": not args.serial_classes}
    if args.strategy == "classified":
        options["select"] = "classified"
    flush = None
    if args.l2 == "cold":
        l2_bytes = torch.cuda.get_device_properties(0).L2_cache_size
        flush = torch.empty((4 * l2_bytes,), dtype=torch.uint8, device="cuda")
    for _ in range(5):
        program.run(values, offsets, out, **options)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    for _ in range(args.calls):
        if flush is not None:
            flush.zero_()
        torch.cuda.synchronize()
        program.run(values, offsets, out, **options)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


def host(args: argparse.Namespace) -> None:
    import time
    torch, program, values, offsets, out = load(args)
    options = {"select": "classified"} if args.strategy == "classified" else {}
    for concurrent in (True, False):
        samples = []
        for _ in range(100 + args.calls):
            torch.cuda.synchronize()
            start = time.perf_counter()
            program.run(values, offsets, out, concurrent=concurrent, **options)
            samples.append((time.perf_counter() - start) * 1e6)
        print(f"classes {'concurrent' if concurrent else 'serial'}: median host time per call "
              f"{statistics.median(samples[100:]):.1f} us over {args.calls} calls")


def summarize(args: argparse.Namespace) -> None:
    lines = Path(args.trace).read_text().splitlines()
    # nsys stats prints notices before the CSV header.
    header = next(index for index, line in enumerate(lines) if "Start (ns)" in line)
    kernels = sorted((int(row["Start (ns)"]), int(row["Duration (ns)"]), row["Name"])
                     for row in csv.DictReader(lines[header:]) if row["Name"].startswith("ragged_"))
    calls: list[list[tuple[int, int, str]]] = []
    for kernel in kernels:
        # Each call begins with the same kernel on the caller's stream (init
        # when classifying) after the previous call's join, so its first
        # kernel separates calls.
        if not calls or kernel[2] == kernels[0][2]:
            calls.append([])
        calls[-1].append(kernel)
    durations = defaultdict(list)
    for call in calls:
        for _, duration, name in call:
            durations[name].append(duration / 1000)
    busy = [sum(duration for _, duration, _ in call) / 1000 for call in calls]
    spans = [(max(start + duration for start, duration, _ in call) - call[0][0]) / 1000 for call in calls]
    print(f"{len(calls)} calls: median kernel time {statistics.median(busy):.1f} us, "
          f"median GPU span {statistics.median(spans):.1f} us")
    total = sum(sum(values) for values in durations.values())
    for name, values in sorted(durations.items(), key=lambda item: -sum(item[1])):
        print(f"  {100 * sum(values) / total:5.1f}%  avg {statistics.mean(values):7.1f} us  x {len(values):3d}  {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    runner = commands.add_parser("run", help="run calls inside a cudaProfilerStart/Stop range")
    runner.add_argument("case", help="a benchmark case.npz")
    runner.add_argument("--op", choices=("sum", "max", "softmax"), required=True)
    runner.add_argument("--compiler", default="build/bin/ragged-compile")
    runner.add_argument("--calls", type=int, default=20)
    runner.add_argument("--serial-classes", action="store_true")
    runner.add_argument("--strategy", choices=("auto", "classified", "naive"), default="auto")
    runner.add_argument("--l2", choices=("warm", "cold"), default="warm")
    runner.set_defaults(handler=run)
    timer = commands.add_parser("host", help="host time per call, each call starting on an idle GPU")
    timer.add_argument("case", help="a benchmark case.npz")
    timer.add_argument("--op", choices=("sum", "max", "softmax"), required=True)
    timer.add_argument("--compiler", default="build/bin/ragged-compile")
    timer.add_argument("--calls", type=int, default=1000)
    timer.add_argument("--strategy", choices=("auto", "classified", "naive"), default="auto")
    timer.set_defaults(handler=host)
    summary = commands.add_parser("summarize", help="summarize an nsys cuda_gpu_trace CSV")
    summary.add_argument("trace")
    summary.set_defaults(handler=summarize)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
