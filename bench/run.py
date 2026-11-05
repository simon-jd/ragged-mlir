"""Correctness-qualified, complete-call CUDA benchmark; never simulates a GPU.

Run from the checkout with PYTHONPATH=python. Every requested method produces a
cell, including unsupported or failing methods. Timing begins only after that
method passes the shared independent oracle. Synchronization here belongs to
the measurement harness, not to CompiledProgram.run's validated contract.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import time
import traceback

import numpy as np

from bench.distributions import DISTRIBUTIONS, describe, make_lengths, offsets_from_lengths
from bench.device import cuda_driver_info, gpu_state, source_state
from oracle import assert_matches, quantize_bf16


ROOT = Path(__file__).resolve().parents[1]
L2_MODES = ("warm", "cold", "both")
METHODS = ("ragged-naive", "ragged-classified", "ragged-auto", "torch-segment", "njt-eager", "njt-compile", "cuda-naive", "cub",
           "cusparse", "triton", "fbgemm")


class NotApplicable(Exception):
    """A known semantic/API incompatibility; unexpected errors remain failures."""


class Unmeasured(Exception):
    """A requested optional comparison dependency is absent, not inapplicable."""


def summarize(samples: list[float]) -> dict:
    if not samples or any(not np.isfinite(x) or x < 0 for x in samples):
        raise ValueError("measurements must be finite, nonnegative, and nonempty")
    return {"count": len(samples), "median": statistics.median(samples),
            "min": min(samples), "max": max(samples),
            "p90": float(np.percentile(samples, 90)), "samples": samples}


def prepare_case(args):
    lengths = make_lengths(args.distribution, args.segments, seed=args.seed,
                           total_elements=args.total_elements, mean_length=args.mean_length)
    offsets = offsets_from_lengths(lengths, args.offsets_dtype)
    values = np.random.default_rng(args.seed + 1).normal(size=(int(offsets[-1]), args.d)).astype(np.float32)
    if args.dtype == "bf16":
        values = quantize_bf16(values)
    return values, offsets, describe(lengths) | {
        "distribution": args.distribution, "seed": args.seed,
        "normalization": "fixed_elements" if args.total_elements is not None else "fixed_segments",
        "mean_length_parameter": args.mean_length,
        "values_sha256": hashlib.sha256(values.astype("<f4").tobytes()).hexdigest(),
        "offsets_sha256": hashlib.sha256(offsets.tobytes()).hexdigest(),
        "op": args.op, "dtype": args.dtype, "offsets_dtype": args.offsets_dtype, "D": args.d,
    }


def ragged_spec(method, args, scale, masked):
    """The ProgramSpec a ragged-* method compiles (bench.precompile builds the same)."""
    from ragged import Hints, Optimizations, ProgramSpec
    optimizations = Optimizations(**{name: False for name in getattr(args, "disable_optimization", [])},
                                  feature_tile=getattr(args, "feature_tile", None))
    # ragged-auto chooses a schedule per call from the cost model;
    # ragged-classified is the same program with device classification forced.
    return ProgramSpec(args.op, args.dtype, args.d, args.offsets_dtype,
                       strategy="naive" if method == "ragged-naive" else "auto", scale=scale, mask=masked,
                       optimizations=optimizations, hints=Hints(**getattr(args, "length_bounds", {})),
                       chip=getattr(args, "chip", "sm_89"))


def make_call(method, args, torch, values, offsets, destination, mask=None):
    """Return a complete-call callable and method preparation evidence."""
    scale = float(np.float32(getattr(args, "scale", 1.0)))

    def baseline_input():
        if scale == 1.0 and mask is None:
            return values
        transformed = values.float()
        if scale != 1.0:
            transformed = transformed * scale
        if mask is not None:
            transformed = transformed + mask
        return transformed

    def baseline_output(result):
        return result.to(values.dtype)

    if method == "fbgemm":
        if args.op == "max" or (args.op == "sum" and args.d != 1):
            raise NotApplicable("FBGEMM has segment_sum_csr for the D=1 sum and jagged_softmax; no max "
                                "or per-feature sum")
        from baselines import fbgemm_reference
        max_len = getattr(args, "length_bounds", {}).get("max_len")
        try:
            configuration = fbgemm_reference.configuration(
                values.shape[0], values.shape[1], offsets.numel() - 1, max_len=max_len)
        except ValueError as exc:
            raise NotApplicable(f"FBGEMM public kernel shape/bound limitation: {exc}") from exc
        try:
            metadata = fbgemm_reference.toolchain()
        except fbgemm_reference.OptionalDependencyUnavailable as exc:
            raise Unmeasured(str(exc)) from exc
        metadata = {**metadata, **configuration,
                    "baseline_strategy": "public FBGEMM segment_sum_csr (D=1 sum) or jagged_softmax; packed input, "
                                         "f32 promotion, original-dtype output; no truncated segments"}
        return lambda: baseline_output(fbgemm_reference.call(
            baseline_input(), offsets, args.op, max_len=max_len)), metadata

    if method == "triton":
        # An isolated cache (the default) makes the first call include compilation;
        # a shared --triton-cache leaves only the tuning in it.
        cache = getattr(args, "triton_cache", None)
        os.environ["TRITON_CACHE_DIR"] = str(Path(cache).resolve() if cache else (destination / "triton-cache").resolve())
        from baselines import triton_reference
        metadata = triton_reference.toolchain().copy()
        start = time.perf_counter_ns()
        def call():
            return baseline_output(triton_reference.call(baseline_input(), offsets, args.op))
        output = call()
        torch.cuda.synchronize()
        del output
        metadata["compile_and_first_call_wall_ms"] = (time.perf_counter_ns() - start) / 1e6
        # The first call times every candidate configuration on this input.
        metadata["tuning"] = triton_reference.last_tuning()
        metadata["mapping"] = ("autotuned on this input: one program per segment, or chunks of a tuned "
                               "row count with a combine kernel; online softmax")
        return call, metadata
    if method in ("cuda-naive", "cub"):
        from baselines.native import NativeBaselines
        baseline = NativeBaselines(args.native_library)
        mapping = {"cuda-naive": "one 256-thread block per segment; threads over rows and up to 32 feature columns; "
                                 "shared-memory tree; softmax in three passes",
                   "cub": "CUB DeviceSegmentedReduce sum and max; softmax: CUB max, element-parallel exp, "
                          "CUB sum, element-parallel division; D>1 on a feature-major copy, one CUB segment "
                          "per (feature, segment), transposed back"}[method]
        return (lambda: baseline_output(baseline.call(method, baseline_input(), offsets, args.op)),
                {**baseline.provenance, "mapping": mapping})
    if method.startswith("ragged-"):
        from ragged import ArtifactCache, Compiler
        record = getattr(args, "toolchain_record", None)
        compiler = Compiler(args.compiler, toolchain_record=record)
        spec = ragged_spec(method, args, scale, mask is not None)
        if record is not None:
            # Precompiled artifacts (bench.precompile); no compiler runs here.
            cache = ArtifactCache(args.cache)
        else:
            # Refuse a reused result directory so 'cold' always means absent artifact.
            cache = ArtifactCache(destination / "cache" / method)
        start = time.perf_counter_ns()
        program = compiler.compile(spec, cache)
        cold_ms = (time.perf_counter_ns() - start) / 1e6
        if record is not None:
            cold_ms = None   # loaded, not compiled: there is no cold compile to time
        elif program.cache_hit:
            raise RuntimeError("cold compile unexpectedly hit the cache")
        start = time.perf_counter_ns()
        hot = compiler.compile(spec, cache)
        hot_ms = (time.perf_counter_ns() - start) / 1e6
        if not hot.cache_hit:
            raise RuntimeError("hot compile did not hit the cache")
        rows = values.shape[0] if args.op == "softmax" else offsets.numel() - 1
        metadata = {"cold_compile_wall_ms": cold_ms, "hot_cache_wall_ms": hot_ms,
                    "cache_key": program.artifact.key.digest,
                    "compiler_sha256": program.artifact.key.compiler_sha256,
                    "toolchain": compiler.toolchain(chip=spec.chip), "spec": spec.to_dict(),
                    "precompiled": record is not None, "last_execution_report": None}

        select = {"ragged-classified": "classified"}.get(method, "auto")
        if select == "classified" and program.naive is None:
            raise NotApplicable("the program does not classify on the device; ragged-auto is the same schedule")
        # The longest segment length, the metadata NJT also receives; without
        # it (--no-max-len) the program keeps device classification.
        max_len = None if getattr(args, "no_max_len", False) else args.sequence_bounds["max_len"]
        metadata["max_len_given"] = max_len is not None

        def call():
            # Caller output allocation is included, as for torch/NJT return values.
            out = torch.empty((rows, args.d), device=values.device, dtype=values.dtype)
            report = program.run(values, offsets, out, mask=mask,
                                 concurrent=not getattr(args, "serial_classes", False),
                                 select=select, max_len=max_len)
            metadata["_last_execution_report_object"] = report
            return out
        return call, metadata
    from baselines import torch_reference
    torch_provenance = {"source_sha256": hashlib.sha256(Path(torch_reference.__file__).read_bytes()).hexdigest()}
    if method == "cusparse":
        if args.op != "sum":
            raise NotApplicable("cuSPARSE SpMM computes segment sums only")
        if offsets.dtype != torch.int64:
            raise NotApplicable("the CSR baseline uses i64 row pointers")
        n = values.shape[0]
        columns = torch.arange(n, dtype=offsets.dtype, device=values.device)
        ones = torch.ones(n, dtype=torch.float32, device=values.device)
        return (lambda: baseline_output(torch_reference.csr_sum(baseline_input(), offsets, columns, ones)),
                {**torch_provenance, "mapping": "torch CSR @ dense (cusparseSpMM): a [B, N] CSR matrix of ones, "
                                                "built per call around per-shape column indices and values"})
    if method == "torch-segment":
        if args.op == "softmax":
            raise NotApplicable("torch.segment_reduce has no softmax operation")
        return lambda: baseline_output(torch_reference.segment_reduce(baseline_input(), offsets, args.op)), torch_provenance
    if offsets.numel() == 1:
        raise NotApplicable("public NJT construction with B=0 is unsupported by the pinned torch")
    if args.offsets_dtype != "i64":
        raise NotApplicable("NJT baseline requires i64 offsets; no hidden conversion is timed")
    fn = {"sum": torch_reference.njt_sum, "max": torch_reference.njt_max,
          "softmax": torch_reference.njt_softmax}[args.op]
    njt_provenance = {**torch_provenance, "baseline_strategy":
        "public NJT construction and explicit -inf padding before amax; N=0 shape identity"
        if args.op == "max" else "public NJT " + args.op}
    # NJT carries its sequence lengths as metadata, so it gets the exact
    # min/max segment lengths in every run (ragged only with --length-hints).
    bounds = getattr(args, "sequence_bounds", None) or getattr(args, "length_bounds", {})
    if method == "njt-compile":
        # Do not substitute eager if Inductor/fullgraph rejects the workload.
        start = time.perf_counter_ns()
        def complete_njt():
            return baseline_output(fn(baseline_input(), offsets, **bounds))
        compiled = torch.compile(complete_njt, fullgraph=True)
        result = compiled()
        torch.cuda.synchronize()
        del result
        return compiled, {**njt_provenance,
            "compile_and_first_call_wall_ms": (time.perf_counter_ns() - start) / 1e6,
            "fullgraph": True,
        }
    return lambda: baseline_output(fn(baseline_input(), offsets, **bounds)), njt_provenance


def _timed(call, torch, method, repeats, flush):
    device_ms, full_wall_ms = [], []
    for _ in range(repeats):
        if flush is not None:
            # Overwrite more than the L2 cache, then start from an idle GPU.
            flush.zero_()
            torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        with torch.cuda.nvtx.range(f"ragged-benchmark/{method}/complete-call"):
            started = time.perf_counter_ns()
            start.record()
            result = call()
            end.record()
            end.synchronize()
            full_wall_ms.append((time.perf_counter_ns() - started) / 1e6)
            device_ms.append(start.elapsed_time(end))
            del result
    return {"complete_call_cuda_event_ms": summarize(device_ms),
            "complete_call_wall_until_complete_ms": summarize(full_wall_ms)}


def measure(call, torch, *, method, warmup, repeats, l2="warm", flush=None):
    """Warm timings in the top-level fields; with an L2 flush buffer, cold ones under cold_l2."""
    for _ in range(warmup):
        out = call()
        del out
    torch.cuda.synchronize()
    baseline_allocated = torch.cuda.memory_allocated()
    baseline_reserved = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    memory_result = call()
    torch.cuda.synchronize()
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    peak = peak_allocated - baseline_allocated
    del memory_result
    timings = _timed(call, torch, method, repeats, None) if l2 in ("warm", "both") else {}
    if l2 in ("cold", "both"):
        timings["cold_l2"] = {**_timed(call, torch, method, repeats, flush),
                              "flush_bytes": flush.numel() * flush.element_size(),
                              "scope": "every call starts after overwriting the flush buffer and synchronizing"}
    return {**timings,
            "peak_additional_allocated_bytes": peak,
            "allocator_baseline_allocated_bytes": baseline_allocated,
            "allocator_peak_allocated_bytes": peak_allocated,
            "allocator_baseline_reserved_bytes": baseline_reserved,
            "allocator_peak_reserved_bytes": peak_reserved,
            "memory_scope": "PyTorch allocator counters; excludes CUDA context/module and non-allocator allocations",
            "allocation_scope": "output, temporaries, workspace, NJT construction and dtype conversions",
            "timing_scope": "all kernels and host enqueue gaps; inputs already on device; no CUDA graph",
            "warmup_calls": warmup}


def measure_graph(call, torch, *, repeats, check) -> dict:
    """Capture one complete call in a CUDA graph and time its replays.

    A replay launches the captured kernels without the Python and driver work
    of a call, so the replay time separates GPU time from host overhead. The
    captured output must pass the same oracle as the call itself.
    """
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):   # warm up off the capture stream, as torch.cuda.graphs recommends
        for _ in range(3):
            warm = call()
            del warm
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = call()
    graph.replay()
    torch.cuda.synchronize()
    oracle = check(result)
    for _ in range(5):
        graph.replay()
    samples = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    del result, graph
    return {"status": "measured", "replay_cuda_event_ms": summarize(samples), "oracle": oracle,
            "scope": "CUDA events around one replay of a graph captured from one complete call; warm L2"}


def run(args) -> tuple[dict, int]:
    report = {"schema": 1, "status": "running", "timestamp_unix_ns": time.time_ns(),
              "command": list(os.sys.argv), "python": platform.python_version(), "methods": [],
              "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "source": source_state(ROOT), "l2": getattr(args, "l2", "warm")}
    try:
        torch = importlib.import_module("torch")
        report["torch"] = torch.__version__
        report["torch_cuda"] = torch.version.cuda
        if not torch.cuda.is_available():
            raise NotApplicable("no CUDA device is available; no GPU measurements were made")
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        from ragged.model import runs_on
        chip = getattr(args, "chip", "sm_89")
        if not runs_on(chip, (props.major, props.minor)):
            raise NotApplicable(f"{chip} cubins do not run on sm_{props.major}{props.minor}")
        report["device"] = {"name": props.name, "capability": [props.major, props.minor],
                            "total_memory_bytes": props.total_memory,
                            "multiprocessors": props.multi_processor_count}
        report["cuda_driver"] = cuda_driver_info(torch.cuda.current_device())
    except (ImportError, NotApplicable) as exc:
        report.update(status="blocked", reason=str(exc))
        return report, 2
    values, offsets, case = prepare_case(args)
    lengths = np.diff(offsets.astype(np.int64))
    args.sequence_bounds = {"min_len": int(lengths.min()) if len(lengths) else 0,
                            "max_len": int(lengths.max()) if len(lengths) else 0}
    args.length_bounds = dict(args.sequence_bounds) if getattr(args, "length_hints", False) else {}
    mask = (np.random.default_rng(args.seed + 2).normal(0, 0.1, values.shape).astype(np.float32)
            if getattr(args, "mask", False) else None)
    scale = float(np.float32(getattr(args, "scale", 1.0)))
    case.update(scale=scale, mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest() if mask is not None else None,
                disabled_optimizations=getattr(args, "disable_optimization", []),
                feature_tile=getattr(args, "feature_tile", None), length_bounds=args.length_bounds,
                njt_sequence_bounds=args.sequence_bounds)
    values_bytes = values.size * (2 if args.dtype == "bf16" else 4)
    case.update(values_bytes=values_bytes, offsets_bytes=offsets.nbytes,
                mask_bytes=mask.nbytes if mask is not None else 0,
                input_bytes=values_bytes + offsets.nbytes + (mask.nbytes if mask is not None else 0),
                input_bytes_scope="resident values, offsets and optional mask; excludes outputs and method workspace")
    report["case"] = case
    np.savez(args.output / "case.npz", values=values, offsets=offsets, **({"mask": mask} if mask is not None else {}))
    with np.errstate(invalid="ignore", over="ignore"):
        oracle_values = np.multiply(values, np.float32(scale), dtype=np.float32)
        if mask is not None:
            oracle_values = np.add(oracle_values, mask, dtype=np.float32)
    tensor = torch.from_numpy(values).to("cuda", dtype=torch.float32 if args.dtype == "f32" else torch.bfloat16)
    bounds = torch.from_numpy(offsets).to("cuda")
    mask_tensor = torch.from_numpy(mask).to("cuda") if mask is not None else None
    l2 = getattr(args, "l2", "warm")
    flush = None
    if l2 != "warm":
        l2_bytes = report["cuda_driver"].get("l2_cache_size_bytes") or 72 << 20
        flush = torch.empty((4 * l2_bytes,), dtype=torch.uint8, device="cuda")
    torch.cuda.synchronize()
    report["gpu_state"] = {"before": gpu_state()}
    for method in args.methods:
        cell = {"method": method}
        try:
            call, details = make_call(method, args, torch, tensor, bounds, args.output, mask_tensor)
            # Bind failed oracle/runtime attempts to the implementation actually
            # prepared, rather than retaining provenance only for timed success.
            cell.update({key: value for key, value in details.items() if not key.startswith("_")})
            result = call()
            actual = result.float().cpu().numpy()
            cell["oracle"] = assert_matches(actual, oracle_values, offsets, operation=args.op, input_dtype=args.dtype)
            del result, actual
            cell.update(measure(call, torch, method=method, warmup=args.warmup, repeats=args.repeats,
                                l2=l2, flush=flush))
            if getattr(args, "cuda_graph", False):
                def check(output):
                    return assert_matches(output.float().cpu().numpy(), oracle_values, offsets,
                                          operation=args.op, input_dtype=args.dtype)
                try:
                    cell["cuda_graph"] = measure_graph(call, torch, repeats=args.repeats, check=check)
                except Exception as exc:   # a method that cannot be captured keeps its other timings
                    cell["cuda_graph"] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"[:500]}
            execution_report = details.pop("_last_execution_report_object", None)
            if execution_report is not None:
                details["last_execution_report"] = execution_report.to_dict()
            cell.update(details)
            cell["status"] = "measured"
        except NotApplicable as exc:
            cell.update(status="not_applicable", reason=str(exc))
        except Unmeasured as exc:
            cell.update(status="unmeasured", reason=str(exc))
        except Exception as exc:
            cell.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            (args.output / f"{method}.stderr.txt").write_text(traceback.format_exc())
        report["methods"].append(cell)
        report["gpu_state"]["after"] = gpu_state()
        # Persist each completed cell before another method can fail or crash.
        (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    failed = any(cell["status"] == "failed" for cell in report["methods"])
    unmeasured = any(cell["status"] == "unmeasured" for cell in report["methods"])
    measured = sum(cell["status"] == "measured" for cell in report["methods"])
    report["measured_method_count"] = measured
    report["status"] = "failed" if failed else "partial" if unmeasured else "complete" if measured else "no_applicable_methods"
    return report, 1 if failed else 2 if unmeasured else 0 if measured else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op", choices=("sum", "max", "softmax"), default="sum")
    parser.add_argument("--dtype", choices=("f32", "bf16"), default="f32")
    parser.add_argument("--offsets-dtype", choices=("i32", "i64"), default="i64")
    parser.add_argument("--d", type=int, default=128)
    parser.add_argument("--distribution", choices=DISTRIBUTIONS, default="long_tail")
    parser.add_argument("--segments", type=int, default=16384)
    parser.add_argument("--total-elements", type=int, default=1 << 20)
    parser.add_argument("--fixed-segments", action="store_true", help="ignore --total-elements")
    parser.add_argument("--mean-length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=711)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--compiler", default="build/bin/ragged-compile")
    parser.add_argument("--chip", choices=("sm_86", "sm_89"), default="sm_89", help="target of the ragged programs")
    parser.add_argument("--toolchain-record", type=Path,
                        help="load ragged programs precompiled into --cache (bench.precompile); runs no compiler")
    parser.add_argument("--cache", type=Path, help="with --toolchain-record: the precompiled artifact cache")
    parser.add_argument("--native-library", default="build-baselines/libragged_baselines.so")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--mask", action="store_true", help="dense f32 additive softmax mask included in every method's complete call")
    parser.add_argument("--disable-optimization", action="append", default=[],
                        choices=("classification", "online_fusion", "warp_mapping", "split_mapping", "shuffle", "pointwise_fusion",
                                 "row_pipelining", "vectorize"))
    parser.add_argument("--feature-tile", type=int, choices=(1, 2, 4, 8, 16, 32))
    parser.add_argument("--length-hints", action="store_true",
                        help="supply the workload's exact min/max segment lengths to ragged (NJT always gets them)")
    parser.add_argument("--no-max-len", action="store_true",
                        help="do not pass the longest segment length to ragged's per-call schedule choice")
    parser.add_argument("--serial-classes", action="store_true",
                        help="launch ragged's segment-class workers one after another on the caller's stream")
    parser.add_argument("--l2", choices=L2_MODES, default="warm",
                        help="time with a warm L2, after flushing it before every call (cold), or both")
    parser.add_argument("--triton-cache", type=Path, help="shared Triton kernel cache (default: one per run)")
    parser.add_argument("--cuda-graph", action="store_true",
                        help="also time replays of a CUDA graph captured from one call (warm L2)")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if (args.toolchain_record is None) != (args.cache is None):
        parser.error("--toolchain-record and --cache go together")
    if not 1 <= args.d <= 1024 or args.warmup < 0 or args.repeats < 1:
        parser.error("require D=1..1024, warmup>=0 and repeats>=1")
    if len(set(args.methods)) != len(args.methods):
        parser.error("methods must be unique")
    if not math.isfinite(args.scale) or abs(args.scale) > np.finfo(np.float32).max:
        parser.error("scale must be finite f32")
    if args.mask and args.op != "softmax":
        parser.error("mask is supported only for softmax")
    if len(set(args.disable_optimization)) != len(args.disable_optimization):
        parser.error("optimization switches must be unique")
    if args.fixed_segments:
        args.total_elements = None
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output directory must be absent or empty to preserve cold-cache measurements")
    args.output.mkdir(parents=True, exist_ok=True)
    report, code = run(args)
    (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"{report['status']}: {args.output / 'results.json'}")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
