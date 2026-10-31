# Benchmarking

`bench/run.py` measures one workload with every requested method;
`bench/matrix.py` plans and runs many. The results in the README come from
`results/bench/`: one directory per workload with its `results.json`, the
ablations under `ablation/`, and `summary.json`, which lists each method's
median CUDA-event time from those files in µs. Each `results.json` records the
command that produced it.

```sh
PYTHONPATH=python:. python -m bench.run \
  --op softmax --dtype bf16 --d 128 --distribution long_tail \
  --segments 4096 --total-elements 262144 --output build/bench/softmax-bf16-d128-long_tail
```

The output directory must be absent or empty (a cold artifact cache is part of
the measurement), so a rerun writes a new directory rather than one under
`results/bench/`. The compiler defaults to `build/bin/ragged-compile`
(`--compiler`) and the CUDA/CUB library to
`build-baselines/libragged_baselines.so` (`--native-library`).

## What is measured

- **Correctness first.** Every method's output is compared with the float64
  oracle at the dtype's tolerance before it is timed; a method that fails is a
  failed cell with its traceback, never a timing.
- **Complete calls.** A call includes output and workspace allocation,
  classification, every kernel, NJT construction, dtype promotion and the final
  cast, whatever the method needs. Inputs are on the device beforehand.
- **Two clocks.** CUDA events around the call on its stream (host launch gaps
  count, since the GPU waits for them) and host wall time until completion.
  After 5 warm-up calls, 30 calls are timed; the median, p90, minimum, maximum
  and every sample are kept, with peak extra allocator memory.
- **Ragged extras.** Cold compile time, hot cache lookup, the compiler hash and
  cache key, and the `ExecutionReport` (classes, grids, kernels, host enqueue
  time) of the last call.
- **Inputs.** Each run saves its input arrays, their hashes, the seed and the
  segment-length histogram.

## Methods

| Method | What it is |
|---|---|
| `ragged-auto` | this compiler with device classification |
| `ragged-naive` | this compiler, one block per segment |
| `torch-segment` | `torch.segment_reduce` (sum and max) with f32 accumulation |
| `njt-eager`, `njt-compile` | `torch.nested` jagged tensors, eager and `torch.compile(fullgraph=True)`; max pads explicitly with `-inf` because NJT's own padding breaks empty and all-`-inf` segments |
| `cuda-naive` | one 256-thread CUDA block per segment, threads over rows and up to 32 feature columns, shared-memory tree (`baselines/cuda/baselines.cu`) |
| `cub` | CUB 3.5 `DeviceSegmentedReduce` sum and max; softmax as CUB max, element-parallel exp, CUB sum and element-parallel division (D=1 only) |
| `triton` | a handwritten Triton kernel, one program per segment and feature tile, fixed 32-row tiles, no autotuning, isolated cache per run |
| `fbgemm` | FBGEMM 1.8 `jagged_softmax`, when installed |

Unsupported combinations (CUB at D>1, softmax in `torch.segment_reduce`) are
`n/a` cells; a missing optional package is unmeasured, not substituted.

## Options

- `--length-hints` passes the workload's exact minimum and maximum segment
  length to ragged and NJT (as `min_seqlen`/`max_seqlen`), which lets ragged
  drop classes and the classify pass. Without it NJT gets no length bounds.
  Hinted and unhinted runs are recorded separately; the committed hinted runs
  measure ragged and `torch.segment_reduce` only.
- `--disable-optimization NAME` (classification, online_fusion, warp_mapping,
  split_mapping, shuffle, pointwise_fusion, row_pipelining) and `--feature-tile`
  for ablations.
- `--serial-classes` launches ragged's segment-class worker chains one after
  another on the caller's stream instead of concurrently.
- `--mask` and `--scale` add the softmax mask and scale to every method.
- `--fixed-segments` keeps the segment count and lets the element count vary;
  by default the element count is fixed.

NVTX ranges named `ragged-benchmark/<method>/complete-call` mark each timed
call for `nsys`. `python -m bench.profile` replays a saved case under Nsight
Systems and reports each ragged kernel's mean time and the GPU span of a call
(first kernel start to last kernel end), which is shorter than the summed
kernel time when the class chains overlap; its docstring has the commands.
`python -m bench.profile host` times the host side of a call alone, starting
every call on an idle GPU so that a full launch queue cannot stall the host.
