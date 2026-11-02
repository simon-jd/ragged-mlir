# Benchmarking

`bench/run.py` measures one workload with every requested method in one
process. `bench/suite.py` runs the suite behind `results/bench/`: 19 workloads
with every method, 6 with length hints and 52 ablations, each for 3 seeds of
the length distribution and 5 repetitions, one process per run.
`bench/summarize.py` turns the raw `results.json` files into `summary.json`,
and `bench/calibrate.py` fits the cost model that chooses ragged's schedule
per call and evaluates it on held-out seeds.

```sh
python -m bench.suite --groups main --seeds 711 712 --methods ragged-classified ragged-naive \
    --output build/calibration --lock /var/tmp/gpu.lock                # 190 processes
python -m bench.calibrate fit build/calibration --train 711 712 --output python/ragged/cost_model.json
python -m bench.suite --output build/runs --lock /var/tmp/gpu.lock --lock-batch 600   # 1,155 processes
python -m bench.summarize build/runs --output results/bench/summary.json
python -m bench.calibrate evaluate build/runs --seeds 713
```

Each `results.json` records the command that produced it, the git commit of
the checkout and any modified tracked files, and the GPU state from
`nvidia-smi` (clocks, temperature, power, utilization, other compute
processes) with the host load average, taken before the first method and
after each one. Clocks are not locked. `summary.json` gives, for each method
and L2 state, the median over runs of each run's median call time, the 25th
and 75th percentiles of those run medians, and the number of runs.

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
  and every sample are kept, with peak extra allocator memory. Every call
  starts on an idle GPU.
- **Warm and cold L2.** `--l2 warm` (the default) times back-to-back calls, so
  inputs that fit the 72 MiB L2 stay there. `--l2 cold` overwrites a buffer of
  four times the L2 size and synchronizes before every timed call (the
  overwritten lines are dirty, so the call also pays for writing them back); `--l2 both`
  does both, recording the cold timings under `cold_l2`. The suite uses
  `both`.
- **Ragged extras.** Cold compile time, hot cache lookup, the compiler hash and
  cache key, and the `ExecutionReport` (classes, grids, kernels, host enqueue
  time) of the last call.
- **Inputs.** Each run saves its input arrays (`case.npz`, which the suite
  deletes after each run), their hashes, the seed and the segment-length
  histogram.

## Methods

| Method | What it is |
|---|---|
| `ragged-auto` | this compiler's adaptive program: per call, the cost model (`python/ragged/selection.py`) chooses device classification or one block per segment from N, B, D and the longest segment length, which the benchmark passes as `max_len` (the bound NJT also receives) |
| `ragged-classified` | the same program with device classification forced (`select="classified"`); what `ragged-auto` runs without `max_len` |
| `ragged-naive` | this compiler, one block per segment |
| `torch-segment` | `torch.segment_reduce` (sum and max) with f32 accumulation |
| `njt-eager`, `njt-compile` | `torch.nested` jagged tensors, eager and `torch.compile(fullgraph=True)`; max pads explicitly with `-inf` because NJT's own padding breaks empty and all-`-inf` segments |
| `cuda-naive` | one 256-thread CUDA block per segment, threads over rows and up to 32 feature columns, shared-memory tree (`baselines/cuda/baselines.cu`) |
| `cub` | CUB 3.5 `DeviceSegmentedReduce` sum and max; softmax as CUB max, element-parallel exp, CUB sum and element-parallel division (D=1 only) |
| `triton` | Triton kernels autotuned on the input: one program per segment, or chunked segments with a combine kernel; online softmax ([baselines/README.md](../baselines/README.md)) |
| `fbgemm` | FBGEMM 1.8 `jagged_softmax`, when installed |

Unsupported combinations (CUB at D>1, softmax in `torch.segment_reduce`) are
`n/a` cells; a missing optional package is unmeasured, not substituted.

## Options

- `--length-hints` passes the workload's exact minimum and maximum segment
  length to ragged, which lets it drop classes and the classify pass. NJT
  always gets them, as `min_seqlen`/`max_seqlen`. Hinted and unhinted runs are
  recorded separately.
- `--disable-optimization NAME` (classification, online_fusion, warp_mapping,
  split_mapping, shuffle, pointwise_fusion, row_pipelining, vectorize) and
  `--feature-tile` for ablations. The suite's ablations time
  `ragged-classified`, the schedule that serial classes, row pipelining and
  online statistics change, and both schedules without vector access at D=128.
- `--serial-classes` launches ragged's segment-class worker chains one after
  another on the caller's stream instead of concurrently.
- `--no-max-len` withholds the longest segment length from `ragged-auto`,
  which then keeps device classification.
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
`bench.profile run` takes `--strategy classified|naive` and `--l2 cold`.
`python -m bench.profile flush` times a call with a warm L2, after the
benchmark's flush and after a flush that only reads the buffer: the
benchmark's `zero_()` leaves the L2 full of dirty lines, whose write-back to
DRAM happens during the next call's reads and is part of every method's
cold-L2 time.
