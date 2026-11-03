# ragged-mlir: an MLIR compiler for segmented reductions on ragged tensors

- An out-of-tree MLIR compiler that generates CUDA kernels for segmented sum,
  max and softmax over ragged tensors. Each call runs one of two compiled
  schedules, chosen by a calibrated cost model from N, B, D and the longest
  segment: a warp, block or multi-block mapping per segment chosen on the
  device, or one block per segment.
- On an RTX 4090, against the fastest of seven comparisons per workload
  (autotuned Triton, CUB, cuSPARSE, FBGEMM, `torch.segment_reduce`,
  compiled NJT and a CUDA block-per-segment kernel), it wins 4 of 19
  workloads, ties 4 (within 3%) and loses 11 with a warm L2; with a cold L2
  it wins 1, ties 3 and loses 15; replaying each call as a CUDA graph, which
  removes host overhead from every method, it wins 5, ties 4 and loses 10.
- It wins the uniform D=128 sum and max and the uniform and long-tail D=1
  softmax. Its losses: the D=1 sum and max (1.1 to 2.2 times slower warm,
  1.1 to 3.1 times on graph replay), the long-tail D=128 sum, where cuSPARSE
  is 1.13 times faster, the bursty D=128 sum and max (1.07 times), where the
  cost model classifies although block per segment would tie, and the bursty
  f32 and long-tail bf16 D=128 softmax, where tuned Triton is 1.05 and 1.38
  times faster.
- The schedule choice is a linear cost model. Fitted first on the two
  benchmark shapes, it was within 3% of the faster schedule for only 115 of
  189 inputs on seven held-out shapes. Refitted on all nine shapes, then run
  on a second held-out set fixed beforehand (8 new shapes, D=1 to 512, 512
  to 200,000 segments, bimodal and exponential lengths; 144 inputs), the
  adaptive call is within 3% for 112 inputs warm and 104 cold, where the
  first model's choices are within 3% for 61 and 65. It still misses
  systematically on the D=2 bimodal sum and max (34 to 37% slower) and the
  D=1 softmax with 200,000 segments (8 to 10%).
- Each time is the median of 15 processes (3 seeds of the length distribution
  × 5 repetitions). Every method passes a float64 oracle before it is timed.

![ragged's speedup over the fastest comparison per workload, warm and cold L2](docs/speedup.svg)

Ragged (jagged) batches, a packed `[N, D]` tensor plus `B+1` segment offsets,
are how variable-length sequences, graph neighborhoods and recommendation
features reach a GPU. Reducing each segment (sum, max, softmax) is simple until
the lengths are skewed: one block per segment leaves most of the GPU idle on a
long segment, one warp per segment wastes it on short ones.

`ragged-mlir` is an out-of-tree compiler on MLIR 23.1.3 that generates CUDA
kernels (sm_89) for segmented sum, max and softmax, picks a mapping per
segment on the device, and chooses per call whether that classification is
worth its cost. It has its own dialect and passes, lowers through the GPU
and NVVM dialects to PTX and `ptxas`, caches compiled artifacts, and launches
them through the CUDA Driver API on PyTorch streams.

```python
from ragged import ArtifactCache, Compiler, ProgramSpec

program = Compiler("build/bin/ragged-compile").compile(
    ProgramSpec("softmax", dtype="bf16", feature_dim=128, offsets_dtype="i64"),
    ArtifactCache("~/.cache/ragged"))
program.run(values, offsets, out, max_len=longest)   # CUDA tensors; runs on the current stream
```

`max_len` (the longest segment, which a jagged tensor such as NJT carries)
lets the cost model choose the schedule; without it the program classifies
on the device.

## How it works

- **Dialect.** `ragged.segment_reduce` and `ragged.segment_softmax` over a
  `[N, D]` memref and i32/i64 offsets, with a semantics contract
  ([docs/semantics.md](docs/semantics.md)): f32 accumulation for f32 and bf16,
  NaN propagation and `-inf` identities for max, empty segments, exact output
  shapes. A verifier rejects ill-formed programs.
- **Mapping.** `ragged-legalize` makes accumulation, identity and NaN handling
  explicit. `ragged-plan` assigns segments to three classes by length: up to
  32 rows one warp (a shuffle tree, no shared memory), up to 4,096 rows one
  128-thread block, longer segments 8 blocks with partial results and a combine
  kernel. Caller-supplied `min_len`/`max_len` hints prune classes at
  compile time; otherwise a classify kernel bins segments into per-class
  worklists on the device, and persistent workers for each class drain them.
  Each warp of the classify kernel ranks its lanes per class with one shuffle
  scan and reserves its queue positions with at most three atomics.
- **Choosing per call.** Classification costs an init and a classify kernel,
  three worker chains (each launched even when its queue is empty) and a
  stream fork and join; one block per segment costs one launch but puts each
  segment on one block. A classifying program therefore also compiles the
  block-per-segment specialization, and before each call a linear cost model
  (`python/ragged/selection.py`) predicts both complete-call times from N, B,
  D, the longest segment, their skew and the bytes that do not fit the L2,
  and runs the faster. Its coefficients are fitted on the benchmark
  workloads and seven further shapes (`bench/calibrate.py`,
  `results/bench/calibration/`) and evaluated on eight shapes they were not
  fitted on ([docs/design-tradeoffs.md](docs/design-tradeoffs.md)). The
  longest segment comes from the caller; without it the program classifies.
- **Scheduling.** Each persistent worker grid is one resident wave of that
  kernel, measured with the occupancy API when the module loads (on an RTX
  4090, up to 1,536 128-thread blocks or 3,072 warps, fewer for kernels with
  more registers, divided among feature tiles and split parts). The three classes touch disjoint segments, so their worker
  chains run on separate streams between a fork after classification and a
  join on the caller's stream, the CTA chain submitted first
  ([docs/classified-abi.md](docs/classified-abi.md)). Why each of these
  choices was made, and what it costs, is in
  [docs/design-tradeoffs.md](docs/design-tradeoffs.md).
- **Softmax.** `ragged-fuse` chooses the statistics schedule per segment
  class. A warp or CTA class computes the maximum and exponential sum in one
  online pass and normalizes in a second pass, in one kernel. Each of a split
  segment's eight blocks computes the same online statistics for its part, a
  combine kernel merges the eight, and a normalization kernel writes the
  output: two reads of the rows instead of three. The online statistics are
  compensated sums against a reference that trails the maximum by at most
  one, so their accuracy does not depend on segment length and they need no
  length hints. In-place softmax uses three passes in every class, the
  schedule that separates reads from writes. Optional f32 scale and additive
  mask.
- **Lowering.** `ragged-to-gpu` emits the kernels with feature-dimension
  tiling (`d_tile`) and gives each thread `d_vector` adjacent features (four
  f32 or eight bf16 when D allows), read and written with one aligned 16-byte
  vector access per row, then the standard GPU → NVVM → PTX pipeline and
  `ptxas` produce a cubin. Compensated f32 summation keeps bf16 and f32 sums within the
  float64 oracle's tolerance. Each lane's row loop issues eight loads before
  folding them in row order, so eight loads are in flight per lane and the
  arithmetic, and therefore every result bit, is that of the one-row loop
  ([docs/optimizations.md](docs/optimizations.md)).
- **Artifacts and runtime.** The compiler writes a cubin and a manifest (kernel
  order, grids, workspace shapes, ptxas resource usage, toolchain provenance).
  The Python runtime validates tensors against the manifest, allocates one
  scratch buffer for all workspace on the consumer stream, and launches through
  a small C++ Driver API module (`runtime/`): one `dlopen`/`cuInit` per process,
  a launch plan per problem size that issues every kernel and the stream fork
  and join in one native call, and modules unloaded only after the work
  launched from them has finished (an event per stream that used them). A
  checked mode validates the offsets on the device before any main kernel runs.

## Results

**Correctness.** Numerical results in the GPU tests and the benchmark are
checked against a float64 NumPy oracle with dtype-specific tolerances
(`oracle/reference.py`). On an RTX 4090 (driver 595.99, CUDA 13.2)
([results/tests/summary.txt](results/tests/summary.txt),
[results/sanitizer/summary.txt](results/sanitizer/summary.txt)):

| Suite | Result |
|---|---|
| GPU tests (`test/gpu`): every op, dtype, D in {1, 2, 6, 7, 12, 16, 24, 40, 128, 1024}, i32/i64 offsets, four length distributions, empty segments, masks, in-place, checked offsets, streams, CUDA graph capture, Driver API faults; both schedules of the adaptive program and its choice; vector access compared bitwise against one feature per thread, row pipelining and concurrent classes against the rolled, serial schedule; misaligned tensors rejected; online and three-pass softmax in every class, including 65,536-row segments; class queues read back for counts, membership and per-warp order; the Triton, CUDA, CUB, cuSPARSE and FBGEMM comparisons | 2,170 passed; 30 strict xfails where FBGEMM's `jagged_softmax` returns NaN |
| compute-sanitizer memcheck, racecheck, initcheck and synccheck on the classified, mask, checked-offsets, scheduling, selection and per-class softmax suites (671 tests, one process per suite and tool), at commit `f0bd690`; the generated GPU code is byte for byte the same since | 0 errors, 0 hazards |
| lit/FileCheck tests of the dialect, passes and lowering (`test/`) | 41 passed |
| host tests (`test/python`): cache, manifest validation, runtime contracts and stream schedule, cost model and schedule choice, benchmark harness, suite and summary | 1,173 passed; CI runs them on a CPU runner ([workflow](.github/workflows/host-tests.yml)), where the 118 that need the compiler or the CUDA library skip |

**Speed.** Complete-call GPU time in µs: CUDA events around the call, so host
launch gaps count, median over 15 processes (seeds 711 to 713 of the length
distribution × 5 repetitions) of each process's median of 30 calls. f32
unless noted; 1M rows in 16,384 segments at D=1, 262,144 rows in 4,096
segments at D=128; uniform (64 rows), long-tail (Pareto) and bursty lengths.
**Warm** L2 is back-to-back calls; **cold** L2 overwrites a 288 MiB buffer and
synchronizes before every call; **graph** replays a CUDA graph captured from
one call, which runs the same kernels without the host's share of the call.
`ragged` is the adaptive program, given the exact longest segment
(`max_len`, the bound NJT also receives); *chose* is its schedule per seed
711/712/713, C classification, N one block per segment. A result is a win
when ragged takes less than 0.97 times the fastest comparison's time, a tie
within 3%, a loss above. The comparisons ([baselines/README.md](baselines/README.md))
are Triton autotuned on each input, CUB 3.5 (at D=128 on a transposed copy),
cuSPARSE SpMM (sums), FBGEMM 1.9 (the D=1 sum and softmax),
`torch.segment_reduce` (sum, max), compiled NJT with exact length bounds and
a CUDA kernel with one block per segment. Clocks are not locked (SM clock
2,520 to 2,805 MHz) and the GPU and host were shared with other work.

| Op | D | Lengths | chose | ragged | fastest comparison | warm | cold | CUDA graph replay: ragged, fastest | graph |
|---|---:|---|---|---:|---|---|---|---|---|
| sum | 1 | uniform | N/N/N | 34 | torch.segment_reduce 15 | loss 2.19× | loss 1.71× vs torch.segment_reduce | 16, NJT compiled 5.4 | loss 3.06× |
| sum | 1 | long tail | C/N/N | 35 | cuSPARSE 22 | loss 1.56× | loss 1.20× vs CUB | 17, Triton 16 | loss 1.07× |
| sum | 1 | bursty | N/N/N | 33 | cuSPARSE 23 | loss 1.47× | loss 1.16× vs CUB | 16, Triton 13 | loss 1.27× |
| sum | 128 | uniform | C/C/C | 136 | torch.segment_reduce 155 | win 0.87× | loss 1.23× vs NJT compiled | 102, NJT compiled 143 | win 0.71× |
| sum | 128 | long tail | C/C/C | 191 | cuSPARSE 169 | loss 1.13× | loss 1.15× vs cuSPARSE | 173, cuSPARSE 160 | loss 1.08× |
| sum | 128 | bursty | C/C/C | 181 | cuSPARSE 169 | loss 1.07× | loss 1.07× vs cuSPARSE | 159, Triton 151 | loss 1.05× |
| max | 1 | uniform | N/N/N | 34 | torch.segment_reduce 15 | loss 2.19× | loss 2.00× vs torch.segment_reduce | 16, NJT compiled 5.4 | loss 3.02× |
| max | 1 | long tail | C/N/N | 34 | Triton 31 | loss 1.11× | tie 1.03× vs CUB | 16, Triton 13 | loss 1.23× |
| max | 1 | bursty | N/N/N | 34 | CUB 27 | loss 1.26× | loss 1.18× vs CUB | 16, Triton 12 | loss 1.35× |
| max | 128 | uniform | C/C/C | 134 | torch.segment_reduce 156 | win 0.86× | loss 1.22× vs NJT compiled | 102, NJT compiled 143 | win 0.71× |
| max | 128 | long tail | C/C/C | 181 | Triton 180 | tie 1.00× | tie 1.02× vs Triton | 163, Triton 162 | tie 1.00× |
| max | 128 | bursty | C/C/C | 181 | Triton 169 | loss 1.07× | loss 1.04× vs Triton | 158, Triton 151 | loss 1.05× |
| softmax | 1 | uniform | N/N/N | 36 | NJT compiled 45 | win 0.80× | win 0.89× vs NJT compiled | 18, NJT compiled 26 | win 0.72× |
| softmax | 1 | long tail | C/C/C | 55 | Triton 65 | win 0.86× | loss 1.08× vs Triton | 34, Triton 46 | win 0.73× |
| softmax | 1 | bursty | N/N/N | 37 | Triton 37 | tie 1.00× | loss 1.16× vs Triton | 18, Triton 19 | win 0.96× |
| softmax | 128 | uniform | C/C/C | 311 | Triton 314 | tie 0.99× | loss 1.07× vs Triton | 288, Triton 296 | tie 0.97× |
| softmax | 128 | long tail | C/C/C | 403 | Triton 392 | tie 1.03× | tie 1.02× vs Triton | 391, Triton 383 | tie 1.02× |
| softmax | 128 | bursty | C/C/C | 504 | Triton 479 | loss 1.05× | loss 1.06× vs Triton | 474, Triton 463 | tie 1.02× |
| softmax bf16 | 128 | long tail | C/C/C | 304 | Triton 221 | loss 1.38× | loss 1.31× vs Triton | 278, Triton 182 | loss 1.53× |

Every method's time in every state, the earlier run with eager NJT and the
forced classified schedule, the ablations, vector access, length hints, the
cost model on held-out shapes and the kernel traces are in
[results/bench/README.md](results/bench/README.md).

What the numbers say:

- **At D=1 the sum and max lose, and not only to host time.** A warm call
  takes 33 to 35 µs, of which the kernel is 14.4 to 14.8 µs
  ([profile.txt](results/bench/profile.txt), [burst-gap](results/bench/README.md#seed-711s-bursty-d1-sum));
  replayed as a graph it takes 16 to 17 µs. The comparisons gain as much from replay:
  `torch.segment_reduce` takes 11 µs, Triton 12 to 16, and compiled NJT 5.4
  µs on uniform lengths, where exact bounds turn it into a dense reduction.
  One block per segment over 16,384 short segments is slower than their
  kernels, not only their launch path.
- **cuSPARSE is the strongest comparison for sums.** Its SpMM takes 22 to 23
  µs at D=1 and 169 to 170 µs at D=128 for every length distribution, 1.13
  times faster than ragged on the long-tail D=128 sum; on graph replay it
  is slower than ragged at D=1 (22 against 16 to 17 µs). CUB at D=128 needs a
  transpose and D·B segments and takes 1.2 to 4.2 ms; FBGEMM's D=1 sum takes
  26 to 55 µs and its `jagged_softmax` 74 µs to 2.9 ms.
- **ragged wins where its mapping matters.** On the uniform D=128 sum and max,
  persistent CTA workers take 134 to 136 µs against 155 for
  `torch.segment_reduce` (102 against 143 for compiled NJT on graph replay);
  on the D=1 softmax its one-pass online statistics beat compiled NJT and
  Triton by 14 to 20%.
- **With a cold L2 it mostly loses, mostly for reasons that are not the
  cold L2.** 10 of the 15 cold losses are warm losses too: the five D=1 sum
  and max losses (host time and short blocks), the long-tail and bursty
  D=128 sum and the bursty D=128 max, and the bursty and bf16 D=128 softmax.
  Of the other five, the uniform D=128 sum and max lose to compiled NJT, a
  dense reduction under exact uniform bounds (187 against 230 µs):
  classification's warm lead there is the part of the 128 MiB input that
  back-to-back calls find in the L2, and cold even block per segment is
  1.16 times slower than NJT, so this is inherent to the schedules. The cold
  L2 costs the D=1 long-tail and bursty softmax 15 and 8 µs against 1 and 2
  µs for Triton (not traced further). On the uniform D=128 softmax block per
  segment would tie (331 against 324 µs for Triton, classified 348) but is
  3% slower warm; the model does not know the L2 state. Only the schedule
  choice was cheap to change: the refit turned the long-tail D=128 max from
  a cold loss (1.03) into a tie, and a margin against classification gains
  less cold than it costs warm
  ([design-tradeoffs](docs/design-tradeoffs.md#choosing-classification-or-one-block-per-segment-per-call)).
  The benchmark's flush leaves the L2 full of dirty lines, which every method
  pays to write back ([results/bench/cold-l2.txt](results/bench/cold-l2.txt)).
- **Classification versus one block per segment.** Forced classification
  loses to block per segment in 10 of 19 warm workloads. The first cost
  model, fitted on these two shapes, removed those losses here but on seven
  other shapes was within 3% of the faster schedule for 115 of 189 inputs:
  with 65,536 segments at D=1 its per-segment term, which had absorbed the
  fixed cost of a call, predicted 158 µs for classification (57 measured),
  and on the uniform D=32 and D=64 sum and max, whose 64 MiB inputs stay in
  the L2, it charged DRAM time for every element and classified, 35 to 41%
  slower ([profile.txt](results/bench/profile.txt): block per segment's
  kernel takes 16 µs there). The refit, trained on those inputs too, gets
  both right; on these workloads it
  classifies the bursty D=128 sum and max, 8% slower than block per segment,
  because N, B, D and the longest segment do not tell bursty from long-tail
  lengths ([results/bench/README.md](results/bench/README.md#the-cost-model-on-held-out-shapes)).

## Build and run

LLVM/MLIR 23.1.3 (`scripts/bootstrap_llvm.sh` builds it), CUDA 13.2 with
`ptxas`, CMake, Ninja, Python 3.12 with PyTorch 2.14 (CUDA 13.2 build) and
nanobind.

```sh
scripts/bootstrap_llvm.sh          # builds LLVM/MLIR in .deps/llvm-build, writes .deps/ragged-source-sdk.cmake
cmake -G Ninja -C .deps/ragged-source-sdk.cmake -S . -B build
ninja -C build && ninja -C build check-ragged          # compiler and lit tests
cmake -S runtime -B build-runtime && cmake --build build-runtime && cmake --install build-runtime --prefix python
python -m pytest                                       # host tests (test/python); a CPU PyTorch is enough
PYTHONPATH=python RAGGED_COMPILE=build/bin/ragged-compile python -m pytest test/gpu
PYTHONPATH=python:. python -m bench.run --op softmax --dtype bf16 --d 128 \
  --distribution long_tail --segments 4096 --total-elements 262144 \
  --output build/bench/softmax-bf16-d128-long_tail
```

With the official LLVM 23.1.3 Linux SDK instead of a source build (how the
published results were built), configure with
`-DMLIR_DIR=<sdk>/lib/cmake/mlir` and point
`-DRAGGED_LLVM_TEST_TOOLS_DIR` at a directory holding `FileCheck` and `not`,
which the SDK does not ship ([docs/provenance.md](docs/provenance.md)).

`bench.run` refuses an output directory that is not empty, so the command
above writes a new one under `build/`; compare it with
`results/bench/runs/main/softmax-bf16-d128-long_tail/seed-711/`. The
`cuda-naive` and `cub` methods need the comparison library that
`baselines/README.md` shows how to build. The main suite (1,155
processes) and the first held-out shapes, the cost model's fit on them, the
comparison group and the second held-out set (run with the fitted model),
the summary, the held-out evaluation of the fitted and of the previous
model, and the chart:

```sh
python -m bench.suite --output build/runs --lock /var/tmp/gpu.lock --lock-batch 600
python -m bench.suite --groups heldout --repetitions 3 --output build/runs --lock /var/tmp/gpu.lock --lock-batch 600
python -m bench.calibrate fit build/runs --groups main heldout --output python/ragged/cost_model.json
python -m bench.suite --groups compare --output build/runs --lock /var/tmp/gpu.lock --lock-batch 600
python -m bench.suite --groups heldout2 --repetitions 3 --output build/runs --lock /var/tmp/gpu.lock --lock-batch 600
python -m bench.summarize build/runs --output build/summary.json
python -m bench.calibrate evaluate build/runs --group heldout2
python -m bench.calibrate evaluate build/runs --group heldout2 --model results/bench/calibration/cost_model-v1.json
python -m bench.chart build/summary.json --group compare --output build/speedup.svg
```

## Limitations

- sm_89 only; one device per program; forward only (no autograd).
- Validated mode trusts the offsets; checked mode validates them on the device
  at the cost of a stream synchronization before the main kernels.
- The schedule choice needs the longest segment length from the caller
  (`max_len`); without it the program always classifies, which is 19 to 55%
  slower than one block per segment at D=1 (the `ragged, classified`
  column). Computing the statistics on the device would need a readback and a
  synchronization per call. A wrong `max_len` costs time, not correctness.
- The cost model is linear in six terms and calibrated on warm times of 82
  workloads in nine shapes (three length distributions) on one GPU, with the
  RTX 4090's 72 MiB L2 as a constant. It sees N, B, D and the longest
  segment, not the distribution of lengths: it classifies the bursty D=128
  sum and max, 8% slower there than block per segment. On the second held-out
  set it picks the slower schedule by more than 3% for 32 of 144 inputs
  warm (40 cold), by 34 to 37% on the D=2 bimodal sum and max
  ([results/bench/README.md](results/bench/README.md#the-cost-model-on-held-out-shapes)).
  With a cold L2 it classifies the uniform D=128 inputs where block per
  segment is 5% faster. Other GPUs are untested.
- Segment-class bounds (32 and 4,096 rows) and the 8-way split are fixed in the
  compiler and the classified ABI, not tuned per GPU or per D.
- Vector access makes the bf16 long-tail softmax slower (302 against 264 µs
  classified) and needs 16-byte-aligned tensors; a view that starts mid-row
  needs a `vectorize=False` specialization.
- Host time is 24 to 26 µs per classified call and 14 to 15 µs per
  block-per-segment call (`results/bench/host.txt`): tensor validation,
  descriptor construction and the scratch allocation in Python, then one
  native call. At D=1 CUB, cuSPARSE, Triton and, on uniform lengths,
  `torch.segment_reduce` are faster; on CUDA-graph replay, without host
  time, Triton still is, and so are `torch.segment_reduce` and compiled NJT
  on uniform lengths.
- The online fold is one dependency chain per lane. When few segments are
  long, as for the split class of the long-tail inputs, its kernel is bound by
  latency rather than bandwidth. Recentering once per batch of eight pipelined
  rows would shorten the chain, but would change results against the rolled
  loop, which row pipelining now reproduces bit for bit.
- Tuned Triton is 1 to 5% faster on skewed D=128 f32 inputs and 1.38 times
  faster on the bf16 long-tail softmax; cuSPARSE is 1.13 times faster on the
  long-tail D=128 sum.
- With a cold L2 the D=128 uniform sum reads its input at 0.59 TB/s. Most of
  the gap to DRAM bandwidth is the write-back of the dirty lines the
  benchmark's flush leaves in the L2 (`results/bench/cold-l2.txt`), which
  every method pays; compiled NJT is still 1.2 times faster there.
- Persistent workers take queue positions with a fixed stride, so their speed
  depends on the order classification leaves in the queues and on which SM
  each worker lands. In separate A/B runs (not part of `results/`), the
  warp-ordered queues made the long-tail D=128 sum and max about 6% slower
  than per-segment atomics, and submitting the split chain before the CTA
  chain made the uniform D=128 sum take 176 µs instead of 133
  ([docs/design-tradeoffs.md](docs/design-tradeoffs.md)).
- One GPU (an RTX 4090 that also drives a desktop and is shared with other
  projects), unlocked clocks, and a host running other work; synthetic length
  distributions plus three seeds. The Triton reference is tuned on each
  benchmark input; `torch_scatter` is not measured.

## Layout

| Path | What it is |
|---|---|
| `include/Ragged/`, `lib/IR/` | the dialect, ops and verifiers |
| `lib/Transforms/` | legalization and mapping selection |
| `lib/Conversion/ToGPU.cpp` | lowering to GPU kernels |
| `tools/` | `ragged-opt` and `ragged-compile` |
| `python/ragged/` | frontend, artifact cache, manifest validation, runtime, schedule cost model (`selection.py`, `cost_model.json`), PyTorch op registration |
| `runtime/` | CUDA Driver API module loader and launcher (nanobind) |
| `oracle/` | float64 reference and test-case minimizer |
| `baselines/`, `bench/` | comparison implementations and the benchmark harness |
| `scripts/` | `bootstrap_llvm.sh` (LLVM/MLIR source build) and `build_baselines.py` (the CUDA/CUB comparison library) |
| `test/` | lit tests (`IR`, `Transforms`, `Conversion`), host tests, GPU tests |
| `docs/` | semantics, ABIs, optimizations, error model, design trade-offs |
| `results/` | benchmark runs and summary, cost-model calibration and evaluation, traces, test and sanitizer summaries |
