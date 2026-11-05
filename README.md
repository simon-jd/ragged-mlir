# ragged-mlir: an MLIR compiler for segmented reductions on ragged tensors

ragged-mlir generates CUDA kernels for segmented sum, max and softmax over
ragged tensors. Skewed segment lengths make any one mapping wasteful: one
block per segment idles the GPU on a long segment, one warp per segment
wastes it on short ones. ragged-mlir bins segments on the device into warp,
block and multi-block workers, and a fitted cost model decides per call
whether that binning beats plain block per segment.

Scope: one RTX 4090 (sm_89), MLIR 23.1.3, CUDA 13.2; f32 and bf16;
synthetic segment lengths, three seeds each. The GPU tests, both held-out
sets and the 19 benchmark workloads also ran on one NVIDIA A10-24Q vGPU
(sm_86) on a Snowflake trial compute pool, driver 570 with the CUDA 13.2
forward-compatibility driver, from programs compiled ahead of time
([results/a10](results/a10/README.md)).

| Measure (warm L2 unless noted) | ragged | Baseline | Baseline value |
|---|---:|---|---:|
| Held-out inputs within 3% of the faster schedule, of 144 | 112 | first cost model | 61 |
| The same, cold L2 | 104 | first cost model | 65 |
| Uniform D=128 sum, µs | 136 | torch.segment_reduce | 155 |
| Uniform D=1 softmax, µs | 36 | compiled NJT | 45 |
| Uniform D=1 sum, µs (loss) | 34 | torch.segment_reduce | 15 |
| Long-tail bf16 D=128 softmax, µs (loss) | 304 | tuned Triton | 221 |

Rows 1 and 2: `results/bench/calibration/heldout2.txt`, on 8 shapes fixed
before the refitted model ran. Rows 3 to 6: `results/bench/summary.json`
(group `compare`), medians of 15 processes. On all 19 benchmark workloads
ragged wins 4, ties 4 and loses 11 against the fastest of seven
comparisons; the cost model was fitted on them, so that is in-sample.

How the 4090's cost model transfers to the A10 (6 MiB of L2 against 72
MiB), on the same 144 held-out inputs, choices replayed on the A10's runs of
both schedules ([results/a10/transfer.txt](results/a10/transfer.txt)):

| On the A10: inputs within 3% of the faster schedule, of 144 | warm | cold |
|---|---:|---:|
| 4090 model as shipped (its 72 MiB L2) | 120 | 113 |
| 4090 model with the A10's L2 (what the runtime does; 126 and 123 as run) | 126 | 122 |
| refitted on A10 runs of the same 246 training inputs | 138 | 134 |

On the A10 ragged wins 3 of the 19 workloads, ties 2 and loses 14 warm, and
wins 3, ties 3 and loses 13 cold.

## Limitations

- It loses 11 of the 19 benchmark workloads warm and 15 cold. The D=1 sum
  and max are 1.1 to 2.2 times slower than the fastest comparison. Tuned
  Triton is 1.38 times faster on the bf16 long-tail softmax. On the A10 it
  loses 14 warm and 13 cold, the uniform D=128 sum, max and softmax among
  them.
- The cost model sees N, B, D and the longest segment, not the length
  distribution. On the held-out set it is more than 3% slower than the
  faster schedule for 32 of 144 inputs warm (40 cold), by 34 to 37% on the
  D=2 bimodal sum and max.
- The schedule choice needs the longest segment (`max_len`) from the
  caller; without it the program always classifies.
- On the A10 the shipped model, with the A10's L2, is more than 3% slower
  than the faster schedule for 18 of 144 held-out inputs warm (22 cold), by
  up to 18% (the D=1 softmax with 200,000 segments, which the refit misses
  too). The 4090 model's fixed costs are half the A10's.
- Targets sm_86 and sm_89 only. One shared RTX 4090 with unlocked clocks,
  and one A10 vGPU without compute-sanitizer, CUPTI or clock locking;
  forward only.

More: [how it works](#how-it-works) · [design trade-offs](docs/design-tradeoffs.md) ·
[results](#results) · [all benchmark tables](results/bench/README.md) · [result files](results/README.md) ·
[build and run](#build-and-run) · [all limitations](#limitations-in-detail)

## Overview and usage

Ragged (jagged) batches, a packed `[N, D]` tensor plus `B+1` segment offsets,
are how variable-length sequences, graph neighborhoods and recommendation
features reach a GPU. Reducing each segment (sum, max, softmax) is simple until
the lengths are skewed.

`ragged-mlir` is an out-of-tree compiler on MLIR 23.1.3 that generates CUDA
kernels (sm_89) for segmented sum, max and softmax, picks a mapping per
segment on the device, and chooses per call whether that classification is
worth its cost. It has its own dialect and passes, lowers through the GPU
and NVVM dialects to PTX and `ptxas`, caches compiled artifacts, and launches
them through the CUDA Driver API on PyTorch streams.

Each call runs one of two compiled schedules: a warp, block or multi-block
mapping per segment, chosen on the device, or one block per segment. The
cost model picks between them from N, B, D and the longest segment.

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

### Correctness

Numerical results in the GPU tests and the benchmark are checked against a
float64 NumPy oracle with dtype-specific tolerances (`oracle/reference.py`).
Every method passes it before it is timed. On an RTX 4090 (driver 595.99,
CUDA 13.2) ([results/tests/summary.txt](results/tests/summary.txt),
[results/sanitizer/summary.txt](results/sanitizer/summary.txt)):

| Suite | Result |
|---|---|
| GPU tests (`test/gpu`): every op, dtype, D in {1, 2, 6, 7, 12, 16, 24, 40, 128, 1024}, i32/i64 offsets, five length distributions, empty segments, masks, in-place, checked offsets, streams, CUDA graph capture, Driver API faults; both schedules of the adaptive program and its choice; vector access compared bitwise against one feature per thread, row pipelining and concurrent classes against the rolled, serial schedule; misaligned tensors rejected; online and three-pass softmax in every class, including 65,536-row segments; class queues read back for counts, membership and per-warp order; the Triton, CUDA, CUB, cuSPARSE and FBGEMM comparisons | 2,330 passed; 30 strict xfails where FBGEMM's `jagged_softmax` returns NaN |
| compute-sanitizer memcheck, racecheck, initcheck and synccheck on the classified, mask, checked-offsets, scheduling, selection and per-class softmax suites (671 tests, one process per suite and tool) | 0 errors, 0 hazards |
| lit/FileCheck tests of the dialect, passes and lowering (`test/`) | 45 passed |
| the GPU tests on one A10-24Q vGPU (sm_86), every program compiled ahead of time and no compiler on the machine ([results/a10](results/a10/README.md)) | 2,324 passed; 6 skipped (the Driver fault probes need `ptxas` and an sm_89 device); the same 30 strict xfails. No compute-sanitizer: GPU debugging is disabled on the vGPU |
| host tests (`test/python`): cache, manifest validation, runtime contracts and stream schedule, cost model and schedule choice, benchmark harness, suite and summary | 1,213 passed; CI runs them on a CPU runner ([workflow](.github/workflows/host-tests.yml)), where the 119 that need the compiler or the CUDA library skip |

The 4090 GPU, lit and host tests ran at `0e4f674`, which adds the sm_86
target and leaves the sm_89 GPU dialect, PTX and manifests of 530 programs
byte for byte unchanged; the compute-sanitizer runs are from `0c4be17`,
before it.

### The cost model on held-out shapes

The schedule choice is a linear cost model (`python/ragged/selection.py`).
Before each call it predicts both complete-call times and runs the faster.

The first version was fitted on the two benchmark shapes. On seven held-out
shapes it was within 3% of the faster schedule for only 115 of 189 inputs.

The refit trained on those inputs too: 246 inputs in nine shapes. It was
then evaluated once on a second held-out set, fixed beforehand: 8 new shapes, D=1 to
512, 512 to 200,000 segments, bimodal and exponential lengths, 144 inputs.
Inputs within 3% of the faster of the two schedules
([results/bench/calibration/heldout2.txt](results/bench/calibration/heldout2.txt)):

| L2 | Refitted model, as run | Refitted model's choices, replayed | First model's choices, replayed |
|---|---:|---:|---:|
| warm | 112 of 144 | 119 of 144 | 61 of 144 |
| cold | 104 of 144 | 107 of 144 | 65 of 144 |

"As run" is the adaptive call measured with the refitted model installed.
"Replayed" applies a model's choices to the same runs' medians of the two
forced schedules.

It still misses systematically in two places. On the D=2 bimodal sum and
max it runs block per segment, 34 to 37% slower than classification. On the
D=1 softmax with 200,000 segments it classifies, 8 to 10% slower. The
per-shape tables are in
[results/bench/README.md](results/bench/README.md#the-cost-model-on-held-out-shapes).

On the A10 vGPU the same model, given the A10's 6 MiB L2 by the runtime,
was within 3% for 126 of the 144 inputs warm and 123 cold as run; replayed
with the 4090's 72 MiB it would have been 120 and 113, and refitted on A10
runs of its 246 training inputs 138 and 134. The protocol was committed
before the A10 runs (`bench/suite.py`, `TRANSFER_FIT_GROUPS`); the per-input
table is [results/a10/transfer.txt](results/a10/transfer.txt).

### Speed on the 19 benchmark workloads (in-sample)

These are the workloads the cost model was fitted on: the refitted model on
all three seeds, the first model on seeds 711 and 712. So the *chose* column
below is in-sample. The ablations behind the optimizations ran on them too.

How the table was measured:

- **Time.** Complete-call GPU time in µs: CUDA events around the call, so
  host launch gaps count. Median over 15 processes (seeds 711 to 713 of the
  length distribution × 5 repetitions) of each process's median of 30 calls.
- **Inputs.** f32 unless noted. 1M rows in 16,384 segments at D=1; 262,144
  rows in 4,096 segments at D=128. Uniform (64 rows), long-tail (Pareto)
  and bursty lengths.
- **L2 state.** *Warm* is back-to-back calls. *Cold* overwrites a 288 MiB
  buffer and synchronizes before every call. *Graph* replays a CUDA graph
  captured from one call: the same kernels without the host's share of the
  call.
- **ragged.** The adaptive program, given the exact longest segment
  (`max_len`, the bound NJT also receives). *chose* is its schedule per seed
  711/712/713: C classification, N one block per segment. Fourteen
  workloads ran the first cost model. The five whose choice the refit
  changes were re-run with the refitted one.
- **Result.** A win when ragged takes less than 0.97 times the fastest
  comparison's time, a tie within 3%, a loss above.
- **Comparisons** ([baselines/README.md](baselines/README.md)). Triton
  autotuned on each input, CUB 3.5 (at D=128 on a transposed copy),
  cuSPARSE SpMM (sums), FBGEMM 1.9 (the D=1 sum and softmax),
  `torch.segment_reduce` (sum, max), compiled NJT with exact length bounds
  and a CUDA kernel with one block per segment.
- **Conditions.** Clocks are not locked (SM clock 2,520 to 2,805 MHz). The
  GPU and host were shared with other work.

Against the fastest comparison, ragged wins 4 workloads, ties 4 and loses 11
with a warm L2. With a cold L2 it wins 1, ties 3 and loses 15. On CUDA-graph
replay, which removes host overhead from every method, it wins 5, ties 4 and
loses 10.

![ragged's speedup over the fastest comparison per workload, warm and cold L2](docs/speedup.svg)

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

### Wins and losses

With a warm L2, ragged wins the uniform D=128 sum and max and the uniform
and long-tail D=1 softmax. It loses:

- the D=1 sum and max: 1.1 to 2.2 times slower warm, 1.1 to 3.1 times on
  graph replay;
- the long-tail D=128 sum, where cuSPARSE is 1.13 times faster;
- the bursty D=128 sum and max, by 1.07 times, where the cost model
  classifies although block per segment would tie;
- the bursty f32 and long-tail bf16 D=128 softmax, where tuned Triton is
  1.05 and 1.38 times faster.

### D=1 sum and max lose, and not only to host time

A warm call takes 33 to 35 µs. The kernel is 14.4 to 14.8 µs of it
([profile.txt](results/bench/profile.txt),
[burst-gap](results/bench/README.md#seed-711s-bursty-d1-sum)). Replayed as
a graph, the call takes 16 to 17 µs.

The comparisons gain as much from replay. `torch.segment_reduce` takes 11
µs and Triton 12 to 16. Compiled NJT takes 5.4 µs on uniform lengths, where
exact bounds turn it into a dense reduction.

So one block per segment over 16,384 short segments is slower than their
kernels, not only their launch path.

### cuSPARSE is the strongest comparison for sums

Its SpMM takes 22 to 23 µs at D=1 and 169 to 170 µs at D=128, for every
length distribution. That is 1.13 times faster than ragged on the long-tail
D=128 sum. On graph replay it is slower than ragged at D=1: 22 µs against
16 to 17.

CUB at D=128 needs a transpose and D·B segments; it takes 1.2 to 4.2 ms.
FBGEMM's D=1 sum takes 26 to 55 µs, and its `jagged_softmax` 74 µs to 2.9
ms.

### ragged wins where its mapping matters

On the uniform D=128 sum and max, persistent CTA workers take 134 to 136 µs
against 155 for `torch.segment_reduce`. On graph replay they take 102 µs
against 143 for compiled NJT.

On the D=1 softmax, the one-pass online statistics beat compiled NJT and
Triton by 14 to 20%.

### With a cold L2 it mostly loses, mostly not because of the cold L2

10 of the 15 cold losses are warm losses too: the five D=1 sum and max
losses (host time and short blocks), the long-tail and bursty D=128 sum,
the bursty D=128 max, and the bursty and bf16 D=128 softmax.

The uniform D=128 sum and max lose to compiled NJT, 230 against 187 µs.
Under exact uniform bounds NJT is a dense reduction. Classification's warm
lead there is the part of the 128 MiB input that back-to-back calls find in
the L2. Cold, even block per segment is 1.16 times slower than NJT, so this
loss is inherent to the schedules.

The cold L2 costs the D=1 long-tail and bursty softmax 15 and 8 µs, against
1 and 2 µs for Triton. This was not traced further.

On the uniform D=128 softmax, block per segment would tie cold (331 against
324 µs for Triton; classified takes 348) but is 3% slower warm. The model
does not know the L2 state.

Only the schedule choice was cheap to change. The refit turned the
long-tail D=128 max from a cold loss (1.03) into a tie. A margin against
classification gains less cold than it costs warm
([design-tradeoffs](docs/design-tradeoffs.md#choosing-classification-or-one-block-per-segment-per-call)).

The benchmark's flush leaves the L2 full of dirty lines, which every method
pays to write back ([results/bench/cold-l2.txt](results/bench/cold-l2.txt)).

### Classification against one block per segment

Forced classification loses to block per segment in 10 of 19 warm
workloads. The first cost model, fitted on these two shapes, removed those
losses here.

On other shapes it made two errors. With 65,536 segments at D=1, its
per-segment term, which had absorbed the fixed cost of a call, predicted
158 µs for classification; 57 µs was measured. On the uniform D=32 and D=64
sum and max, whose 64 MiB inputs stay in the L2, it charged DRAM time for
every element and classified, 35 to 41% slower
([profile.txt](results/bench/profile.txt): block per segment's kernel takes
16 µs there).

The refit, trained on those inputs too, gets both right. On these 19
workloads it classifies the bursty D=128 sum and max, 8% slower than block
per segment. N, B, D and the longest segment do not tell bursty from
long-tail lengths
([results/bench/README.md](results/bench/README.md#the-cost-model-on-held-out-shapes)).

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

The default target is sm_89. For sm_86 (an A10 or an RTX 30-series GPU),
give `ProgramSpec` `chip="sm_86"`, `bench.run` and `bench.suite`
`--chip sm_86`, and the GPU tests `RAGGED_TEST_CHIP=sm_86`; an sm_86 program
also runs on sm_89. A GPU machine without the MLIR toolchain runs programs
compiled elsewhere: `bench.precompile` compiles a suite's programs into a
cache and writes `toolchain.json`, the compiler identity that
`Compiler(toolchain_record=...)` keys the cache with instead of running the
compiler. How the A10 runs used it, and the cost model's transfer test:

```sh
python -m bench.precompile --chip sm_86 --groups heldout heldout2 compare --cache build/sm86-cache
python -m bench.suite --groups heldout2 --repetitions 3 --output build/a10-runs --chip sm_86 \
  --toolchain-record build/sm86-cache/toolchain.json --cache build/sm86-cache
RAGGED_TEST_CHIP=sm_86 RAGGED_TOOLCHAIN_RECORD=build/sm86-cache/toolchain.json \
  RAGGED_TEST_CACHE=build/sm86-cache PYTHONPATH=python:. python -m pytest test/gpu
python -m bench.calibrate transfer results/a10/runs --output build/cost_model-a10.json
```

The cache for the GPU tests is filled by one run of them with the compiler
and the same `RAGGED_TEST_CHIP` and `RAGGED_TEST_CACHE`.

## Limitations in detail

- Two targets, sm_86 and sm_89 (an sm_86 program also runs on sm_89),
  tested on an RTX 4090 and on one A10-24Q vGPU; one device per program;
  forward only (no autograd). On the vGPU there is no compute-sanitizer,
  CUPTI or clock locking, and its benchmarks ran once, in four jobs on a
  trial compute pool.
- Validated mode trusts the offsets; checked mode validates them on the device
  at the cost of a stream synchronization before the main kernels.
- The schedule choice needs the longest segment length from the caller
  (`max_len`); without it the program always classifies, which is 19 to 55%
  slower than one block per segment at D=1 (the `ragged, classified`
  column). Computing the statistics on the device would need a readback and a
  synchronization per call. A wrong `max_len` costs time, not correctness.
- The cost model is linear in six terms and calibrated on warm times of 82
  workloads in nine shapes (three length distributions) on the RTX 4090; its
  L2 term uses the L2 size of the device each call runs on. It sees N, B, D and the longest
  segment, not the distribution of lengths: it classifies the bursty D=128
  sum and max, 8% slower there than block per segment. On the second held-out
  set it picks the slower schedule by more than 3% for 32 of 144 inputs
  warm (40 cold), by 34 to 37% on the D=2 bimodal sum and max
  ([results/bench/README.md](results/bench/README.md#the-cost-model-on-held-out-shapes)).
  With a cold L2 it classifies the uniform D=128 inputs where block per
  segment is 5% faster. On the A10 vGPU it is within 3% for 126 of the 144
  warm (122 cold); the misses are the D=48 sum and max, classified 7 to 12%
  slower, the D=1 softmax with 200,000 segments (up to 18%) and some D=2
  inputs (3 to 8%). Refitted on A10 runs it is within 3% for 138 and 134, so
  a new GPU needs its own calibration runs for the last few percent
  ([results/a10](results/a10/README.md#the-cost-model-on-the-a10)).
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
- One RTX 4090 that also drives a desktop and is shared with other
  projects, unlocked clocks, and a host running other work; synthetic length
  distributions plus three seeds. The A10 runs are one vGPU on a trial
  compute pool, measured once, with the SM clock at 1,695 MHz and no
  temperature or power readings. The Triton reference is tuned on each
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
| `scripts/` | `bootstrap_llvm.sh` (LLVM/MLIR source build), `build_baselines.py` (the CUDA/CUB comparison library) and `compare_compilers.py` (whether two compiler builds emit the same sm_89 code) |
| `test/` | lit tests (`IR`, `Transforms`, `Conversion`), host tests, GPU tests |
| `docs/` | semantics, ABIs, optimizations, error model, design trade-offs |
| `results/` | benchmark runs and summary, cost-model calibration and evaluation, traces, test and sanitizer summaries; `results/a10/` the same for the A10 vGPU |
