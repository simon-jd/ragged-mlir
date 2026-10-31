# ragged-mlir: an MLIR compiler for segmented reductions on ragged tensors

Ragged (jagged) batches, a packed `[N, D]` tensor plus `B+1` segment offsets,
are how variable-length sequences, graph neighborhoods and recommendation
features reach a GPU. Reducing each segment (sum, max, softmax) is simple until
the lengths are skewed: one block per segment leaves most of the GPU idle on a
long segment, one warp per segment wastes it on short ones.

`ragged-mlir` is an out-of-tree compiler on MLIR 23.1.3 that generates CUDA
kernels (sm_89) for segmented sum, max and softmax and picks a mapping per
segment at run time. It has its own dialect and passes, lowers through the GPU
and NVVM dialects to PTX and `ptxas`, caches compiled artifacts, and launches
them through the CUDA Driver API on PyTorch streams.

```python
from ragged import ArtifactCache, Compiler, ProgramSpec

program = Compiler("build/bin/ragged-compile").compile(
    ProgramSpec("softmax", dtype="bf16", feature_dim=128, offsets_dtype="i64"),
    ArtifactCache("~/.cache/ragged"))
program.run(values, offsets, out)   # CUDA tensors; runs on the current stream
```

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
- **Scheduling.** Each persistent worker grid is one resident wave of that
  kernel, measured with the occupancy API when the module loads (on an RTX
  4090, 1,536 128-thread blocks or 3,072 warps, divided among feature tiles and
  split parts). The three classes touch disjoint segments, so their worker
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
  tiling (`d_tile`), then the standard GPU → NVVM → PTX pipeline and `ptxas`
  produce a cubin. Compensated f32 summation keeps bf16 and f32 sums within the
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
(`oracle/reference.py`). On an RTX 4090 (driver 595.99, CUDA 13.2):

| Suite | Result |
|---|---|
| GPU tests (`test/gpu`): every op, dtype, D in {1, 7, 16, 128, 1024}, i32/i64 offsets, four length distributions, empty segments, masks, in-place, checked offsets, streams, CUDA graph capture, Driver API faults; row pipelining and concurrent classes compared bitwise against the rolled, serial schedule; online and three-pass softmax in every class, including 65,536-row segments; class queues read back for counts, membership and per-warp order | 1,972 passed; 164 FBGEMM cases skip (optional package not installed) |
| compute-sanitizer memcheck, racecheck, initcheck and synccheck on the classified, mask, checked-offsets, scheduling and per-class softmax suites (595 tests; eight stream-ordering tests whose cuBLAS delay work takes hours under racecheck run under memcheck only) | 0 errors, 0 hazards |
| lit/FileCheck tests of the dialect, passes and lowering (`test/`) | 40 passed |
| host tests (`test/python`): cache, manifest validation, runtime contracts and stream schedule, benchmark harness | 1,693 passed |

**Speed.** Complete-call GPU time in microseconds (CUDA events around the
call, so host launch gaps count; median of 30 calls after 5 warm-up calls), f32,
1M rows in 16,384 segments for D=1 and 262,144 rows in 4,096 segments for
D=128 (`bench/run.py`, `results/bench/`). Each method passes the oracle before
it is timed. `ragged` is the default path with device classification;
`ragged-naive` is the same compiler with one block per segment
(`strategy="naive"`). None of the comparisons is tuned for these inputs
([baselines/README.md](baselines/README.md)):

- `torch.segment_reduce` from PyTorch 2.14 (sum and max);
- a handwritten Triton kernel: one program per segment and feature tile,
  fixed 32-row tiles, no autotuning, and three kernels for softmax;
- CUB 3.5 `DeviceSegmentedReduce` for sum and max at D=1; its softmax is two
  CUB reductions (max, sum) plus two one-block-per-segment kernels (exp,
  normalize);
- a CUDA kernel with one block per segment whose threads loop over feature
  columns, so at D=1 one thread of each block does all the work.

"Earlier version" is this compiler before row pipelining, device-sized worker
grids, concurrent classes, native launch plans, online softmax statistics in
every class and warp-aggregated classification, measured on the same inputs in
an earlier run whose records are not part of `results/`. Each case is a single
run with one seed. The GPU also drives a desktop; a case in which any method's
p90 exceeds its median by more than 25% is re-run (none in this run). L2 is not
flushed between timed calls: the D=1 inputs (4 MiB) fit in the 4090's 72 MiB
L2, and part of a 128 MiB D=128 input can stay there from one call to the next.

| Op | D | Segment lengths | ragged | ragged-naive | earlier version | torch.segment_reduce | Triton | CUB | CUDA, one block per segment |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| sum | 1 | uniform, 64–64 | 48 | 33 | 99 | 15 | 28 | 29 | 26 |
| sum | 1 | long tail, 11–300,412 | 45 | 83 | 130 | 5,572 | 348 | 46 | 5,957 |
| sum | 1 | bursty, 0–6,394 | 39 | 32 | 84 | 229 | 30 | 25 | 151 |
| sum | 128 | uniform, 64–64 | 138 | 166 | 177 | 156 | 167 | – | 159 |
| sum | 128 | long tail, 16–9,570 | 194 | 245 | 510 | 722 | 236 | – | 1,094 |
| sum | 128 | bursty, 0–6,513 | 179 | 169 | 202 | 481 | 171 | – | 733 |
| max | 1 | uniform, 64–64 | 49 | 33 | 98 | 15 | 28 | 30 | 26 |
| max | 1 | long tail, 11–300,412 | 42 | 54 | 129 | 3,943 | 303 | 51 | 5,076 |
| max | 1 | bursty, 0–6,394 | 40 | 32 | 83 | 159 | 31 | 26 | 130 |
| max | 128 | uniform, 64–64 | 139 | 166 | 170 | 156 | 167 | – | 160 |
| max | 128 | long tail, 16–9,570 | 185 | 198 | 498 | 342 | 207 | – | 927 |
| max | 128 | bursty, 0–6,513 | 180 | 169 | 202 | 266 | 167 | – | 617 |
| softmax | 1 | uniform, 64–64 | 55 | 35 | 189 | – | 53 | 72 | 78 |
| softmax | 1 | long tail, 11–300,412 | 83 | 266 | 315 | – | 1,860 | 679 | 36,403 |
| softmax | 1 | bursty, 0–6,394 | 48 | 35 | 160 | – | 80 | 64 | 809 |
| softmax | 128 | uniform, 64–64 | 313 | 335 | 672 | – | 626 | – | 399 |
| softmax | 128 | long tail, 16–9,570 | 438 | 604 | 1,444 | – | 846 | – | 4,138 |
| softmax | 128 | bursty, 0–6,513 | 506 | 556 | 681 | – | 646 | – | 4,413 |

bf16 softmax at D=128 on the long tail: ragged 264 µs, ragged-naive 472,
earlier version 1,164, Triton 472.

What the numbers say:

- **On the long-tail lengths ragged is the fastest method in every case.** At
  D=1, sum 45 µs and max 42, against 46 and 51 for CUB, 300–350 for Triton
  and 4 to 6 ms for `torch.segment_reduce` and the CUDA block-per-segment
  kernel, which serialize the 300,412-row segment; softmax takes 83 µs
  against 679 for the CUB composition and 1,860 for Triton. At D=128, sum 194
  vs 236 µs for Triton, max 185 vs 207, softmax 438 vs 846 (bf16 264 vs 472),
  and the sum is almost four times faster than `torch.segment_reduce` (722).
- **Where ragged loses.** Device classification has a fixed cost on every
  call: a sum runs six kernels, two of them (init and classify) for
  classification, plus a stream fork and join, and the D=1 uniform sum spends
  22 µs on the host (`results/bench/host.txt`). The compiler's own
  block-per-segment mode, `ragged-naive`, is faster than `ragged` in 8 of the
  19 cases: all six uniform and bursty cases at D=1 (for example sum 33 vs
  48 µs on uniform lengths, softmax 35 vs 48 on bursty ones) and the bursty
  sum and max at D=128 (169 vs 179 and 169 vs 180). The fastest comparison
  method beats `ragged` in 7 of the 19: `torch.segment_reduce` on the uniform
  D=1 sum and max (15 µs against 48 and 49), CUB on the bursty D=1 sum and max
  (25 and 26 against 39 and 40), Triton on the bursty D=128 sum and max (171
  and 167 against 179 and 180) and on the uniform D=1 softmax (53 against 55).
  Without length hints the default path classified every case
  (`classes` in `results/bench/summary.json`); nothing picks the
  block-per-segment mode from the data.
- **Length hints help at D=1 and hurt at D=128.** Given exact length bounds
  (`--length-hints`), the uniform case compiles to one block-per-segment
  kernel: 32 µs instead of 48 for the D=1 sum and 35 instead of 55 for the D=1
  softmax, but 166 instead of 138 for the D=128 sum and 335 instead of 313
  for the D=128 softmax. On the long-tail lengths the bounds prune no class
  (45.6 vs 45.1 µs at D=1, 194.5 vs 193.5 at D=128).
- **Softmax at D=128 runs near the bandwidth of its two passes.** With uniform
  lengths every 64-row segment falls in the CTA class, whose online kernel
  takes 283 µs (`results/bench/profile.txt`) to read and write 128 MiB each:
  0.95 TB/s against the 4090's 1.01 TB/s DRAM bandwidth, which is only
  possible because most of its second read of each segment (32 KiB) hits in
  cache. The call (313 µs) beats block-per-segment (399) and Triton (626). On
  the bursty lengths 99% of the rows are in 40 split segments of about 6,500
  rows; the call's GPU span is 492 µs, against 400 µs to read the data twice
  and write it once at the DRAM bandwidth, and it beats Triton (646).
- **With uniform lengths at D=128, sum and max are fastest too** (138 and 139
  µs, against 156 for `torch.segment_reduce`, 159 and 160 for the CUDA kernel
  and 167 for Triton), close to the 133 µs it takes to read 128 MiB at the
  DRAM bandwidth. The fastest of the 30 sum calls took 127 µs, below that
  bound, so part of the input came from L2.
- NJT (`torch.nested`) is not competitive as configured here. It is built
  without `max_seqlen` (`bench/run.py` passes the bounds only with
  `--length-hints`, and the hinted runs did not include NJT), so eager sum
  and max pad every segment to the full row count and run out of memory
  (requests of 64 and 512 GiB), and eager softmax fails on an int32 index
  limit in four of seven cases; `torch.compile(fullgraph=True)` takes 7 to
  29 ms per call for sum and max and fails to compile softmax.

**Where the speedup comes from.** Ablations, ragged only, µs
(`results/bench/ablation/`), each turning off one change in the same build:
`--disable-optimization online_fusion` (three passes in every class),
`--disable-optimization row_pipelining` and `--serial-classes`:

| Workload | ragged | three passes in every class | rows not pipelined | classes serial | earlier version |
|---|---:|---:|---:|---:|---:|
| sum, D=1, uniform | 48 | – | 49 | 54 | 99 |
| sum, D=1, long tail | 45 | – | 49 | 53 | 130 |
| sum, D=1, bursty | 39 | – | 41 | 42 | 84 |
| sum, D=128, uniform | 138 | – | 135 | 139 | 177 |
| sum, D=128, long tail | 194 | – | 215 | 215 | 510 |
| sum, D=128, bursty | 179 | – | 181 | 183 | 202 |
| max, D=1, uniform | 49 | – | 47 | 55 | 98 |
| max, D=1, long tail | 42 | – | 47 | 52 | 129 |
| max, D=1, bursty | 40 | – | 40 | 41 | 83 |
| max, D=128, uniform | 139 | – | 131 | 149 | 170 |
| max, D=128, long tail | 185 | – | 212 | 203 | 498 |
| max, D=128, bursty | 180 | – | 180 | 183 | 202 |
| softmax, D=1, uniform | 55 | 80 | 53 | 60 | 189 |
| softmax, D=1, long tail | 83 | 89 | 98 | 100 | 315 |
| softmax, D=1, bursty | 48 | 59 | 48 | 51 | 160 |
| softmax, D=128, uniform | 313 | 634 | 312 | 326 | 672 |
| softmax, D=128, long tail | 438 | 651 | 600 | 487 | 1,444 |
| softmax, D=128, bursty | 506 | 666 | 494 | 512 | 681 |
| softmax bf16, D=128, long tail | 264 | 348 | – | – | 1,164 |

Online statistics are faster in every softmax case, and concurrent classes in
all 18 f32 cases. Row pipelining helps most on the long tail (215 vs 194 µs for the
D=128 sum, 600 vs 438 for the D=128 softmax) and is slower in six cases:
max with uniform lengths by 6% at D=128 (139 vs 131 µs) and 5% at D=1 (49 vs
47), the uniform D=1 softmax by 4%, the uniform D=128 sum and the bursty D=128
softmax by 3%, and the uniform D=128 softmax by less than 1%. With one run per
case, differences of a few percent may be noise.

Kernel traces (`results/bench/profile.txt`, `python -m bench.profile`) show
where the time goes. With the classes run one after another, the D=128
long-tail softmax's CTA class takes 316 µs in its one online kernel and its
warp class 64 µs; run concurrently, the call's GPU span drops from 471 to
420 µs although each kernel takes longer while they share the GPU. The
classify kernel takes 7 µs at D=1, and the D=1 long-tail sum's GPU span is
27 µs of a 45 µs call. The D=128 long-tail sum's CTA worker takes 155 µs with
the classes serial, most of that call's 194 µs span.

Launching every kernel and the stream fork and join from one native call
cuts the host time of a call (each call started on an idle GPU,
`python -m bench.profile host`, `results/bench/host.txt`, which compares the
native launch plans with an earlier runtime that launched each kernel from
Python): 23 µs instead of 64 for a sum at D=1 (six kernels) and 30 µs instead
of 99 for the D=128 long-tail softmax when it launched 13 kernels; with online
statistics in every class it launches 7 and takes 24 µs. The stream fork and
join add 1 to 2 µs instead of 13. Because CUDA events count host launch gaps,
host time shows in the GPU times too, at D=128 as well, where a kernel
otherwise waits for the host to enqueue it. Two other differences from the
earlier version have no switch, so the ablations do not isolate them:
device-sized worker grids (the earlier version capped worker grids at 256
blocks) and compiling out the feature guard of full tiles.

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
above writes a new one under `build/` that can be compared with
`results/bench/softmax-bf16-d128-long_tail/results.json`. The `cuda-naive` and
`cub` methods need the comparison library that `baselines/README.md` shows how
to build. `python -m bench.matrix` plans and runs whole matrices.

## Limitations

- sm_89 only; one device per program; forward only (no autograd).
- Validated mode trusts the offsets; checked mode validates them on the device
  at the cost of a stream synchronization before the main kernels.
- Segment-class bounds (32 and 4,096 rows) and the 8-way split are fixed in the
  compiler and the classified ABI, not tuned per GPU or per D.
- At small sizes the host still costs about 20 µs per call, mostly Python:
  tensor validation, descriptor construction and the scratch allocation.
- The online fold is one dependency chain per lane. When few segments are
  long, as for the split class of the long-tail inputs, its kernel is bound by
  latency rather than bandwidth. Recentering once per batch of eight pipelined
  rows would shorten the chain, but would change results against the rolled
  loop, which row pipelining now reproduces bit for bit.
- The default path does not choose between device classification and one
  block per segment; `ragged-naive` is faster on 8 of the 19 benchmark cases.
- Persistent workers take queue positions with a fixed stride, so their speed
  depends on the order classification leaves in the queues and on which SM
  each worker lands. In separate A/B runs (not part of `results/`), the
  warp-ordered queues made the long-tail D=128 sum and max about 6% slower
  than per-segment atomics, and submitting the split chain before the CTA
  chain made the uniform D=128 sum take 176 µs instead of 133
  ([docs/design-tradeoffs.md](docs/design-tradeoffs.md)).
- The benchmark is one run per case with one seed, on one GPU, without
  flushing L2 between calls, against comparison implementations that are not
  tuned.

## Layout

| Path | What it is |
|---|---|
| `include/Ragged/`, `lib/IR/` | the dialect, ops and verifiers |
| `lib/Transforms/` | legalization and mapping selection |
| `lib/Conversion/ToGPU.cpp` | lowering to GPU kernels |
| `tools/` | `ragged-opt` and `ragged-compile` |
| `python/ragged/` | frontend, artifact cache, manifest validation, runtime, PyTorch op registration |
| `runtime/` | CUDA Driver API module loader and launcher (nanobind) |
| `oracle/` | float64 reference and test-case minimizer |
| `baselines/`, `bench/` | comparison implementations and the benchmark harness |
| `test/` | lit tests (`IR`, `Transforms`, `Conversion`), host tests, GPU tests |
| `docs/` | semantics, ABIs, optimizations, error model |
