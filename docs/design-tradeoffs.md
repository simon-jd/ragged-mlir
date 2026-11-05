# Design trade-offs

The compiler makes a handful of structural choices: what is compiled, how
segments are mapped to warps and blocks, how the class queues are built and
launched, and how softmax keeps its statistics. This document records, for
each, what else could have been done and what the choice gives up, with the
measurements behind it.

Numbers are complete-call microseconds on an RTX 4090 from `results/bench/`:
medians over 15 runs with a warm L2 from `summary.json` unless a sentence
says otherwise, plus `profile.txt`, `host.txt`, `cold-l2.txt` and
`calibration/`. "Classified" is the adaptive program with classification
forced (the `ragged, classified` column). A few comparisons come
from separate A/B runs whose records are not in `results/`; those are marked
"A/B". "D" is the feature dimension, "long tail" the
Pareto-distributed segment lengths of the benchmark, and `ragged-naive` the
compiler's own one-block-per-segment mode.

## A compiler rather than a kernel library

**Decision.** Express segmented sum, max and softmax as MLIR operations and
compile one specialization per (operation, dtype, D, offset type, length
hints, optimization controls), cached by a key over the source IR and the
toolchain.

**Alternatives.** A hand-written CUDA library with templates over the same
parameters; a Triton kernel per operation.

**Why.** The choices that matter are made per program and are easier to
reason about as passes than as template flags: which segment classes exist
(`ragged-plan`), whether softmax statistics may be fused (`ragged-fuse`, which
rejects a forced fusion that would be illegal instead of silently falling
back), and how features are tiled. The verifier rejects ill-formed programs
before any kernel exists, and every artifact carries a manifest the runtime
checks before it launches anything.

**Cost.** An LLVM/MLIR 23 build dependency, compile time on first use of a
specialization, and an ABI between the compiler and the runtime that has to
be versioned (`classified-v3`) and validated.

## Three segment classes, chosen on the device

**Decision.** Segments up to 32 rows go to a warp, up to 4,096 rows to a
128-thread block, and longer ones to eight blocks that write partial results
for a combine kernel. A classify kernel bins segments into per-class queues
on the device; length hints prune classes at compile time.

**Alternatives.** One block per segment (the `naive` strategy, kept as
`ragged-naive`); sorting segments by length; binning on the host.

**Why.** One block per segment puts a long segment on one SM. With the
300,412-row segment of seed 711, `ragged-naive` takes 84 µs for the D=1 sum
and 264 µs for the D=1 softmax, against 47 and 81 µs classified
(`calibration/refit-training.txt`, main runs); the D=128 long-tail softmax takes 485 µs
naive and 404 µs classified. Persistent CTA workers also beat many short
blocks: on the uniform D=128 sum the 16,384 blocks of 64 rows take 149 µs,
the CTA workers 101 µs (`profile.txt`). Sorting costs a full sort per call.
Host binning needs the offsets on the host, which means a device-to-host copy
and a synchronization in the middle of the call.

**Cost.** Two prologue kernels (init and classify), three class chains, each
launched even when its queue is empty, and a stream fork and join on every
call. That cost is not always repaid: one block per segment is faster in 10
of the 19 workloads (14 cold), by 19 to 55% on every D=1 workload but the
long-tail softmax and by 7% on the bursty D=128 sum and max. On the uniform
D=1 sum the classified GPU span is 30 µs (classify 6.5, CTA workers 18.7,
the empty warp and split chains 2.7 to 2.8 µs each) against one 14.8 µs
kernel, and the host takes 24 µs against 14 (`host.txt`). On the bursty D=128
sum the 40 long segments already fill 160 blocks of one block per segment
(149 µs), so splitting them (147 µs) buys nothing and the prologue and the
warp chain beside it are extra. The next section is how the runtime avoids
paying this where it does not help. The thresholds are fixed, not tuned per
GPU or per D.

## Choosing classification or one block per segment per call

**Decision.** A program that classifies also compiles its block-per-segment
specialization. Before each call, a linear cost model predicts both
complete-call times from N, B, D, the element size and the longest segment
L, and the call runs the faster schedule. The terms are a fixed cost, the
elements N·D, the elements each thread of the block that owns the longest
segment reads (L × features per tile / 128, the critical path of block per
segment), the segment count, the elements weighted by log2 of the skew
L/(N/B), and the MiB of the working set (input rows plus output) beyond the
72 MiB L2. Nonnegative coefficients per operation family (sum and max;
softmax) and schedule are fitted to warm times of 246 inputs in nine
shapes: the 19 benchmark workloads and the 63 of the first held-out set,
seeds 711 to 713 (`bench/calibrate.py`, `calibration/fit.txt`). The margin
by which classification must win is chosen by leave-one-shape-out
cross-validation and came out as zero. L comes from the caller
(`run(max_len=...)`); without it the program classifies.

**Alternatives.** Always classify (the previous behavior, still the
`ragged, classified` column); compute the statistics on the device and read
them back; cache statistics per offsets tensor; compile one kernel that
decides on the device and lets the other schedule's kernels exit early.

**Why.** The decision needs the longest segment, and the caller often has
it: a jagged tensor such as NJT carries its maximum sequence length, and the
benchmark gives NJT exactly that bound. With it the choice is a few
multiplications on the host. A device readback would add a synchronization
to every call (and cannot run during CUDA graph capture), a per-tensor cache
would act on stale statistics when offsets change in place, and launching
both schedules' kernels would add the other schedule's launches to every
call.

**The first fit and its two misses.** Version 1
(`calibration/cost_model-v1.json`, `calibration/fit-v1.txt`) was fitted on
seeds 711 and 712 of the benchmark workloads alone: two shapes, D=1 with
16,384 segments and D=128 with 4,096. Its sixth term was the row bytes,
which for f32 is the element count again. It stayed within 1.4% of
`ragged-naive` on those shapes but was within 3% of the faster schedule for
only 115 of 189 inputs on seven held-out shapes (`heldout-shapes.txt`),
with two systematic misses that the traces in `profile.txt` explain:

- *D=1 with 65,536 segments of 16 rows on average: it ran block per segment,
  12 to 115% slower.* In the training data the segment count moved with D,
  so the fit could not tell a per-segment cost from what else separates a
  D=1 call from a D=128 one. Classification's fit had no constant, and its
  per-segment term (2.34 µs per thousand segments) carried the fixed cost of
  a classified call, 24 µs of host time for six launches and a stream fork
  (`host.txt`) plus the init and classify kernels. At four times the
  segments it predicted 158 µs for classification and 112 for block per
  segment; they take 57 and 71. What grows with the segment count is block
  per segment's kernel, one 128-thread block per segment (12 resident per
  SM, so 65,536 blocks are 43 waves on 128 SMs): 53.6 µs here against 14.8
  µs for 16,384 segments of 64 rows, while the classified span stays at 30
  µs (the warp class takes 16-row segments, a warp each).
- *Uniform D=32 and D=64 sum and max: it classified, 35 to 41% slower.*
  Both per-element coefficients came from the D=128 inputs, 128 MiB that do
  not fit the L2 (block per segment's kernel reads them at 0.9 TB/s), where
  classification's persistent workers read faster (1.0 against 1.2 µs per
  2^18 elements). The 64 MiB D=32 and D=64 inputs stay in the L2 between
  calls: block per segment's kernel takes 16 µs, the classified span 25 to
  27 µs, and the call is decided by classification's fixed cost (48 against
  35 µs), not by the bytes. The model had no term that knew the L2 size.

Refitted on those shapes as well, with the bytes beyond the L2 as the sixth
term (version 2), the fixed costs separate from the per-segment ones (36
µs and 0.31 µs per thousand segments for classified sums and maxes, 21 µs
and 0.72 µs for block per segment), and elements that stay in the L2 cost
0.11 and 0.16 µs per 2^18 against 1.8 and 2.0 µs per MiB beyond it.
In-sample it is within 3% for 160 of the 189 first held-out inputs and 50
of the 57 benchmark ones (`refit-training.txt`).

**Evaluation.** Since the first held-out inputs had informed the change, a
second held-out set was fixed and committed before the refitted model ran
(`HELDOUT2_SHAPES`): 8 shapes, segment and row counts no earlier group
used, D from 1 to 512, bimodal and exponential lengths, 144 inputs. Run
with version 2 and evaluated once (`calibration/heldout2.txt`), the
adaptive call is within 3% of the faster schedule for 112 of 144 inputs
warm and 104 cold; replayed on the same runs, version 1's choices are
within 3% for 61 and 65. Version 2 still misses systematically on the D=2
bimodal sum and max (block per segment, 34 to 37% slower), the D=1 softmax
with 200,000 segments (classification, 8 to 10%) and D=192 with 768
segments (classification, 4 to 5%).

**Cost.** A second compile per specialization and a second module per
program. The model sees N, B, D and the longest segment, not the
distribution of lengths, so bursty and long-tail lengths of the same shape
look alike: refitted, it classifies every seed of the bursty D=128 sum and
max, 8% slower there than block per segment (181 against 168 µs), and those
two benchmark workloads went from ties to losses against the fastest
comparison. It is calibrated on warm times on one GPU: with a cold L2 it
classifies the uniform D=128 sum, max and softmax, where block per segment
is 5% faster. A margin against
classification would move cold choices the right way but costs more warm:
over the left-out shapes the mean excess over the faster schedule is 2.66%
warm and 2.06% cold at no margin, and 3.21% and 1.84% at 2%, the best cold
(`fit.txt`). Without `max_len` the program pays classification's cost
everywhere. A wrong `max_len` costs time, never correctness, because both
schedules are exact against the oracle.

**On a second GPU.** The sixth term counts the bytes beyond the L2 of the
device a call runs on (the runtime reads its size once per device); the
model stores the 4090's 72 MiB only as what it was fitted with. On one
A10-24Q vGPU (sm_86, 72 multiprocessors, 6 MiB of L2) the same 144 inputs of
the second held-out set were measured, under a protocol committed before the
runs (`bench/suite.py`, `TRANSFER_FIT_GROUPS`; `results/a10/transfer.txt`).
Replayed on those runs, the shipped model with the 4090's L2 is within 3% of
the faster schedule for 120 inputs warm and 113 cold; with the A10's L2,
126 and 122 (126 and 123 as run). The L2 swap fixes the D=2 bimodal sum and
max and the D=16 softmax but classifies the D=48 sum and max, 7 to 12%
slower than block per segment (every one of these inputs exceeds a 6 MiB
L2). Refitted on A10 runs of the 246 training inputs, the model is within 3%
for 138 and 134. Its fixed costs are twice the 4090's (74 and 46 µs for a
classified and a block-per-segment sum or max, against 36 and 21). The D=1 softmax
with 200,000 segments stays a miss on both GPUs and in both fits.

## Persistent workers with a static stride

**Decision.** Each worker kernel launches one resident wave, sized at module
load with `cuOccupancyMaxActiveBlocksPerMultiprocessor` (up to 1,536
128-thread blocks or 3,072 warps on a 4090, fewer for kernels with more
registers, divided among feature tiles and split parts). Worker `x` processes queue positions `x, x + grid, x + 2·grid, …`.

**Alternatives.** A grid of one block per queue entry, which needs the queue
length on the host; a fixed cap (an earlier version used `min(B, 256)`); or
fetching the next position with an atomic counter.

**Why.** The grid does not depend on device-side counts, so there is no
readback, and one wave keeps every SM busy; the 256-block cap had left most of
the GPU idle at D=1. A per-item atomic would put back the same-address
contention that the classify section below removes.

**Cost.** Static assignment cannot rebalance. Long-tail CTA segments range
from 33 to 4,096 rows, so some workers finish late, and the kernel's speed
depends on which queue positions land on which SM, which the next two
sections run into.

## Building the class queues

**Decision.** Each warp ranks its lanes within their classes with one
shuffle scan over three packed 8-bit counts; lanes 0–2 then add the warp's
count for one class each to that class's counter. That is at most three
atomics per warp, and a warp's segments of one class occupy consecutive
queue positions in segment order.

**Alternatives.** One atomic per segment (an earlier version); aggregating
per 128-thread block through shared memory.

**Why.** Atomics on the same address serialize in L2. In the A/B runs, one
atomic per segment (16,384 i64 atomics on three counters) made classify the
largest kernel at D=1 (14.4 µs); warp aggregation brought it to 8.0 µs and
block aggregation to 6.6 µs, but block aggregation's 128-segment runs made the
CTA workers slower: the uniform D=128 sum with the classes run serially took
149 µs with block runs, 143 µs with per-segment atomics and 133 µs with warp
runs. Over the 18-case matrix, warp aggregation (with the submission order
below) was 8% faster than per-segment atomics in geometric mean and block
aggregation 4.5% (A/B). In the committed kernel trace the warp-aggregated
classify takes 6.6 µs on the D=1 long-tail sum.

**Cost.** In the A/B runs the long-tail sum and max at D=128 were 5 to 6%
slower than with per-segment atomics, in every aggregated variant: with the
classes serial, the CTA sum worker took 155 µs instead of 129. It came from
the queue order, not the submission order. The trace of the current build
(`profile.txt`) has that worker at 123 µs with warp-ordered queues, so the
cost may no longer apply; it has not been measured again against
per-segment atomics.

## Running the classes concurrently, CTA chain first

**Decision.** After classify, the runtime records an event, launches the CTA
chain and then the split chain on two side streams that wait for it, the
warp chain on the caller's stream, and joins at the end. The classes touch
disjoint segments, so no other ordering is needed.

**Alternatives.** Run the chains one after another on the caller's stream
(kept as `run(concurrent=False)`); submit the longest-running chain first.

**Why.** The chains of a long-tail input overlap: the D=128 long-tail sum
takes 191 µs concurrent and 202 µs serial, and concurrent classes are faster
in all 18 f32 workloads of the ablation. The submission order was measured, not
reasoned out: with the split chain first, the uniform D=128 sum took 176 µs,
and with the CTA chain first 133 µs (A/B, medians of two matrix runs). In a
separate measurement, `CUDA_DEVICE_MAX_CONNECTIONS=1`, which serializes
dispatch across streams, removed most of the split-first penalty. So it comes
from dispatching the split chain's first kernel, a full resident wave, beside
the CTA workers from another hardware queue. CTA first avoids it.

**Cost.** On the D=1 long-tail inputs, where the split chain is the longest,
CTA first was about 2 µs slower (A/B). The effect depends on hardware
dispatch and could change on another GPU or driver.

## Launching from C++ with per-shape plans

**Decision.** The native module keeps a launch plan per (N, B, concurrent):
kernel handles, grids, stream lanes and a template of the scratch
descriptors. One native call relocates the descriptors to this call's
scratch, issues every launch and the fork and join, and returns. All
workspace is one 256-byte-aligned allocation.

**Alternatives.** Launch each kernel from Python; capture a CUDA graph per
shape.

**Why.** Against an earlier runtime that launched each kernel from Python
(A/B), host time per call dropped from 64 to 23 µs for a D=1 sum, and from 99
to 30 µs for a D=128 softmax that then had 13 kernels; the current build
takes 24 and 26 µs, and a block-per-segment call 14 and 15 µs (`host.txt`). Because the benchmark's CUDA events count the
gaps in which the GPU waits for the host, that is visible in end-to-end
times too. A graph would need its kernel parameters updated for every call's
tensors; it was not tried, because the launch plan already brought host time
to about 25 µs.

**Cost.** Plans are cached per shape (cleared at 256), and the native code
has to join every stream it forked even when a launch fails in the middle,
which it does before raising.

## Softmax statistics: online in every class

**Decision.** Compute the maximum and the exponential sum in one pass, then
normalize in a second pass over the same rows, in one kernel. The running
reference trails the maximum by at most one: it moves only when a value
exceeds it by more than 1.0, so every mass is at most e. Masses accumulate
with compensated (Kahan) addition, and on a move both the sum and its
correction are rescaled.

**Alternatives.** Three passes (maximum, exponential sum, normalize), each a
kernel; the textbook online softmax that rescales at every new maximum.

**Why.** It reads the rows twice instead of three times and launches one
kernel instead of three: with three passes in every class the D=128
long-tail softmax takes 631 µs and the uniform one 624 µs, with online
statistics 404 and 310 µs, and the uniform case's CTA kernel moves its data at
0.96 TB/s against the 4090's 1.01 TB/s DRAM bandwidth (warm L2). Rescaling at
every new maximum multiplies the sum by many factors close to one and loses
accuracy on slowly increasing inputs; a lagging reference does not, and GPU
tests with such ramps hold it to the fixed tolerance. With compensated lanes the error does not grow with segment length, so
the online schedule needs no length bound; the tests run 65,536-row segments.

**Cost.** The fold is one dependency chain per lane: the reference update,
two exponentials and the compensated add. When few segments are long, as in
the long tail's split class, the kernel waits on latency rather than memory.
In-place softmax keeps three passes, because reads must finish at a kernel
boundary before writes begin.

## Split segments: per-part statistics and a combine kernel

**Decision.** Each of a split segment's eight blocks computes online
statistics for its part; a combine kernel takes the largest of the eight
references and the compensated sum of the masses rescaled to it; a third
kernel normalizes.

**Alternatives.** Three-pass split softmax (five kernels: partial maxima,
their combine, partial exponential sums against the combined maximum, their
combine, normalization); one cooperative kernel with a grid-wide barrier.

**Why.** Three kernels and two reads instead of five and three: the bursty
D=128 softmax, almost all split segments, takes 651 µs with three passes in
every class and 506 µs with online statistics (tuned Triton 479). A grid-wide
barrier would need a cooperative launch with every block
resident, which cannot share the GPU with the other classes' workers.

**Cost.** On the long tail the per-part kernel is bound by the latency of the
fold: with the classes serial it takes 39 µs. Run concurrently, it runs
beside the CTA chain, which takes longer (`profile.txt`), so the call does not
wait on it.

## Row pipelining without reassociation

**Decision.** Each lane's row loop issues eight loads, then folds them in row
order; a rolled loop covers the remainder. When the feature tile divides D,
the bounds guard is compiled out, so the loads carry no branch.

**Alternatives.** The rolled loop; unrolling with eight independent
accumulators combined at the end.

**Why.** Eight loads in flight per lane instead of one: the D=128 long-tail
max takes 180 µs against 189 µs without, the softmax 404 against 493. With
vector access a block has four times the row lanes, so the long-tail sum
gains less (191 against 192).
Folding in row order keeps every floating-point operation in the rolled
loop's order, so results are bitwise identical with the control on or off,
which a GPU test checks. Independent accumulators would shorten the
dependency chain further but change results with the unroll factor.

**Cost.** The online fold's chain is not shortened. Recentering once per
batch of eight rows would shorten it, at the price of the bitwise guarantee.
Pipelining is not faster everywhere: in three of the 18 f32 workloads it is
slower, by 0.1 to 3% (uniform D=1 softmax, 55 against 53 µs).

## Vector access that keeps the feature tiles

**Decision.** Each thread owns `d_vector` adjacent features, the largest
power of two dividing D that fits 16 bytes (four f32, eight bf16), and reads
and writes them with one aligned vector access per row (`ld.global.v4` for
f32). A feature tile still spans min(32, next power of two of D) features,
now covered by 32 / `d_vector` threads, so a 128-thread block has
`d_vector` times as many row lanes; the runtime rejects tensors whose base is
not aligned to the access.

**Alternatives.** One feature per thread (`vectorize=False`); keeping 32
threads per tile, so that one tile spans 128 f32 features and D=128 needs
one tile instead of four.

**Why.** Fewer, wider load instructions and more rows in flight per block.
The gain shows where one block owns a long segment: block per segment takes
196 µs instead of 236 on the long-tail D=128 sum, 485 instead of 572 on the
long-tail softmax and 506 instead of 534 on the bursty softmax, and the
classified long-tail softmax 404 instead of 437; elsewhere both schedules
are within 4% of one feature per thread (no-vectorize ablation). Keeping 32
threads per tile cut the blocks per segment by four, and in A/B runs on seed
712 it was slower where few blocks carry the work: block per segment took
209 µs instead of 168 on the bursty D=128 sum and 837 instead of 506 on the
bursty softmax. With the tile width kept, the per-feature fold and tree are
those of the scalar mapping with the same tile in threads, so a GPU test
compares the vector kernels bitwise against `vectorize=False,
feature_tile=8`.

**Cost.** Registers (the ptxas counts in each run's report): the f32 online
softmax CTA kernel uses 72 instead of 48, so its workers fit 7 instead of 10
blocks per SM (896 instead of 1,280 at D=128). With eight bf16 features per
thread it uses 159 registers instead of 45 and fits 3 blocks per SM, and the bf16
long-tail softmax is slower than with one feature per thread: 302 µs
classified against 264, and block per segment 459 against 442. Capping bf16
at four features per thread was not tried. It also does not help the cold-L2 uniform D=128
sum (229 µs, 227 scalar): most of that call's gap to DRAM bandwidth is the
write-back of the dirty lines the benchmark's flush leaves in the L2. After
a flush that only reads, block per segment takes 163 µs instead of 214 and
`torch.segment_reduce` 152 instead of 206 (`cold-l2.txt`). Tensors that
start mid-row (a view at an odd element offset) need a `vectorize=False`
specialization.

## Compensated f32 instead of f64

**Decision.** Accumulate sums and exponential sums in f32 with Kahan
compensation, for f32 and bf16 inputs alike.

**Alternatives.** Plain f32; f64 accumulation; pairwise trees within a lane.

**Why.** With four row lanes per block, a plain f32 accumulator fails the
float64 oracle's fixed tolerance on a 16,384-row segment of 0.1s. f64 runs at
1/64 of f32 throughput on this GPU. Compensation costs four extra f32
operations per element in kernels that are bound by memory.

**Cost.** Not isolated by any committed measurement; the arithmetic is more
involved to read, and the correction must be cleared for non-finite sums so that
`inf - inf` does not turn a valid infinity into NaN.

## Strict manifests and a float64 oracle

**Decision.** The runtime validates every manifest field against the source
specification (kernel roles in order, workspace names, shapes and dtypes,
grid symbols) before loading a module, and every benchmarked method must
match a float64 NumPy oracle before it is timed.

**Alternatives.** Trust the compiler's output; compare against PyTorch.

**Why.** A stale artifact or a compiler change that alters the schedule fails
before a kernel runs, not as wrong numbers. The oracle has fixed,
dtype-specific tolerances that no optimization may loosen, and schedule-only
optimizations (row pipelining, concurrent classes) are checked bitwise
against the schedule they replace.

**Cost.** Every manifest change, such as per-class fusion, needs a matching
runtime change and tests; older manifests stay readable only through
explicit defaults.
