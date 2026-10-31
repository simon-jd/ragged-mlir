# Design trade-offs

The compiler makes a handful of structural choices: what is compiled, how
segments are mapped to warps and blocks, how the class queues are built and
launched, and how softmax keeps its statistics. This document records, for
each, what else could have been done and what the choice gives up, with the
measurements behind it.

Numbers are complete-call microseconds on an RTX 4090 from `results/bench/`
(`summary.json`, the ablations, `profile.txt` and `host.txt`). A few
comparisons come from separate A/B runs whose records are not in `results/`;
those are marked "A/B". "D" is the feature dimension, "long tail" the
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

**Why.** One block per segment puts a long segment on one SM. On the long-tail
inputs `ragged-naive` takes 83 µs for the D=1 sum, 266 µs for the D=1
softmax and 604 µs for the D=128 softmax, against 45, 83 and 438 µs with
classification. Sorting costs a full sort per call. Host binning needs the
offsets on the host, which means a device-to-host copy and a synchronization
in the middle of the call.

**Cost.** Two prologue kernels (init and classify), three class chains and a
stream fork and join on every call. That cost is not always repaid:
`ragged-naive` is faster in 8 of the 19 benchmark cases, every uniform and
bursty case at D=1 (the uniform D=1 sum takes 33 µs naive, 48 µs classified
and 15 µs in `torch.segment_reduce`) and the bursty D=128 sum and max.
Nothing chooses between the two automatically; exact length hints that leave one class compile the block-per-segment kernel, which
is slower than classification for the uniform D=128 sum (166 against 138 µs).
The thresholds are fixed, not tuned per GPU or per D. A segmented reduction
specialized for D=1 (CUB) is faster on the bursty input (25 µs against 39).

## Persistent workers with a static stride

**Decision.** Each worker kernel launches one resident wave, sized at module
load with `cuOccupancyMaxActiveBlocksPerMultiprocessor` (1,536 128-thread
blocks or 3,072 warps on a 4090, divided among feature tiles and split
parts). Worker `x` processes queue positions `x, x + grid, x + 2·grid, …`.

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
classify takes 6.8 µs on the D=1 long-tail sum.

**Cost.** In the A/B runs the long-tail sum and max at D=128 were 5 to 6%
slower than with per-segment atomics, in every aggregated variant: with the
classes serial, the CTA sum worker took 155 µs instead of 129 (155 µs in the
committed trace too). It comes from the queue order, not the submission
order.

## Running the classes concurrently, CTA chain first

**Decision.** After classify, the runtime records an event, launches the CTA
chain and then the split chain on two side streams that wait for it, the
warp chain on the caller's stream, and joins at the end. The classes touch
disjoint segments, so no other ordering is needed.

**Alternatives.** Run the chains one after another on the caller's stream
(kept as `run(concurrent=False)`); submit the longest-running chain first.

**Why.** The chains of a long-tail input overlap: the D=128 long-tail sum
takes 194 µs concurrent and 215 µs serial, and concurrent classes are faster
in all 18 f32 cases of the ablation. The submission order was measured, not
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
(`host.txt`), host time per call dropped from 64 to 23 µs for a D=1 sum, and
from 99 to 30 µs for a D=128 softmax that then had 13 kernels. Because the
benchmark's CUDA events count the gaps in which the GPU waits for the host,
that is visible in end-to-end times too. A graph would need its kernel
parameters updated for every call's tensors; it was not tried, because the
launch plan already brought host time to about 23 µs.

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
long-tail softmax takes 651 µs and the uniform one 634 µs, with online
statistics 438 and 313 µs, and the uniform case's CTA kernel moves its data at
0.95 TB/s against the 4090's 1.01 TB/s DRAM bandwidth. Rescaling at
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
D=128 softmax, almost all split segments, takes 666 µs with three passes in
every class and 506 µs with online statistics (Triton 646). A grid-wide
barrier would need a cooperative launch with every block
resident, which cannot share the GPU with the other classes' workers.

**Cost.** On the long tail the per-part kernel is bound by the latency of the
fold: with the classes serial it takes 61 µs. Run concurrently, it runs
beside the CTA chain, which takes longer (`profile.txt`), so the call does not
wait on it.

## Row pipelining without reassociation

**Decision.** Each lane's row loop issues eight loads, then folds them in row
order; a rolled loop covers the remainder. When the feature tile divides D,
the bounds guard is compiled out, so the loads carry no branch.

**Alternatives.** The rolled loop; unrolling with eight independent
accumulators combined at the end.

**Why.** Eight loads in flight per lane instead of one: the D=128 long-tail
sum takes 194 µs against 215 µs without, the softmax 438 against 600.
Folding in row order keeps every floating-point operation in the rolled
loop's order, so results are bitwise identical with the control on or off,
which a GPU test checks. Independent accumulators would shorten the
dependency chain further but change results with the unroll factor.

**Cost.** The online fold's chain is not shortened. Recentering once per
batch of eight rows would shorten it, at the price of the bitwise guarantee.
Pipelining is not faster everywhere: in six of the 18 f32 cases it is slower,
by up to 6% (uniform D=128 max, 139 against 131 µs), all with uniform lengths
except the bursty D=128 softmax (3%).

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
