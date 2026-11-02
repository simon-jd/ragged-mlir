# Device-classified dispatch ABI

This document fixes the compiler/runtime boundary for `dispatch_abi =
"classified-v3"`. Version 2 differs from version 1 in two places: worker grids
are sized to the device (`min_B_resident` replaces `min(B, 256)`), and the
worker chains of different segment classes may run concurrently. Version 3
indexes the split partials by split-queue position instead of segment ID, so
they have `B_split` rows instead of `B`. The kernel calling convention remains
`abi = "mlir-memref64-v1"`, with each memref expanded as described in `abi.md`.
The existing single-class, non-debug warp/CTA ABI remains supported unchanged.
This protocol describes implementation requirements, not evidence of GPU
execution correctness; GPU tests and sanitizer results are tracked separately.

## Metadata and buffers

The manifest retains `schema: 1`, `classes`, `thresholds: [32,4096]`, `fusion`,
`alias_safe`, `toolchain`, `kernels` and `workspace`. Classified manifests add
`dispatch_abi: "classified-v3"`, `split_parts: 8`, `debug_visits: bool`, and
`backend: "classified-persistent"`. All kernels take exactly the same argument
list, in this order: `values`, `offsets`, `out`, optional `mask`, followed by the
declared workspace list below. Optional workspaces are omitted from every
signature. The separate offsets validators retain their own smaller signature.

For softmax, `class_fusion` lists the statistics schedule of each class, in
the order of `classes`: `online` or `three_pass`. For a warp or CTA class,
`online` is one kernel that computes the maximum and exponential sum in one
pass over the segment and normalizes in a second; for split it is a per-part
online kernel, a combine kernel and normalization (roles below). `three_pass`
uses separate maximum, exponential-sum and normalization phases. `fusion`
summarizes the list: `online` or
`three_pass` when every class agrees, otherwise `mixed`. Reductions declare
`fusion: "not_applicable"` and no `class_fusion`. A manifest without
`class_fusion` is uniform, every class using `fusion`; a runtime that predates
`class_fusion` rejects `mixed` as an unknown schedule.

| Workspace name | dtype | shape | Present when |
| --- | --- | --- | --- |
| `class_counts` | i64 | `[3]` | always |
| `segment_ids` | i64 | `[3, "B"]` | always |
| `partials` | f32 | `["B_split", 8*D]` | split class is active |
| `partial_sums` | f32 | `["B_split", 8*D]` | split softmax uses online statistics |
| `max_stats` | f32 | `["B", D]` | softmax with a split class or a three-pass class |
| `sum_stats` | f32 | `["B", D]` | softmax with a split class or a three-pass class |
| `visits` | i32 | `["B"]` | `debug_visits = true` |
| `logits` | f32 | `["N", D]` | scale/mask transformation is materialized |

`D` and `8*D` above are emitted as concrete positive integer dimensions.
`B_split` is `min(B, N // (T + 1))`, where `T` is the largest bound among the
warp (32) and CTA (4,096) classes present: every split segment is longer than
`T`, so no more than that many segments can be split, and the runtime sizes the
buffers without reading a device counter. For D=128 softmax with 4,096
segments of 262,144 rows that is 63 rows (0.5 MB for both buffers) instead of
4,096 (33.6 MB). The
symbolic workspace dimensions are `"B"` and, for logits, `"N"`. The runtime validates all
shape products, byte counts, dtypes and argument order before allocating on the
active CUDA stream. There is no expression interpreter in the manifest.
The runtime derives workspace bytes from the resolved shapes and dtype sizes.
It checks each contiguous shape/stride product, per-buffer bytes and the total
byte sum against signed i64 before allocating. Segment IDs and queue capacities
remain i64; the i32 visits dtype stores a count of one, not a narrowed segment ID.

Queue rows 0, 1 and 2 mean warp, CTA and split. `segment_ids[class, position]`
contains the **original segment ID**. Classification reserves positions per
warp: each warp ranks its lanes within their classes with a shuffle scan over
three packed 8-bit counts, and lanes 0-2 each add the warp's count for one
class to that class's i64 counter, at most three atomics per warp. Every lane
then writes its segment ID at the returned base plus its rank, so a warp's
segments of one class occupy consecutive positions in segment order; the order
of warps within a queue is unspecified. (Aggregating per 128-thread block
through shared memory cut the kernel further in an A/B run, 8.0 to 6.6 µs at
D=1, but its 128-segment runs made the D=128 CTA workers slower than the
warp's 32-segment runs; see [design-tradeoffs.md](design-tradeoffs.md).) Queue entries beyond the device
counter are uninitialized and must never be read. Each queue has capacity B.
`partials[position, part*D + feature]` stores f32 partial results, where
`position` is the segment's position in the split queue.
Three-pass softmax reuses this buffer for max then sum, separated by kernel
boundaries. Online split softmax stores each part's exponential reference
there and its mass, the exponential sum relative to that reference, at the
same index of `partial_sums`; `split_combine_online` takes the largest
reference of the eight parts and the compensated sum of the masses rescaled to
it, and writes them to `max_stats` and `sum_stats` for `split_normalize`. A
part without mass (empty, or all `-inf`) has reference `-inf` and contributes
nothing; a NaN or `+inf` reference makes the feature NaN, as everywhere else.

## Launch dimensions and ordering

Kernel `grid` dimensions use positive integers or the following finite symbol
vocabulary; none is executable text:

| Symbol | Meaning |
| --- | --- |
| `B` | segment count, retained for the existing ABI |
| `ceil_B_128` | `(B // 128) + (B % 128 != 0)` |
| `min_B_resident` | `min(B, W)`, where `W` is the kernel's device worker count below |

`ceil(D/(d_tile*d_vector))` is emitted as a concrete integer. The default tile is
`min(32, next power of two of D) / d_vector` threads; `feature_tile` can override
it ([optimizations.md](optimizations.md)). Validated B=0
requires N=0 and returns with no allocations or launches. Checked B=0 first
runs the separate offsets validation. Otherwise the prologue runs on the
caller's stream, preceded by optional materialization:

1. Role `init`: grid `[1,1,1]`, block `[128,1,1]`. Clear all three counters and,
   when enabled, initialize visits with a thread-strided loop over B.
2. Role `classify`: grid `["ceil_B_128",1,1]`, block `[128,1,1]`.
3. The worker chain of each active class: warp kernels, CTA kernels and split
   kernels, each chain in manifest order, as below.

Worker grid x is `"min_B_resident"`, grid y is `ceil(D/(d_tile*d_vector))`. A worker CTA loops
over queue positions starting at block x and advancing by grid x, stopping at
the device counter for its class. All threads in the CTA follow the same queue
loop and finish with a uniform barrier before the next queue entry can reuse
shared storage. There is no host counter readback or host synchronization.

The worker count `W` of a kernel is one resident wave: after loading the
module, the runtime asks the driver how many blocks of that kernel fit on one
multiprocessor (`cuOccupancyMaxActiveBlocksPerMultiprocessor`, with its block
size and dynamic shared memory), multiplies by the multiprocessor count and
divides by the kernel's grid y·z, at least 1. On an RTX 4090 (128
multiprocessors) a 32-thread warp worker gets 24 blocks per multiprocessor and
a 128-thread worker 12, so at D=1 the warp, CTA and split-partial grids are
3,072, 1,536 and 192 wide. `W` depends only on the device and the compiled
kernel, never on device counters.

Workers of different classes read disjoint queue rows, process disjoint
original segments, and write disjoint output rows and workspace rows
(`max_stats`, `sum_stats` and `visits` are indexed by segment, `partials`
and `partial_sums` by split-queue position, and only the split chain uses
them). The
runtime may therefore run the class chains concurrently: after `classify` it
records an event on the caller's stream, launches the CTA chain and then the
split chain on two streams that wait for it, launches the warp chain on the
caller's stream,
and makes the caller's stream wait for an event recorded at the end of each
other chain. Work enqueued on the caller's stream after the call, including
reuse of the scratch allocated on it, is ordered after every worker. The
native module issues the fork, the launches and the join in one call (a launch
plan, [abi.md](abi.md)); if a launch fails it still joins the streams it forked
before raising. Its two side streams per caller stream are non-blocking,
inherit the caller's priority, are created on that stream's first concurrent
call and kept for the process, with their events; run a call on a stream once
before capturing it into a CUDA graph, so that they and the launch plan exist. CUDA graph capture
of a validated call records the same fork and join. `run(..., concurrent=False)`
launches the chains one after another on the caller's stream, in manifest
order. `ExecutionReport.kernel_names` and `grids` list kernels in submission
order, and `streams` counts the streams the call used.

A classified program compiled by `Compiler.compile` (unless `adaptive=False`)
also holds its block-per-segment specialization as `program.naive`, a separate
artifact with the direct ABI of [abi.md](abi.md). `run(..., max_len=L)` lets
the cost model in `ragged/selection.py` choose between the two for that call;
`select="classified"` or `select="naive"` forces one, and without `max_len`
the classified launch sequence above runs. `ExecutionReport.selected` names
the schedule that ran and `selection` holds the statistics and predicted
times behind the choice. Debug-visit specializations have no companion.

Warp workers use 32 threads and f32 `gpu.shuffle` reductions; disabling shuffle
selects a shared-memory reduction tree with uniform barriers. CTA and split
workers use 128 threads and uniform shared-memory reduction barriers. The
[optimization contract](optimizations.md) defines mapping controls and the
optional materialization phase.

| Operation/mapping | Kernel roles, in order | grid z |
| --- | --- | --- |
| warp or CTA sum/max | `warp_sum`, `warp_max`, `cta_sum` or `cta_max` | 1 |
| warp or CTA softmax, `class_fusion` online | `warp_online` or `cta_online` | 1 |
| warp or CTA softmax, `class_fusion` three_pass | `<class>_max_stats`, `<class>_sum_stats`, `<class>_normalize` | 1 |
| split sum/max | `split_partial_sum`, `split_combine_sum` or corresponding `_max` roles | 8, then 1 |
| split softmax, `class_fusion` online | `split_partial_online`, `split_combine_online`, `split_normalize` | 8,1,8 |
| split softmax, `class_fusion` three_pass | `split_partial_max`, `split_combine_max_stats`, `split_partial_exp_sum`, `split_combine_sum_stats`, `split_normalize` | 8,1,8,1,8 |

Each worker kernel additionally declares `segment_class` (`warp`, `cta`, or
`split`) and its role. `init` and `classify` have no `segment_class` field.
`shared_bytes` is zero because workgroup storage is static;
`static_shared_bytes` records the IR allocation. Final ptxas resource reports
remain the authoritative compiled resource evidence.

For split work, quotient/remainder partitioning divides the segment into eight
contiguous disjoint ranges. Empty ranges contribute zero to sum and negative
infinity to max. All partial and statistics storage is f32; only final output
writes cast to bf16. Sum and exponential-sum use compensated f32 lane
accumulation. The combine stages preserve the same NaN and infinity rules.

When static hints prune classes, classification routes to an available class:
an available warp handles lengths <=32, an available CTA handles lengths
<=4096, and the last available class handles the remainder. A single split
class therefore accepts every segment. This prevents omitted classes from
silently dropping a segment. Single warp/CTA programs enter this protocol only
when debug visits are requested; otherwise their existing direct launches stay
unchanged.

## Aliasing and debug visits

`debug_visits` is an optional boolean source-op attribute, default false. Its
presence forces the classified protocol even for one warp/CTA class. Exactly
one atomic visit increment occurs in the final output stage per original
segment: feature tile zero, thread zero, and (for split normalization) part
zero. Empty segments are counted too. Intermediate statistics and partial
kernels never increment visits. The runtime may let tests retain the allocated
visits tensor; ordinary execution never reads it back or synchronizes. Python
exposes `ProgramSpec(debug_visits=True)` and
`program.run(values, offsets, out, stream=stream, debug_buffers=buffers)`, where
`buffers` must initially be an empty dictionary. Successful submission fills it
with all named workspace device tensors so tests can inspect counters, worklists
and visits after explicitly synchronizing that stream. B=0 leaves it empty.
Supplying `debug_buffers` to a non-debug specialization is rejected.

`alias_safe` is true only when every class uses three passes (`fusion:
"three_pass"`), including split softmax; in-place softmax compiles to that
schedule. Each read/statistics phase finishes at a kernel boundary before that
class writes outputs. Different classes operate on disjoint original segments, so running
their chains concurrently keeps in-place softmax exact. Online
softmax and sum/max require disjoint values/out buffers. Partial overlap and
workspace aliasing are always invalid.
