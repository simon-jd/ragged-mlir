# Optimization controls

Source operations accept seven boolean attributes, defaulting to true:
`classification`, `online_fusion`, `warp_mapping`, `split_mapping`, `shuffle`,
`pointwise_fusion` and `row_pipelining`. Optional `feature_tile` is an i64 power of two from 1
through 32. Python groups these options in `Optimizations`, held by
`ProgramSpec.optimizations`; they specialize the source IR and cache key.

| Control | Behavior when disabled |
| --- | --- |
| classification | Direct CTA mapping for every segment; no classification/worklist kernels. |
| online_fusion | Softmax uses separate max, exponential sum and normalization phases in every class. |
| warp_mapping | CTA covers lengths that would otherwise use the warp class. |
| split_mapping | CTA covers lengths that would otherwise use split kernels. |
| shuffle | Warp workers keep 32 threads and use a shared-memory reduction tree with uniform barriers. |
| pointwise_fusion | A separate kernel materializes f32 transformed inputs when scale or mask is present. |
| row_pipelining | Each lane's row loop loads and folds one row per iteration. |

`feature_tile` overrides the default next power of two of D, capped at 32.
Feature tile 1 is an ablation of feature packing; it does not mean that all
features are processed in one CTA. A worker uses `thread_id % tile` as feature
lane and `thread_id / tile` as row lane, advances rows by `threads / tile`, and
launches `ceil(D/tile)` feature tiles. Tail features remain guarded; when the
tile divides D every tile is full and the guard is compiled out, so loads carry
no branch. Runtime binds the resolved tile and grid to the source option.

`row_pipelining` restructures each lane's row loop (sum, max and exponential-sum
reductions, online statistics, and normalization). The main loop issues eight
loads, rows `r, r+s, ..., r+7s` for the lane stride `s`, before folding them
into the accumulators in that order; a rolled loop covers the last `count mod 8`
rows. Every floating-point operation happens in the same order as in the rolled
loop, so results are bitwise identical with the control on or off
(`test/gpu/test_scheduling.py`); the gain is eight loads in flight per lane
instead of one. Each element is still read and then written by its own lane,
so in-place softmax is unaffected.

Disabling mapping choices preserves coverage: CTA handles the remaining length
range, and an empty candidate set falls back to CTA. `strategy="naive"` forces
the CTA class; its existing `debug_visits=true` instrumentation still uses the
single-class classified ABI. Explicit `classification=false` conflicts with
debug visits and is rejected. `force_fusion=true` conflicts with
`online_fusion=false` and is rejected.

Online statistics are chosen per segment class and recorded in `class_fusion`
([classified-abi.md](classified-abi.md)). They need no length bound (see the
arithmetic below), so a class uses them unless online fusion is disabled, the
output aliases the input, or the class's tree exceeds `smem_per_cta`: a
shuffle warp needs no shared memory, a 128-thread block 1 KiB. Split segments
keep a kernel boundary between blocks: each of the eight parts writes its own
statistics and a combine kernel merges them before normalization.
`force_fusion=true` requires online statistics in every class. A disjoint
softmax can retain online statistics when only pointwise fusion is disabled.
In-place softmax conservatively retains separate statistics/read and output
phases in every class.

## Requested and effective settings

Sidecars record:

```json
{
  "optimizations": {
    "requested": {
      "classification": true, "online_fusion": true,
      "warp_mapping": true, "split_mapping": true,
      "shuffle": true, "pointwise_fusion": true, "row_pipelining": true,
      "feature_tile": null
    },
    "effective": {
      "classification": true, "online_fusion": true,
      "warp_mapping": true, "split_mapping": true,
      "shuffle": true, "pointwise_fusion": false, "row_pipelining": true
    }
  },
  "d_tile": 32,
  "pointwise_materialized": false
}
```

Requested fields bind exactly to the source specification. Effective
classification means the classified dispatch ABI is used. Effective warp/split
mean those classes are present; effective shuffle means warp is present and
shuffle was requested. Effective online fusion means at least one class uses
online statistics (`fusion` is `online` or `mixed`).
Effective pointwise fusion means scale differs from one or mask is present,
and those operations execute within input reads. Effective row pipelining
equals the request. If no such transformation
exists, both effective pointwise fusion and materialization are false. Static
bounds, aliasing, resources and other requested controls can make an enabled
optimization inapplicable; the effective record describes emitted kernels.
Execution reports retain these compiled settings even for B=0. Actual
`kernel_names`, `classes`, `grids` and device workspace describe that call:
validated B=0 launches no kernels and allocates no workspace; checked B=0 still
runs its separate offsets validation before returning.

## Pointwise materialization

Disabling online fusion alone does not disable scale/mask fusion. Disabling
pointwise fusion when a transformation exists adds `logits`, an f32 `[N,D]`
workspace, at the end of the existing ordered workspace list. Scale applies to
sum/max as well as softmax; an additive f32 mask is defined only for softmax.
All main kernels receive the same complete input and workspace descriptor list.

The first main kernel has role `materialize`, name `ragged_0_materialize`, grid
`[256,1,1]`, block `[128,1,1]`, and zero dynamic/static shared memory. It walks
the flattened N*D domain with a grid-stride loop, promotes values to f32,
multiplies scale, adds mask if present, and stores only logits. Runtime checks
the workspace shape/byte products before allocation. These fixed launch
dimensions are a correctness implementation, not a measured performance choice.

Subsequent compute kernels read logits and never reload values or mask data,
nor apply scale/addition a second time. Their output cast still follows the
original output dtype. Classified init/classify follow materialize in the main
launch list; all other scheduling remains unchanged. Offsets validation remains
a separate 17-parameter protocol and runs before materialization in checked
mode. It receives neither mask nor logits. B=0 skips all main work after any
required validation. With N=0 and B>0, materialize has no loop iterations, while
the main reductions still write empty-segment identities.

The extra device workspace is N*D*4 bytes. Allocation and materialization belong
in complete-call reports and benchmarks. There is no fast-math relaxation,
host counter readback or new synchronization in the validated path. Disabling
both fusion controls removes both statistics fusion and pointwise fusion; the
two effects remain separately measurable. GPU numerical and sanitizer checks
and [ptxas resource-budget enforcement](resource-budgets.md) are validated separately.

## Online statistics arithmetic

The online kernel uses a stable exponential reference and compensated f32 mass
accumulation. The reference starts at the first finite value and advances only
when a new value exceeds it by more than 1.0. It is a normalization reference,
not necessarily the exact running maximum. The mass and its rounding correction
are both rescaled on an advance. Thus individual masses are at most `exp(1)`,
and actual reference changes attenuate older rounding errors by less than
`exp(-1)`. Small increases in values avoid repeated nearly-unit exponential
scalings, while Kahan addition preserves long runs of small probability masses.
The warp/CTA tree merges these exponential references and masses before the
normalization traversal; for split, each part's tree result is merged with the
other parts' in a combine kernel, largest reference first, then a compensated
sum of the rescaled masses. Accuracy does not depend on segment length: each
lane's sum is compensated and every mass is at most `exp(1)`, so a lane that
folds 16,384 rows meets the same fixed tolerance as a short one
(`test/gpu/test_class_fusion.py` runs 65,536-row segments through the split and
CTA classes). NaN, positive infinity, and all-negative-infinity
features retain the documented output semantics. No fast-math reassociation or
tolerance relaxation is used. CPU float32 replay and compiler checks cover the
known precision counterexamples; execution on a CUDA device remains required.
