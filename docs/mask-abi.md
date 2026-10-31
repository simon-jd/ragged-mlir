# Additive softmax mask ABI

`ragged.segment_softmax` and its legalized `ragged.map_segments` form accept an
optional fourth operand, after values, offsets and out. The mask is a dense,
contiguous default-memory-space `memref<NxDxf32>` with exactly the values shape.
There is no broadcasting, boolean mask or bf16 mask. Sum/max reductions reject
a mask operand. The generic spelling is:

```mlir
"ragged.segment_softmax"(%values, %offsets, %out, %mask)
  {scale = 5.000000e-01 : f32}
  : (memref<?x7xbf16>, memref<?xi64>, memref<?x7xbf16>, memref<?x7xf32>) -> ()
```

The optional custom syntax is `into %out mask %mask : memref<?x7xf32>` followed
by the existing attributes and values/offsets/out type list. Mask presence is
determined by the operand, not by a source `mask` attribute. Python's
`ProgramSpec.mask` controls operand generation and therefore changes the IR and
cache key. At runtime, `mask=None` is accepted exactly when the specialization
does not request a mask.

Python passes the tensor using `program.run(values, offsets, out, mask=mask)`.
The `segment_softmax` and `njt_softmax` adapters accept the same `mask=` keyword;
`bind(program)` produces a callable taking `(values, offsets, mask)` for a
masked specialization. The mask remains a dense [N,D] tensor when values come
from a contiguous jagged NestedTensor adapter.

Every input read used by statistics or normalization computes, in order:

```
x = f32(values[row, feature])
x = x * f32(scale)             # omitted for the default scale of one
x = x + mask[row, feature]
```

These are ordinary f32 multiply/add operations with no fast-math flags or
intermediate bf16 conversion. Negative infinity follows ordinary IEEE addition:
for example positive infinity plus a negative-infinity mask produces NaN. A
feature containing NaN or positive infinity, or containing only negative
infinity, follows the existing softmax NaN contract. There is no special
zero-output rule for a fully masked feature. The same transformed input is used
by online statistics, its normalization rescan, all three-pass stages and all
split partial/normalization stages.

All newly generated sidecars include a top-level `mask` boolean. Runtime binds
it to `ProgramSpec.mask`. Older sidecars without the field mean false. With a
mask, every main kernel—including classified init/classify—has argument names
`values, offsets, out, mask`, followed by the existing ordered workspace list.
The mask is a rank-2 `mlir-memref64-v1` descriptor, adding seven scalar 64-bit
parameters to every main kernel. No-mask signatures remain unchanged. The
schema and calling-convention ABI versions remain 1 and `mlir-memref64-v1`.
For example, online softmax has 26 scalar parameters, direct three-pass softmax
has 40, and full classified three-pass softmax with debug visits has 64. The
two offsets validators retain 17 parameters in all cases.

Mask is an input, not workspace. It must have the same device and [N,D] shape
as values and f32 dtype. Output must not overlap mask, including exact aliasing;
this is checked before any allocation or kernel launch. Mask may alias values
for read-only input use, provided out is disjoint. In-place values/out combined
with an overlapping mask is rejected. The IR verifier diagnoses aliases visible
as identical operands; runtime checks actual buffer overlap.

Offsets validation is unchanged: its separate init/check kernels take only
`values, offsets, validation_status`, retain 17 scalar descriptor parameters,
and never receive or read mask. Main workspace shapes and per-class scheduling
are unchanged by the mask. Compiler/lit and PTX/cubin evidence do not establish
GPU numerical or sanitizer correctness; those tests require CUDA execution.
