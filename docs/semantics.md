# Ragged semantics, version 1

This contract applies equally to every generated mapping and baseline. Passing
FileCheck establishes IR structure, not numerical correctness on a device.

## Storage and types

`values` is a contiguous, row-major `[N,D]` memref, `1 <= D <= 1024`, with
compile-time D and runtime N. Inputs and outputs have the same element type,
f32 or bf16. Arithmetic accumulates in f32; bf16 results round to nearest, ties
to even. `offsets` is a contiguous i32 or i64 vector of length B+1.

Offsets must start at zero, be nondecreasing, stay within `[0,N]`, and end at N.
For i32, N must fit a signed 32-bit integer. Empty segments and B=0 are valid;
B=0 necessarily implies N=0 and offsets `[0]`. Negative or overflowing shape
products must be rejected before constructing a device descriptor. The caller
provides all output storage. Reductions output `[B,D]`; softmax outputs `[N,D]`.

An absent element is absent, not an implicit zero. Reductions are along the
ragged (N) dimension independently for each feature column.

Static `min_len`/`max_len` hints are caller promises about every segment,
including empty segments. Validated execution assumes them; checked execution
must validate them before a pruned dispatch can run. Optional scalar `scale`
is represented as f32 and multiplies each promoted input in f32 before the
segment operation.

Softmax optionally takes an additive mask with exactly shape `[N,D]`, dense
row-major layout and f32 element type, including when values are bf16. Each
logit is computed as `f32(values) * f32(scale) + mask` in f32 before the
existing softmax operation. The transformed logits are not rounded back to
bf16 before reduction. Broadcasting, boolean masks and ragged mask objects are
unsupported. Values already contain their input-dtype quantization; the mask
retains f32 precision. Non-finite behavior below applies to the transformed
logits: all -inf logits yield NaNs, while mixed finite/-inf logits assign zero
to the -inf entries. Ordinary f32 arithmetic applies even when scale is zero,
so zero times infinity yields NaN.

Python enables this input with `ProgramSpec("softmax", mask=True)` and supplies
it using `program.run(..., mask=mask_tensor)`. Mask presence must match the
compiled specialization and participates in the source IR hash. The mask must
be on the values CUDA device, be contiguous, and require no gradients. Its
storage must not overlap offsets or output, including partially overlapping
views. Read-only overlap with values is allowed; if values is also the in-place
output, the output-overlap restriction still applies. Mask lifetimes and caller
cross-stream dependencies follow the same rules as values. Checked offsets
validation does not load mask data and keeps its separate 17-argument ABI.

## Values

| Operation | Empty segment | Non-finite inputs in a feature column |
|---|---|---|
| sum | +0 | NaN propagates; +inf plus -inf produces NaN |
| max | -inf | NaN propagates; otherwise ordinary maximum |
| softmax | no stored output | NaN, +inf, or all -inf produces NaN for the entire column |

Softmax is `exp(x - max(x)) / sum(exp(x - max(x)))`. In a column with at least
one finite value and no NaN or +inf, -inf inputs map to zero. Signed-zero sign
bits are not promised for nonempty reductions. Reassociation is permitted only
within the numerical bound below. Fast-math flags that drop NaNs or infinities
are forbidden.

The current CTA sum and softmax denominator use f32 compensated per-thread
accumulation before the shared-memory reduction. When the running total is
non-finite, compensation resets to zero so ordinary NaN/infinity propagation
is preserved. This addresses the measured error of a long serial f32 sum;
the numerical bounds below remain unchanged across mappings.

## Numerical validation

The independent `oracle/` module converts already-quantized inputs to float64
and never imports compiler code. It computes the mathematical reference and
compares output after conversion to float64. f32 bounds are rtol=2e-5,
atol=2e-6; bf16 bounds are rtol=2e-2, atol=2e-3. Sum uses
`atol + rtol * sum(abs(input))` to cover cancellation without changing the
tolerance per mapping. Other operations use `atol + rtol * abs(reference)`.
NaN masks and infinity signs must agree exactly; an infinity is not close to a
finite value. The bounds are fixed before GPU tuning. An implementation that
fails them needs a corrected algorithm or an explicitly revised, reviewed
contract, not a per-test relaxed tolerance.

For each nonempty softmax feature column with finite probabilities, all
probabilities must lie in `[0,1]` and their float64 sum must differ from one by
at most `rtol + atol`. This additional normalization check prevents large
segments with individually tiny probabilities from incorrectly passing with
all-zero output. Finite sum error bounds presume no f32 intermediate overflow;
tests must distinguish unavoidable final output-dtype overflow from erroneous
infinities or overflow followed by cancellation.

## Aliasing and validation modes

Reductions require disjoint input and output storage. A softmax implementation
may write in place only when all reads that could observe overwritten values
are separated from writes by kernel boundaries. Partial overlap is rejected.
Declared non-aliasing is a caller contract that the Python runtime must check.
The compiler rejects unsupported aliasing instead of silently ignoring it.

`validated` mode assumes that the caller has established the offsets contract;
it must not copy offsets to the host or synchronize. `checked` mode must validate
every condition before an unsafe kernel can access memory and report the failed
condition. Its generated init/check kernels record the earliest index for seven
conditions, then the runtime reads back 56 bytes on the selected stream and
raises before main computation if any condition failed. Static hints are checked
regardless of `offsets_checked`. Older artifacts lacking the validation ABI
reject checked mode. See `checked-abi.md`. Read-only input lifetimes and output
lifetimes extend through completion on the selected CUDA stream.

