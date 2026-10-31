# Legality and implementation rules

The source verifier owns structural legality: rank, supported element and
offset types, static feature dimension, contiguous layout, compatible output
shape, valid hints and declared alias rules. Dynamic offsets are a runtime
contract; the IR verifier cannot prove their contents.

`ragged-legalize` makes accumulation, identity and NaN semantics explicit.
`ragged-plan` chooses length classes and feature tiles from hints and target
resource budgets. `ragged-fuse` chooses the softmax statistics of each segment
class. It selects fused (online) statistics for a class only with a supported
resource footprint and disjoint storage; split softmax then computes online
statistics per part and merges them in a separate combine kernel. A multi-CTA reduction
requires an intervening kernel boundary; a CTA barrier cannot replace that
boundary. A forced illegal fusion is a diagnostic, not a fallback with changed
semantics. Unknown target chips and unsupported resource limits are rejected.

All block-wide barriers must execute under uniform CTA control flow. Bounds
checks around loads and stores may diverge, but must not conditionally skip a
barrier. Each segment must be classified exactly once. Outputs are written at
the original segment/element index, never at the worklist index.

Planning an optimization is not evidence that its kernel is implemented.
Unavailable mappings must explicitly fail or produce a report that identifies
the actual implemented fallback. A report may only list classes and kernel
names that were actually launched. Neither a CPU oracle nor a successful
ptxas invocation proves device-side synchronization, bounds safety, stream
compatibility or performance; the GPU tests and compute-sanitizer check those.
