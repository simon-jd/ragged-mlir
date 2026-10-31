# Error categories and evidence

The Python API distinguishes invalid input, missing tools, failed compilation,
and unusable artifacts. Errors propagate to callers; there is no CPU numerical
fallback. A compiler or resource failure prevents cache publication.

| Failure | Diagnostic / exception | Handling |
| --- | --- | --- |
| Unsupported dtype, rank, D, layout, hints, controls or budget | `ValidationError`; C++ verifier diagnostics include the source location | Correct the specification; no valid compiled program is produced |
| Illegal schedule rejected by a compiler pass | Pass diagnostic, surfaced by Python as `CompilationError` | Use a legal specialization; an invalid combination caught earlier by the frontend raises `ValidationError` |
| Missing executable or unidentifiable toolchain | `ToolchainUnavailable` | Build/configure the pinned compiler and tools |
| ptxas failure, timeout, missing cubin, or newly emitted sidecar that is missing/unreadable/invalid JSON | `CompilationError` | Inspect retained diagnostics; failed output is not published to cache |
| Incomplete ptxas evidence or measured resource over budget | Compiler diagnostic naming the entry/constraint, surfaced as `CompilationError` | Fix the evidence or choose a feasible explicit budget; no automatic tuning retry |
| Corrupt binary/metadata, wrong key/provenance, malformed ABI/resources | `CacheMismatch` | Reject before module loading; do not treat a corrupt entry as a valid cache hit |
| More than 16 specializations for one signature | `SpecializationLimit` | Explicitly manage the cache or reduce specialization choices |
| Tensor shape/dtype/layout/alias/gradient or stream mismatch | `ValidationError` | Correct the call before allocating main workspace or launching kernels |
| Missing PyTorch/native extension, CPU tensors, wrong GPU architecture | `RuntimeUnavailable` | Install the supported execution environment; no implicit backend substitution |
| CUDA Driver error | Native `RuntimeError` with the CUDA operation, error name/code and description | Preserve the original device diagnostic; the local cuInit probe reports no device |

In `mode="checked"`, `ValidationError` also carries `.condition`, `.index`,
and `.errors`. The errors dictionary records the earliest offending index for
each detected condition: `first_nonzero`, `final_mismatch`, `negative`,
`above_n`, `nonmonotone`, `below_min_len`, and `above_max_len`. The named
condition is selected by the protocol's fixed priority order; it is not an
assertion that this condition has the lowest index across all categories.
See [checked offsets ABI](checked-abi.md) for exact index meanings and readback
cost. Validated mode assumes the offsets contract and does not check contents.

Host fault injection and actual compiler rejection tests cover cache corruption,
tool failures, malformed metadata, invalid resource reports, budget boundaries,
and source legality. Checked-offset device diagnostics, illegal memory access,
races, synchronization, ordering and in-flight lifetime still require the GPU
suite and compute-sanitizer, which `test/gpu` and the README's results cover.

Two online softmax arithmetic counterexamples found during development are
covered by the corrected, stable online algorithm without any change to the
oracle tolerances.
