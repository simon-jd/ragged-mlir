# Python resource budgets and evidence

`ProgramSpec` accepts `smem_per_cta=49152` and `regs_budget=65536`.
Both are positive signed i64 integers. The CLI equivalents are
`--smem-per-cta` and `--regs-budget`. Values below the old planning floors are
valid requests; compilation can reject them when the emitted kernels cannot
fit. Both attributes are emitted explicitly in source IR, participate in its
canonical hash, and consume the existing per-signature specialization limit.

`smem_per_cta` limits measured static shared bytes plus dynamic launch bytes
for each kernel. `regs_budget` limits measured registers per thread multiplied
by the kernel's threads per CTA. This register product is the budget metric;
it is not an occupancy measurement or an estimate of hardware allocation
granularity. Larger budgets do not change the supported target or enable
dynamic shared-memory opt-in. Neither budget establishes a performance benefit.

New `Compiler.compile` results, including cache hits, require a complete
`resources` object with `status="measured"` and `source="ptxas"`. The ordered
records cover every main kernel, including pointwise materialization, followed
by every offsets-validation kernel. Runtime verifies exact entry coverage,
names and order; strict integer types and signed i64 arithmetic; threads from
the launch block; register products; static/dynamic shared totals; and both
requested budgets against the source specification. Missing, duplicate,
partial, unmeasured or inconsistent evidence rejects before module loading.
Compiler-side resource errors and malformed sidecars never publish a cache
artifact. Duplicate JSON fields also reject before publication.

The ptxas `static_shared_bytes` measurement is separate from a launch kernel's
`static_shared_bytes` IR allocation estimate. Optimizations can remove IR
allocations. The runtime does not substitute this estimate for a missing
measurement. Stack frames and spill store/load bytes remain explicit measured
fields; these budgets do not impose a separate zero-spill requirement.

Direct construction of legacy `CompiledProgram` objects without resource
metadata remains available only for the two default budgets. Their execution
reports have `resources=None`; they cannot claim measured resource acceptance.
`Compiler.compile` rejects such entries even at the default budgets. Compiler
binary identity and the source IR hash prevent ordinary cache reuse across
this transition.

`ExecutionReport.resources` is a detached copy of compiled resource evidence.
It describes all compiled entries, including validation entries when running
in validated mode, and remains present for B=0. Actual launches are described
separately by `kernel_names` and `grids`. Reporting resource evidence adds no
device readback, allocations or synchronization. Mutating a report does not
alter the runtime's retained manifest.
