# Resource budgets

`regs_budget` (default 65536) and `smem_per_cta` (default 49152 bytes) are positive signed-i64 source attributes. They constrain every compiled entry, including materialization, classification, and the two checked-offset validation kernels. The former 4096-register and 512-byte minimums are removed. No tuning or mapping retry occurs after a measured budget failure.

For cubin output, the compiler requires a complete `ptxas -v` report from the same MLIR serialization invocation that produced the cubin. It checks each entry exactly once, binds its launch block dimensions to the manifest, and rejects missing, duplicate, or unknown entries and missing register/stack/spill records. An otherwise complete `Used ...` line with no `smem` clause means zero static shared memory. Budgets are checked before opening any output, sidecar, or resource log file, so rejection does not publish a partial artifact or overwrite an existing artifact.

The sidecar schema is:

```json
{
  "resources": {
    "status": "measured",
    "source": "ptxas",
    "requested": {"regs_budget": 65536, "smem_per_cta": 49152},
    "kernels": [
      {
        "name": "ragged_0_sum",
        "registers_per_thread": 24,
        "threads_per_cta": 128,
        "registers_per_cta": 3072,
        "static_shared_bytes": 512,
        "dynamic_shared_bytes": 0,
        "shared_bytes_per_cta": 512,
        "stack_frame_bytes": 0,
        "spill_store_bytes": 0,
        "spill_load_bytes": 0
      }
    ]
  }
}
```

The example entry is illustrative; actual values come from ptxas. Entries follow the main kernel launch list, then the validation kernel list. `threads_per_cta` is the product of the three launch block dimensions. `registers_per_cta` is measured registers per thread times that product; `shared_bytes_per_cta` is measured static shared bytes plus the launch's dynamic `shared_bytes`. These must not exceed the requested budgets. Kernel-level `static_shared_bytes` outside this object is an IR allocation estimate and can differ after compiler elimination.

PTX output reports `resources = {"status":"unmeasured", "source":null, "requested":{"regs_budget":65536,"smem_per_cta":49152}, "kernels":[]}` with the actual requested values. Legal, plan, GPU, and NVVM IR carry the corresponding `ragged.resources` attribute. None of these outputs claims a post-ptxas resource measurement. Cubin assembly uses `-v` and the target's default optimization level; an independent PTX assembly comparison should use the same architecture, optimization level, and toolkit. The complete compiler log is retained as `<output>.ptxas.log` only after acceptance.

The register product is the defined budget measure, not an occupancy estimate or a claim about hardware allocation granularity. Stack and spill counts are reported without a new rejection threshold. Measurements establish resource constraints; numerical execution, sanitizer checks, and performance require a CUDA device. See [Python resource contracts](python-resources.md) for cache and runtime validation.
