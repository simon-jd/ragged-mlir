# Checked offsets ABI

Every new compiler artifact contains separate validation kernels in the same
cubin. They are described by a `validation` object, outside the main `kernels`
and `workspace` lists. Main computation signatures, launch order and workspace
remain unchanged. `validated` execution never allocates, launches or reads back
validation state. `checked` execution validates before any main kernel and
before any output write, including when `offsets_checked = true`.

```json
{
  "abi": "offsets-check-v1",
  "workspace": [{"name": "validation_status", "dtype": "i64", "shape": [7]}],
  "conditions": ["first_nonzero", "final_mismatch", "negative", "above_n",
                 "nonmonotone", "below_min_len", "above_max_len"],
  "sentinel": 9223372036854775807,
  "min_len": null,
  "max_len": null,
  "kernels": [
    {"name": "ragged_0_offsets_init", "role": "init", "grid": [1,1,1],
     "block": [128,1,1], "shared_bytes": 0, "static_shared_bytes": 0,
     "args": ["values", "offsets", "validation_status"]},
    {"name": "ragged_0_offsets_check", "role": "check",
     "grid": ["ceil_B1_128",1,1], "block": [128,1,1],
     "shared_bytes": 0, "static_shared_bytes": 0,
     "args": ["values", "offsets", "validation_status"]}
  ]
}
```

`min_len` and `max_len` are null when absent and the exact source i64 hints
otherwise. Runtime binds them to the requested program. The two kernels share
the existing `mlir-memref64-v1` convention: rank-2 values, rank-1 offsets and
rank-1 status expand to 7+5+5 = 17 scalar 64-bit arguments. The status memref is
`memref<7xi64>`. Values is used only for its leading dimension N; neither kernel
loads or stores values data, and neither has an output argument.

Init writes INT64_MAX to all seven slots. Check visits each offset index i in
[0,B] once and uses signed i64 atomic minimum to retain the earliest offending
index independently in each slot. The conditions and index meanings are:

| Slot | Condition | Reported index |
| --- | --- | --- |
| 0 | offsets[0] != 0 | 0 |
| 1 | offsets[B] != N | B |
| 2 | offsets[i] < 0 | offset i |
| 3 | offsets[i] > N | offset i |
| 4 | offsets[i+1] < offsets[i] | segment i, the left endpoint |
| 5 | valid segment length < min_len | segment i |
| 6 | valid segment length > max_len | segment i |

Length subtraction occurs only after both endpoints are in [0,N] and the pair
is monotone, so extreme i64 offsets cannot overflow the subtraction. The
monotonicity comparison itself does not subtract. All supplied length hints
are checked regardless of the `offsets_checked` attribute.

`ceil_B1_128` means `ceil((B+1)/128)` and is resolved with overflow-safe integer
arithmetic after validating the host offsets shape. It is a fixed symbol, not
an expression interpreter. B=0 still launches validation and checks offsets[0];
the usual empty-program main-kernel early return happens only after validation
succeeds. Init and check execute in order on the actual caller stream. The
runtime then copies the seven i64 slots to pinned host memory, waits on that
stream, and raises `ValidationError` if any condition failed. Its `errors`
dictionary contains every failed condition and its earliest index; `condition`
and `index` identify the first failed condition in the listed order. The message
contains `<condition> at index <i>` for every failure. Main allocation and
computation proceed only after success. B=0 with N>0 is also checked on device,
where either the first or final offset condition must fail.

Python selects this behavior with `program.run(..., mode="checked")`.
The custom-op, NJT convenience functions and `bind(program, mode="checked")`
accept the same mode; their default remains `validated`. Fake/meta execution
checks shapes and mode only, without reading offsets or proving device validity.

A successful `ExecutionReport` includes both validation kernels and their grids
before the main launches. `workspace_bytes` includes the 56-byte device status;
`validation_workspace_bytes` identifies this component. The additional 56-byte
pinned host buffer is not device workspace. `validation_host_ms` measures host
wall time across validation allocation, launches, transfer and stream wait;
it may include earlier work already queued to that stream and is not a CUDA
event timing. In checked mode `host_enqueue_ms` includes that wait as part of
the complete host call. B=0 success reports two validation launches, 56 bytes
of device workspace and no main classes. A failing check returns no success
report and submits no main work.

Older artifacts without `validation` remain usable in validated mode and must
be rejected in checked mode. `test/gpu/test_checked.py` covers every
condition on the device, including validation ordered after earlier writes on
another stream.
