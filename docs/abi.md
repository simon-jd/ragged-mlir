# Kernel ABI: mlir-memref64-v1

The runtime launches compiled kernel symbols through the CUDA Driver API in
the already-current context and the caller's PyTorch CUDA stream. It does not
create an independent CUDA context. Supported targets are exactly compute
capability 8.9; a different target must be rejected before launch.

Kernel arguments use the default MLIR ranked memref descriptor expansion,
not the bare-pointer calling convention. All descriptor integers are 64-bit.
The ordered scalar fields of a rank-r descriptor are:

```
allocated pointer, aligned pointer, offset,
size[0], ..., size[r-1], stride[0], ..., stride[r-1]
```

Pointers are passed as 64-bit CUDA addresses. The Python adapter uses the
tensor's `data_ptr()` for both pointers and passes offset zero, because
`data_ptr()` already includes a tensor view's storage offset. The buffers are
contiguous; sizes and strides are checked to fit signed 64-bit descriptor
arithmetic. Buffers are not deallocated by the kernel.

Reduction kernels receive `values` (rank 2, 7 fields), `offsets` (rank 1,
5 fields), and `out` (rank 2, 7 fields): 19 scalar arguments. Three-phase
softmax additionally receives two f32 `[B,D]` statistics buffers, `max_stats`
and `sum_stats`, for 33 scalar arguments. Every kernel must declare the same
complete argument sequence: `values`, `offsets`, `out`, optional `mask`, followed
by each declared workspace in order. The mask adds seven rank-two fields; see
[mask ABI](mask-abi.md). The manifest is the launch-order authority; optimized
softmax schedules may omit scratch buffers they do not use.

The [classified dispatch extension](classified-abi.md) adds typed rank-one and
rank-two workspaces for i64 class counters and original segment IDs, f32 split
partials, optional f32 softmax statistics, and optional i32 visit counters.
The runtime validates exact buffer names, dtypes, shapes, roles and launch order
against this protocol and the source specialization. Every worker grid y must
equal `ceil(D/(d_tile*d_vector))`, using the compiled feature tile and features
per thread. By default `d_vector` is the largest power of two dividing D that
fits 16 bytes, and the tile is `min(32, next power of two of D) / d_vector`
threads; an explicit optimization option can select a tile of 1, 2, 4, 8, 16 or 32
with one feature per thread ([optimizations.md](optimizations.md)). Shape and stride products, per-buffer bytes and total workspace
bytes must fit signed i64. CUDA grid limits are checked before allocation or
launch. The only new grid symbols are `ceil_B_128` and `min_B_resident`;
metadata strings are never evaluated as expressions. Device counters do not
determine host allocations or launch grids. Disabling pointwise fusion can add
a final f32 `logits[N,D]` workspace and a leading materialize kernel; see
[optimization controls](optimizations.md).

For the baseline, grid x is B and grid y is the number of feature tiles.
The compiler emits the actual block dimensions and feature tiling in the
manifest. Empty grids cause no kernel launch. Static workgroup allocations
live in GPU shared address space; `shared_bytes` records any dynamic shared
memory launch allocation. The two statistics buffers, when needed, cost
`2 * B * D * sizeof(float)` bytes. All workspace buffers of a call share one
scratch allocation from PyTorch's allocator, each at a 256-byte aligned offset,
made on the active stream (inside `with torch.cuda.stream(active_stream)` when
that is not the current stream). This is required even for uninitialized
scratch: a cached allocation can have pending writes from its earlier use.
`record_stream` extends allocation lifetime; it does not establish a dependency
on work already queued to a different allocation stream. Internal workspace
must therefore be allocated on its consumer stream; only the caller's tensors
are recorded on it. This adds no host synchronization.

The main kernels of a call run through a launch plan that the native module
builds once per (N, B): the kernel handles, grids and stream lanes in
submission order, and the scratch descriptors with their pointer fields as
offsets. A call then makes one native call with the inputs' descriptors and
the scratch base, which relocates the descriptors and issues every launch and,
for concurrent class chains, the fork and join (see
[classified-abi.md](classified-abi.md)). Checked mode's offsets validation
launches separately, before the plan. See the documented [PyTorch CUDA stream
semantics](https://docs.pytorch.org/docs/2.14/notes/cuda.html#cuda-streams).

For externally supplied `values`, `offsets`, and `out`, the caller must establish
cross-stream dependencies before `run` (for example, a consumer stream waiting on
the producer stream or an event). `run(..., stream=B)` may be called while A is
current; it allocates its own workspace on B and launches all kernels on B. Output
consumers on another stream must likewise wait for B. `record_stream` protects
external buffer lifetime without silently synchronizing unrelated caller work.

`ProgramSpec(..., debug_visits=True)` emits a separate cached specialization.
For this specialization, `run(..., debug_buffers=buffers)` accepts an initially
empty Python dictionary and fills it with the named workspace device tensors.
This is an opt-in lifetime handle for inspection; `run` never reads these debug
buffers back. In default `validated` mode it performs no host synchronization.
The caller must synchronize the execution stream before
reading these tensors on the host. Worklist entries beyond each corresponding
device count are uninitialized. With B=0, N must also be zero; validated mode returns no
launches, reports zero workspace bytes and leaves the dictionary empty.

`run(..., mode="checked")` uses the separate [checked offsets ABI](checked-abi.md).
It allocates a 56-byte device status buffer on the execution stream, launches
init and check kernels, copies only those seven i64 slots to pinned host memory,
and synchronizes that stream before any main scratch allocation or launch.
All supplied static length hints are checked, even with `offsets_checked=true`.
An invalid input raises `ValidationError` with indexed conditions; B=0 still
checks offsets[0]. This explicit mode includes validation work and its workspace
in the execution report. Artifacts without a validation object remain usable
only in validated mode.

## Artifact and launch metadata

`ragged-compile --emit=cubin -o program.cubin` writes an ELF cubin and
`program.cubin.json`. The sidecar describes the ABI version, target, actual
kernel launch sequence, argument descriptor names, workspace shapes, and
dispatch explanation. The cache wraps this sidecar with the complete cache
key, a cubin SHA256 and a metadata SHA256 before atomically publishing it.
Invalid fields, target mismatches, unsupported ABI versions and incomplete
artifacts must fail before `cuModuleLoadData`/`cuLaunchKernel`.

A loaded module is shared by every program compiled to the same artifact on a
device. It stays loaded while any of them is open; when the last one is closed
(or collected), the runtime records an event on every stream the module
launched on and unloads it only after those events complete, since
`cuModuleUnload` is not stream-ordered. Modules still loaded at exit are
unloaded the same way, before interpreter teardown.

The compiler's source/IR tests and the T0 compilation matrix establish the
parameter structure only after their recorded runs pass. Real pointer,
stream-ordering and in-flight lifetime tests still need an sm_86 or sm_89 device.
