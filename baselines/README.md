# Comparison implementations

The benchmark (`bench/run.py`) and the GPU tests use these. The compiler and
its runtime never import or link them. All of them follow the same semantics
as ragged: f32 accumulation for f32 and bf16, NaN-propagating max with `-inf`
for empty segments, outputs cast back to the input dtype, offsets assumed valid
(no host readback), the PyTorch current stream.

- `torch_reference.py`: `torch.segment_reduce` (sum, max) with f32
  accumulation, and NJT (`torch.nested`, jagged layout) eager and
  `torch.compile(fullgraph=True)` versions. NJT max pads explicitly with
  `-inf` and reduces the dense `[B, max_len, D]` tensor, because NJT's own
  padding uses a finite minimum that breaks empty and all-`-inf` segments;
  that padded tensor can be far larger than the packed input. The benchmark
  passes `min_seqlen`/`max_seqlen` only with `--length-hints`; without them
  the eager sum and max pad every segment to the total row count.
- `cuda/baselines.cu` and `native.py`:
  - `cuda-naive`: one 256-thread block per segment. The threads form up to 32
    feature lanes (contiguous columns for a warp) times 256/lanes row lanes,
    so at D=1 all 256 threads stride over the rows and at D=128 eight row
    lanes cover 32 columns at a time. Each thread folds its rows (compensated
    f32 sums, NaN-propagating max) and a shared-memory tree combines the row
    lanes. Softmax makes three passes over the segment (max, sum of
    exponentials, normalize) inside the same block.
  - `cub`, at D=1: CUB 3.5 `DeviceSegmentedReduce` sum and max with a
    NaN-propagating max operator. Softmax is CUB's segmented max, an
    element-parallel kernel for `exp(x - max)` that finds each element's
    segment by binary search over the offsets, CUB's segmented sum of those
    values, and an element-parallel division.

  Each call includes bf16 promotion, CUB workspace queries and allocation,
  every kernel and the output cast.
- `triton_reference.py`: a handwritten Triton 3.8 kernel, 32-row tiles and up
  to 32 feature columns per program, one program per segment and feature tile;
  softmax uses max, denominator and normalize kernels. Sum uses lane
  compensation and a fixed reduction tree. No autotuning. It uses the CUDA
  toolkit's ptxas and libdevice (`TRITON_PTXAS_PATH`, `TRITON_LIBDEVICE_PATH`),
  with FMA fusion and flush-to-zero disabled.
- `fbgemm_reference.py`: FBGEMM 1.8 `jagged_softmax` on packed values and the
  original offsets, with `max_L=N` unless length hints give the exact maximum.
  Optional; see [compatibility](../docs/compatibility.md).

Build the CUDA/CUB library against CCCL 3.5.0 headers:

```sh
curl -fL https://github.com/NVIDIA/cccl/archive/refs/tags/v3.5.0.tar.gz -o .deps/cccl-v3.5.0.tar.gz
tar -xzf .deps/cccl-v3.5.0.tar.gz -C .deps
python scripts/build_baselines.py      # writes build-baselines/libragged_baselines.so and build.json
```

`CUB_VERSION=300500` is enforced at compile and load time; `build.json` keeps
the command, hashes, compiler version and ptxas resource report. Device tests
for these implementations are `test/gpu/test_native_baselines.py`,
`test_triton_baseline.py`, `test_njt_max_baseline.py` and
`test_fbgemm_baseline.py`.
