# Tested environment

| Component | Version | Used for |
| --- | --- | --- |
| LLVM/MLIR | 23.1.3, official Linux X64 SDK (a source build with assertions also works) | the compiler and lit tests |
| CUDA Toolkit | 13.2 (nvcc/ptxas 13.2.86) | `ptxas`, the CUDA/CUB comparison library |
| CUDA driver | 595.99 on an RTX 4090 (sm_89) | all GPU tests and benchmarks |
| Python | 3.12 with nanobind 3.1 | runtime extension, host tests |
| PyTorch | 2.14.1+cu132 | GPU tests, benchmarks, custom-op registration; the CPU build runs the host tests |
| Triton | 3.8.0, using the toolkit's ptxas and libdevice | the Triton comparison kernel |
| CCCL / CUB | 3.5.0 (`CUB_VERSION=300500`, checked at compile and load time) | the CUB comparison |
| FBGEMM_GPU | 1.9.0+cu130, optional | `segment_sum_csr` (D=1 sum) and `jagged_softmax` comparisons |

FBGEMM publishes no CUDA 13.2 build: its releases dropped cu132 wheels. The
`fbgemm-gpu==1.9.0+cu130` wheel from the PyTorch cu130 index is built for
PyTorch 2.14 and loads in the pinned PyTorch 2.14.1+cu132 environment, whose
CUDA 13.2 runtime it uses under CUDA minor-version compatibility
(`pip install --no-deps fbgemm-gpu==1.9.0+cu130 --index-url
https://download.pytorch.org/whl/cu130`). The GPU tests check it against the
oracle; it fails on logits outside the offsets' integer range and on `+inf`
after a finite logit (strict xfails in `test/gpu/test_fbgemm_baseline.py`),
which the benchmark's inputs do not contain.
Source and compiler provenance (LLVM commit, ptxas version, compiler hash) is
recorded in every artifact manifest; see [provenance](provenance.md).
