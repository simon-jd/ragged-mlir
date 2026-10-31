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
| FBGEMM_GPU | 1.8.0, optional | `jagged_softmax` comparison when installed; not installed for the published results |

FBGEMM 1.8.0 documents PyTorch 2.13 and CUDA 12.6/13.0 as its tested pairing;
using it with PyTorch 2.14 is outside that matrix, which is why it is optional.
Source and compiler provenance (LLVM commit, ptxas version, compiler hash) is
recorded in every artifact manifest; see [provenance](provenance.md).
