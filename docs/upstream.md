# Upstream boundaries

ragged-mlir is an out-of-tree project. Its dialect, verifiers, mapping decisions and
lowering are project code. MLIR dialect infrastructure, GPU/NVVM conversion,
NVPTX instruction selection and CUDA ptxas are upstream components. No upstream
contribution is claimed by this repository.

The source package is LLVM/MLIR 23.1.3, SHA256 and commit recorded in
`environment.lock`. `scripts/bootstrap_llvm.sh` verifies the published archive
digest before extraction and builds a separate `.deps/llvm-build` directory.
There are currently no patches. The original upstream source is not modified.

The lowering emits `vector.load` and `vector.store` for vector access, never
`vector.transfer_read` (`test/Conversion/cta.mlir` checks this): the default
GPU-to-NVVM pipeline leaves a rank-1 `vector.transfer_read` in a `gpu.module`
unlowered, as reported in
[llvm/llvm-project#214404](https://github.com/llvm/llvm-project/issues/214404)
(opened by another user; this project has not commented on it or sent a fix).
The issue's reproducer, run through `gpu-lower-to-nvvm-pipeline` with the
LLVM 23.1.3 SDK's `mlir-opt`, fails the same way here ("LLVM Translation
failed for operation: builtin.unrealized_conversion_cast"); LLVM main has
not been tried.
