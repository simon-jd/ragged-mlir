# Upstream boundaries

ragged-mlir is an out-of-tree project. Its dialect, verifiers, mapping decisions and
lowering are project code. MLIR dialect infrastructure, GPU/NVVM conversion,
NVPTX instruction selection and CUDA ptxas are upstream components. No upstream
contribution is claimed by this repository.

The source package is LLVM/MLIR 23.1.3, SHA256 and commit recorded in
`environment.lock`. `scripts/bootstrap_llvm.sh` verifies the published archive
digest before extraction and builds a separate `.deps/llvm-build` directory.
There are currently no patches. The original upstream source is not modified.

The `vector.transfer_read` lowering failure of LLVM issue #214404 reproduces on
release 23.1.3; LLVM main has not been tested, and no fix or contribution has
been submitted upstream. No issue
comments, assignments or pull requests have been sent. The corresponding
32-bit shuffle workaround is what the warp class uses; the GPU tests cover it.

