# Compiler provenance

`ragged-compile --toolchain-json` records the linked LLVM version, LLVM base
revision, revision source, target chip/features and measured ptxas version.
Every cubin sidecar repeats the same information. Python also hashes the actual
compiler executable into the cache key, preventing two differently built or
modified compilers from sharing a cached binary under the same release label.

When the LLVM SDK defines `LLVM_REVISION`, its header supplies the revision and
`llvm_revision_source` is `llvm-vcs-header`. Official LLVM source archives lack
git metadata, so their generated header can leave that macro undefined. In that
case configure Ragged with:

```sh
cmake -S . -B build-source \
  -DMLIR_DIR="$PWD/.deps/llvm-build/lib/cmake/mlir" \
  -DRAGGED_LLVM_SOURCE_ARCHIVE="$PWD/.deps/llvm-project-23.1.3.src.tar.xz"
```

CMake requires LLVM 23.1.3 and verifies the complete archive SHA256 against
`c44186a7762ed28954be72e5ff6df9808e0779d4f1bf014ecc4e7e211d31ee34` before
using its known base revision `0d261d1ca552c95a8f007e061c787ac7132fbcbc`.
The report labels this source `verified-source-archive` and includes
`llvm_source_archive_sha256`. An unrecognized archive, wrong version or absent
revision without an archive fails configuration. The fallback identifies the
release base; it does not assert that an extracted source tree is unmodified.
The compiler executable hash distinguishes the current statically linked build.

`scripts/bootstrap_llvm.sh` writes `.deps/ragged-source-sdk.cmake`, an initial
CMake cache file binding the SDK to its verified source archive.
`RAGGED_LLVM_SOURCE_ARCHIVE` overrides the archive path.
