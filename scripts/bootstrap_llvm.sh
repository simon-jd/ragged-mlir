#!/usr/bin/env bash
# Build the locked upstream release without installing or modifying upstream.
set -euo pipefail
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
deps_dir=${RAGGED_DEPS_DIR:-"$project_root/.deps"}
jobs=${RAGGED_BUILD_JOBS:-16}
python_bin=${RAGGED_PYTHON:-"$project_root/.venv/bin/python"}
version=23.1.3
archive="llvm-project-$version.src.tar.xz"
sha256=c44186a7762ed28954be72e5ff6df9808e0779d4f1bf014ecc4e7e211d31ee34
logs_dir="$deps_dir/llvm-logs"
mkdir -p "$deps_dir" "$logs_dir"
if [[ ! -f "$deps_dir/$archive" ]]; then
  curl -fL --retry 3 "https://github.com/llvm/llvm-project/releases/download/llvmorg-$version/$archive" -o "$deps_dir/$archive.part"
  mv "$deps_dir/$archive.part" "$deps_dir/$archive"
fi
printf '%s  %s\n' "$sha256" "$deps_dir/$archive" | sha256sum --check
if [[ ! -d "$deps_dir/llvm-project-$version.src/llvm" ]]; then
  tar -xJf "$deps_dir/$archive" -C "$deps_dir"
fi
cmake -S "$deps_dir/llvm-project-$version.src/llvm" -B "$deps_dir/llvm-build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DLLVM_ENABLE_PROJECTS=mlir -DLLVM_TARGETS_TO_BUILD='X86;NVPTX' \
  -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_ENABLE_RTTI=ON -DLLVM_ENABLE_EH=ON \
  -DMLIR_ENABLE_CUDA_RUNNER=ON \
  -DLLVM_BUILD_EXAMPLES=OFF -DLLVM_INCLUDE_TESTS=ON \
  -DLLVM_PARALLEL_LINK_JOBS=2 -DPython3_EXECUTABLE="$python_bin" \
  -DCUDAToolkit_ROOT=/usr/local/cuda \
  2>&1 | tee "$logs_dir/llvm-configure.log"
/usr/bin/time -v -o "$logs_dir/llvm-build-resources.txt" \
  cmake --build "$deps_dir/llvm-build" --parallel "$jobs" \
  --target mlir-opt mlir-tblgen FileCheck not mlir-runner mlir_cuda_runtime \
  2>&1 | tee "$logs_dir/llvm-build.log"
du -sh "$deps_dir/llvm-build" > "$logs_dir/llvm-build-size.txt"
# Preserve the explicit archive binding for the out-of-tree project build.
# Upstream source tarballs intentionally have no git revision header.
cat > "$deps_dir/ragged-source-sdk.cmake" <<EOF
set(MLIR_DIR "$deps_dir/llvm-build/lib/cmake/mlir" CACHE PATH "" FORCE)
set(RAGGED_LLVM_SOURCE_ARCHIVE "$deps_dir/$archive" CACHE FILEPATH "" FORCE)
set(RAGGED_LLVM_TEST_TOOLS_DIR "$deps_dir/llvm-build/bin" CACHE PATH "" FORCE)
EOF
printf 'Source SDK ready; configure Ragged with: cmake -G Ninja -C %q -S %q -B %q\n' \
  "$deps_dir/ragged-source-sdk.cmake" "$project_root" "$project_root/build"

