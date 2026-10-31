// REQUIRES: cuda-ptxas
// RUN: not ragged-compile --features=+ptx999 --toolchain-json 2>&1 | FileCheck %s --check-prefix=BAD
// RUN: not ragged-compile %s --features=+ptx999 --emit=cubin -o %t.bad 2>&1 | FileCheck %s --check-prefix=BAD
// RUN: ragged-compile %s --features=+ptx80 --emit=ptx -o %t.80.ptx
// RUN: FileCheck %s --check-prefix=PTX80 < %t.80.ptx
// RUN: ragged-compile %s --features=+ptx81 --emit=ptx -o %t.81.ptx
// RUN: FileCheck %s --check-prefix=PTX81 < %t.81.ptx
// BAD: unsupported NVPTX target feature: +ptx999
// PTX80: .version 8.0
// PTX81: .version 8.1
func.func @sum(%v: memref<?x128xf32>, %o: memref<?xi64>, %r: memref<?x128xf32>) {
  ragged.segment_reduce %v, %o into %r {kind = "sum", strategy = "naive"} : memref<?x128xf32>, memref<?xi64>, memref<?x128xf32>
  return
}
