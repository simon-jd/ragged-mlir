// REQUIRES: cuda-ptxas
// RUN: ragged-compile %s --chip=sm_86 --emit=ptx -o %t.ptx
// RUN: FileCheck %s < %t.ptx
// RUN: FileCheck %s --check-prefix=MANIFEST < %t.ptx.json
// RUN: not ragged-compile %s --chip=sm_90 --emit=ptx -o %t.bad.ptx 2>&1 | FileCheck %s --check-prefix=CHIP
// CHECK: .target sm_86
// CHECK: .visible .entry ragged_0_sum(
// MANIFEST: "chip":{{ ?}}"sm_86"
// CHIP: only --chip=sm_86 or sm_89
func.func @main(%v: memref<?x7xf32>, %o: memref<?xi64>, %r: memref<?x7xf32>) {
  ragged.segment_reduce %v, %o into %r {kind = "sum", strategy = "naive"} : memref<?x7xf32>, memref<?xi64>, memref<?x7xf32>
  return
}
