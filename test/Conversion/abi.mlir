// REQUIRES: cuda-ptxas
// RUN: ragged-compile %s --emit=ptx -o %t.ptx
// RUN: FileCheck %s < %t.ptx
// CHECK: .target sm_89
// CHECK: .visible .entry ragged_0_max(
// CHECK-COUNT-19: {{^[ \t]*}}.param .u64
// CHECK-NEXT: )
func.func @max(%v: memref<?x16xf32>, %o: memref<?xi64>, %r: memref<?x16xf32>) {
  ragged.segment_reduce %v, %o into %r {kind = "max", strategy = "naive"} : memref<?x16xf32>, memref<?xi64>, memref<?x16xf32>
  return
}
