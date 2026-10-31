// RUN: not ragged-compile %s --emit=gpu 2>&1 | FileCheck %s
// CHECK: entry must contain only one ragged operation on values, offsets, out and optional mask arguments, followed by return
func.func @unexpected_host_code(%v: memref<?x16xf32>, %o: memref<?xi64>, %r: memref<?x16xf32>) {
  %c0 = arith.constant 0 : index
  ragged.segment_reduce %v, %o into %r {kind = "max", strategy = "naive"} : memref<?x16xf32>, memref<?xi64>, memref<?x16xf32>
  return
}
