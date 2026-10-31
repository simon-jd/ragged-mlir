// RUN: ragged-opt %s -ragged-legalize | FileCheck %s
func.func @sum(%v: memref<?x16xbf16>, %o: memref<?xi64>, %r: memref<?x16xbf16>) {
  // CHECK: ragged.map_segments
  // CHECK-SAME: accumulator = f32
  // CHECK-SAME: identity = "zero"
  // CHECK-SAME: nan = "propagate"
  // CHECK-SAME: operation = "sum"
  ragged.segment_reduce %v, %o into %r {kind = "sum"} : memref<?x16xbf16>, memref<?xi64>, memref<?x16xbf16>
  return
}
func.func @max(%v: memref<?x1xf32>, %o: memref<?xi32>, %r: memref<?x1xf32>) {
  // CHECK: identity = "negative_infinity"
  // CHECK-SAME: operation = "max"
  ragged.segment_reduce %v, %o into %r {kind = "max"} : memref<?x1xf32>, memref<?xi32>, memref<?x1xf32>
  return
}
