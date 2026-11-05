// RUN: ragged-opt %s -ragged-legalize -ragged-plan | FileCheck %s
// The target chip is an operation attribute; sm_89 is the default.
func.func @default_chip(%v: memref<?x128xf32>, %o: memref<?xi32>, %r: memref<?x128xf32>) {
  // CHECK-LABEL: func.func @default_chip
  // CHECK: ragged.dispatch
  // CHECK-SAME: chip = "sm_89"
  ragged.segment_reduce %v, %o into %r {kind = "sum"} : memref<?x128xf32>, memref<?xi32>, memref<?x128xf32>
  return
}
func.func @sm86(%v: memref<?x128xf32>, %o: memref<?xi32>, %r: memref<?x128xf32>) {
  // CHECK-LABEL: func.func @sm86
  // CHECK: ragged.dispatch
  // CHECK-SAME: chip = "sm_86"
  // CHECK-SAME: d_tile = 8 : i64
  // CHECK-SAME: d_vector = 4 : i64
  ragged.segment_reduce %v, %o into %r {kind = "sum", chip = "sm_86"} : memref<?x128xf32>, memref<?xi32>, memref<?x128xf32>
  return
}
