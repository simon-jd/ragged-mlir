// RUN: ragged-opt %s -ragged-legalize -ragged-plan | FileCheck %s
func.func @short(%v: memref<?x128xf32>, %o: memref<?xi32>, %r: memref<?x128xf32>) {
  // CHECK: ragged.dispatch
  // CHECK-SAME: classes = ["warp"]
  // CHECK-SAME: d_tile = 8 : i64
  // CHECK-SAME: d_vector = 4 : i64
  // CHECK-SAME: reason = "static length bounds select one class"
  // CHECK-NOT: ragged.classify
  // CHECK: mapping = "warp"
  ragged.segment_reduce %v, %o into %r {kind = "sum", max_len = 32 : i64} : memref<?x128xf32>, memref<?xi32>, memref<?x128xf32>
  return
}
func.func @long_tail(%v: memref<?x7xf32>, %o: memref<?xi64>, %r: memref<?x7xf32>) {
  // CHECK: classes = ["warp", "cta", "split"]
  // CHECK-SAME: d_tile = 8 : i64
  // CHECK: ragged.classify
  // CHECK: mapping = "classified"
  ragged.segment_softmax %v, %o into %r : memref<?x7xf32>, memref<?xi64>, memref<?x7xf32>
  return
}
