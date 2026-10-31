// RUN: ragged-opt %s -ragged-legalize -ragged-plan | FileCheck %s
// CHECK: classes = ["warp"]
// CHECK-SAME: device_classification = true
// CHECK: ragged.classify
func.func @short_debug(%v: memref<?x1xf32>, %o: memref<?xi64>, %r: memref<?x1xf32>) {
  ragged.segment_reduce %v, %o into %r {kind = "sum", max_len = 32 : i64, debug_visits = true} : memref<?x1xf32>, memref<?xi64>, memref<?x1xf32>
  return
}
// CHECK: classes = ["split"]
// CHECK-SAME: device_classification = true
// CHECK: ragged.classify
func.func @only_split(%v: memref<?x1xf32>, %o: memref<?xi64>, %r: memref<?x1xf32>) {
  ragged.segment_reduce %v, %o into %r {kind = "max", min_len = 4097 : i64} : memref<?x1xf32>, memref<?xi64>, memref<?x1xf32>
  return
}
