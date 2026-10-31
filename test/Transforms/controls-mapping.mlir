// RUN: ragged-opt %s -ragged-legalize -ragged-plan -ragged-fuse | FileCheck %s
// CHECK: classes = ["cta"]
// CHECK-SAME: device_classification = false
// CHECK: fusion = "three_pass"
func.func @fallback(%v: memref<?x7xf32>, %o: memref<?xi64>, %r: memref<?x7xf32>) {
  ragged.segment_softmax %v, %o into %r {warp_mapping = false, split_mapping = false, max_len = 32 : i64, online_fusion = false} : memref<?x7xf32>, memref<?xi64>, memref<?x7xf32>
  return
}
// CHECK: classes = ["cta"]
// CHECK-SAME: device_classification = false
// CHECK-SAME: reason = "device classification disabled; direct CTA mapping"
func.func @direct(%v: memref<?x7xf32>, %o: memref<?xi64>, %r: memref<?x7xf32>) {
  ragged.segment_reduce %v, %o into %r {kind = "sum", classification = false} : memref<?x7xf32>, memref<?xi64>, memref<?x7xf32>
  return
}
