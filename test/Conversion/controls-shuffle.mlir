// RUN: ragged-opt %s -ragged-pipeline | FileCheck %s
// CHECK: ragged.d_tile = 1 : i64
// CHECK: grid = ["B", 7, 1]
// CHECK: shuffle = false
// CHECK-LABEL: gpu.func @ragged_0_online(
// CHECK-SAME: workgroup({{.*}}memref<64xf32
// CHECK-NOT: gpu.shuffle
// CHECK: gpu.barrier
// CHECK: arith.maximumf
// CHECK: gpu.barrier
// CHECK-NOT: gpu.shuffle
func.func @shared(%v: memref<?x7xf32>, %o: memref<?xi64>, %r: memref<?x7xf32>) {
  ragged.segment_softmax %v, %o into %r {max_len = 32 : i64, shuffle = false, feature_tile = 1 : i64} : memref<?x7xf32>, memref<?xi64>, memref<?x7xf32>
  return
}
