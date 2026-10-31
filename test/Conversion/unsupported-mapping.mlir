// RUN: ragged-opt %s -ragged-pipeline | FileCheck %s
// Unbounded lengths now produce real queues and eight-way split stages; every
// class uses online statistics.
// CHECK-DAG: ragged.backend = "classified-persistent"
// CHECK-DAG: ragged.dispatch_abi = "classified-v2"
// CHECK-DAG: grid = ["min_B_resident", 4, 8]
// CHECK-DAG: name = "partials", shape = ["B", 1024]
// CHECK-DAG: gpu.func @ragged_0_split_normalize
// CHECK-DAG: gpu.func @ragged_0_split_combine_online
// CHECK-DAG: gpu.func @ragged_0_split_partial_online
// CHECK-DAG: gpu.func @ragged_0_cta_online
// CHECK-DAG: gpu.func @ragged_0_warp_online
// CHECK-DAG: gpu.shuffle idx
// CHECK-DAG: gpu.func @ragged_0_classify
// CHECK-DAG: memref.atomic_rmw addi {{.*}} : (i64, memref<3xi64>) -> i64
// CHECK-DAG: gpu.func @ragged_0_init
func.func @long_tail(%v: memref<?x128xf32>, %o: memref<?xi64>, %r: memref<?x128xf32>) {
  ragged.segment_softmax %v, %o into %r : memref<?x128xf32>, memref<?xi64>, memref<?x128xf32>
  return
}
