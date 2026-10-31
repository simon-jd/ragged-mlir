// RUN: ragged-opt %s -ragged-pipeline | FileCheck %s
// CHECK-DAG: ragged.dispatch_abi = "classified-v3"
// CHECK-DAG: role = "warp_online"
// CHECK-DAG: role = "cta_online"
// CHECK-NOT: partials
// CHECK-LABEL: gpu.func @ragged_0_cta_online
// CHECK: memref<256xf32, #gpu.address_space<workgroup>>
// CHECK: gpu.grid_dim x
// CHECK: arith.maximumf
// CHECK: arith.divf
// CHECK: gpu.barrier
// CHECK-LABEL: gpu.func @ragged_0_warp_online
// CHECK: gpu.shuffle down
// CHECK: arith.maximumf
// CHECK: gpu.shuffle idx
// CHECK: arith.divf
// CHECK: gpu.barrier
func.func @softmax(%v: memref<?x7xf32>, %o: memref<?xi64>, %r: memref<?x7xf32>) {
  ragged.segment_softmax %v, %o into %r {max_len = 4096 : i64, force_fusion = true} : memref<?x7xf32>, memref<?xi64>, memref<?x7xf32>
  return
}
