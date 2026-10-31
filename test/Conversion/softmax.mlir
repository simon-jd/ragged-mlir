// RUN: ragged-opt %s -ragged-pipeline | FileCheck %s
// CHECK: gpu.func @ragged_0_normalize
// CHECK: math.exp
// CHECK: arith.divf
// CHECK: gpu.func @ragged_0_sum_stats
// CHECK: scf.for {{.*}} iter_args({{.*}}) -> (f32, f32)
// CHECK: math.exp
// CHECK: arith.subf
// CHECK: arith.addf
// CHECK: math.absf
// CHECK: arith.cmpf olt
// CHECK: arith.select
// CHECK: scf.yield {{.*}} : f32, f32
// CHECK: gpu.barrier
// CHECK: gpu.func @ragged_0_max_stats
// CHECK: arith.maximumf
// CHECK: gpu.barrier
func.func @softmax(%v: memref<?x128xf32>, %o: memref<?xi64>) {
  ragged.segment_softmax %v, %o into %v {strategy = "naive"} : memref<?x128xf32>, memref<?xi64>, memref<?x128xf32>
  return
}
