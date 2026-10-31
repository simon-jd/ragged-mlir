// RUN: ragged-opt %s -ragged-pipeline | FileCheck %s
// CHECK: grid = [256, 1, 1], name = "ragged_0_materialize", role = "materialize"
// CHECK: ragged.pointwise_materialized = true
// CHECK: name = "logits", shape = ["N", 7]
// CHECK-LABEL: gpu.func @ragged_0_normalize(
// CHECK-SAME: %[[LOGITS:[a-z0-9]+]]: memref<?x7xf32>) kernel
// CHECK-NOT: memref.load %arg0
// CHECK-NOT: memref.load %arg3
// CHECK: memref.load %[[LOGITS]]
// CHECK: arith.divf
// CHECK-LABEL: gpu.func @ragged_0_sum_stats(
// CHECK: memref.load %arg6
// CHECK: math.exp
// CHECK-LABEL: gpu.func @ragged_0_max_stats(
// CHECK: memref.load %arg6
// CHECK: arith.maximumf
// CHECK-LABEL: gpu.func @ragged_0_materialize(
// CHECK: gpu.grid_dim x
// CHECK: scf.for
// CHECK: memref.load %arg0
// CHECK: arith.extf
// CHECK: arith.mulf
// CHECK: memref.load %arg3
// CHECK: arith.addf
// CHECK: memref.store {{.*}} %arg6
// CHECK-NOT: arith.truncf
func.func @materialize(%v: memref<?x7xbf16>, %o: memref<?xi64>, %r: memref<?x7xbf16>, %m: memref<?x7xf32>) {
  "ragged.segment_softmax"(%v, %o, %r, %m) {classification = false, online_fusion = false, pointwise_fusion = false, scale = 5.000000e-01 : f32} : (memref<?x7xbf16>, memref<?xi64>, memref<?x7xbf16>, memref<?x7xf32>) -> ()
  return
}
