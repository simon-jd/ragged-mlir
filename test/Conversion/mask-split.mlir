// RUN: ragged-opt %s -ragged-pipeline | FileCheck %s
// CHECK: args = ["values", "offsets", "out", "mask", "class_counts", "segment_ids", "partials", "partial_sums", "max_stats", "sum_stats"]
// CHECK-LABEL: gpu.func @ragged_0_split_normalize(
// CHECK: memref.load %arg3{{.*}} : memref<?x7xf32>
// CHECK-NEXT: arith.addf
// CHECK: arith.divf
// CHECK-LABEL: gpu.func @ragged_0_split_combine_online(
// CHECK-NOT: memref.load %arg3
// CHECK-LABEL: gpu.func @ragged_0_split_partial_online(
// CHECK: memref.load %arg3{{.*}} : memref<?x7xf32>
// CHECK-NEXT: arith.addf
// CHECK: arith.maximumf
// CHECK: math.exp
// CHECK-LABEL: gpu.func @ragged_0_cta_online(
// CHECK: memref.load %arg3{{.*}} : memref<?x7xf32>
// CHECK-NEXT: arith.addf
// CHECK-LABEL: gpu.func @ragged_0_warp_online(
// CHECK: memref.load %arg3{{.*}} : memref<?x7xf32>
// CHECK-NEXT: arith.addf
func.func @masked(%v: memref<?x7xf32>, %o: memref<?xi64>, %r: memref<?x7xf32>, %m: memref<?x7xf32>) {
  "ragged.segment_softmax"(%v, %o, %r, %m) : (memref<?x7xf32>, memref<?xi64>, memref<?x7xf32>, memref<?x7xf32>) -> ()
  return
}
