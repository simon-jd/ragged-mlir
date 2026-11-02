// RUN: ragged-opt %s -ragged-pipeline | FileCheck %s
// The lane loop must carry both its f32 total and rounding error. A finite
// guard clears the error for inf/NaN without changing the total's IEEE result.
// The rolled loop shows one step; row-pipelining.mlir folds eight such steps
// per iteration in the same order.
// CHECK: gpu.func @ragged_0_sum
// CHECK: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[INF:.*]] = arith.constant 0x7F800000 : f32
// CHECK: scf.for {{.*}} iter_args(%[[TOTAL:.*]] = %[[ZERO]], %[[ERROR:.*]] = %[[ZERO]]) -> (f32, f32)
// CHECK: %[[VALUE:.*]] = scf.if
// CHECK: %[[CORRECTED:.*]] = arith.subf %[[VALUE]], %[[ERROR]] : f32
// CHECK: %[[NEXT:.*]] = arith.addf %[[TOTAL]], %[[CORRECTED]] : f32
// CHECK: %[[DELTA:.*]] = arith.subf %[[NEXT]], %[[TOTAL]] : f32
// CHECK: %[[RAWERROR:.*]] = arith.subf %[[DELTA]], %[[CORRECTED]] : f32
// CHECK: %[[MAGNITUDE:.*]] = math.absf %[[NEXT]] : f32
// CHECK: %[[FINITE:.*]] = arith.cmpf olt, %[[MAGNITUDE]], %[[INF]] : f32
// CHECK: %[[NEXTERROR:.*]] = arith.select %[[FINITE]], %[[RAWERROR]], %[[ZERO]] : f32
// CHECK: scf.yield %[[NEXT]], %[[NEXTERROR]] : f32, f32
// CHECK: gpu.barrier
func.func @sum(%v: memref<?x128xf32>, %o: memref<?xi64>, %r: memref<?x128xf32>) {
  ragged.segment_reduce %v, %o into %r {kind = "sum", strategy = "naive", row_pipelining = false, vectorize = false} : memref<?x128xf32>, memref<?xi64>, memref<?x128xf32>
  return
}
