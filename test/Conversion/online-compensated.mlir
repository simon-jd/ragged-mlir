// RUN: ragged-opt %s -ragged-pipeline | FileCheck %s
// Stable exponential references prevent nearly-unit rescalings on slow ramps.
// The three loop states are reference, mass, and Kahan rounding correction.
// CHECK: gpu.func @ragged_0_online
// CHECK: %[[ONE:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: scf.for {{.*}} iter_args(%[[ANCHOR:.*]] = {{.*}}, %[[SUM:.*]] = {{.*}}, %[[ERROR:.*]] = {{.*}}) -> (f32, f32, f32)
// CHECK: %[[MAX:.*]] = arith.maximumf %[[ANCHOR]], {{.*}} : f32
// CHECK: %[[DELTA:.*]] = arith.subf %[[MAX]], %[[ANCHOR]] : f32
// CHECK: arith.cmpf ogt, %[[DELTA]], %[[ONE]] : f32
// CHECK: arith.cmpf uno, %[[ANCHOR]],
// CHECK: %[[REFERENCE:.*]] = arith.select {{.*}}, %[[MAX]], %[[ANCHOR]] : f32
// CHECK: %[[EXPARG:.*]] = arith.subf %[[ANCHOR]], %[[REFERENCE]] : f32
// CHECK: %[[SCALE:.*]] = math.exp %[[EXPARG]] : f32
// CHECK: arith.mulf %[[SUM]], %[[SCALE]] : f32
// CHECK: arith.mulf %[[ERROR]], %[[SCALE]] : f32
// CHECK: arith.subf
// CHECK: arith.addf
// CHECK: arith.subf
// CHECK: arith.subf
// CHECK: math.absf
// CHECK: arith.cmpf olt
// CHECK: arith.constant 0x7FC00000 : f32
// CHECK: scf.yield {{.*}} : f32, f32, f32
func.func @softmax(%v: memref<?x128xf32>, %o: memref<?xi64>, %r: memref<?x128xf32>) {
  ragged.segment_softmax %v, %o into %r {min_len = 33 : i64, max_len = 4096 : i64, force_fusion = true} : memref<?x128xf32>, memref<?xi64>, memref<?x128xf32>
  return
}
