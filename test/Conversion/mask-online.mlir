// RUN: ragged-opt %s -ragged-pipeline | FileCheck %s
// CHECK: ragged.mask = true
// CHECK-LABEL: gpu.func @ragged_0_offsets_check(
// CHECK-SAME: memref<?x7xbf16>, {{.*}}memref<?xi32>, {{.*}}memref<7xi64>) kernel
// CHECK-LABEL: gpu.func @ragged_0_online(
// CHECK-SAME: %[[MASK:[a-z0-9]+]]: memref<?x7xf32>
// CHECK: %[[INPUT:.*]] = memref.load {{.*}} : memref<?x7xbf16>
// CHECK-NEXT: %[[PROMOTED:.*]] = arith.extf %[[INPUT]] : bf16 to f32
// CHECK: %[[SCALED:.*]] = arith.mulf %[[PROMOTED]]
// CHECK-NEXT: %[[BIAS:.*]] = memref.load %[[MASK]]{{.*}} : memref<?x7xf32>
// CHECK-NEXT: arith.addf %[[SCALED]], %[[BIAS]] : f32
// CHECK: arith.maximumf
// CHECK: gpu.shuffle idx
// Normalization must reload and transform the same input, then cast only output.
// CHECK: memref.load {{.*}} : memref<?x7xbf16>
// CHECK-NEXT: arith.extf
// CHECK: arith.mulf
// CHECK-NEXT: memref.load %[[MASK]]
// CHECK-NEXT: arith.addf
// CHECK: arith.divf
// CHECK: arith.truncf
func.func @masked(%v: memref<?x7xbf16>, %o: memref<?xi32>, %r: memref<?x7xbf16>, %m: memref<?x7xf32>) {
  ragged.segment_softmax %v, %o into %r mask %m : memref<?x7xf32> {max_len = 32 : i64, scale = 5.000000e-01 : f32} : memref<?x7xbf16>, memref<?xi32>, memref<?x7xbf16>
  return
}
