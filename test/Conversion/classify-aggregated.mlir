// RUN: ragged-opt %s -ragged-pipeline | FileCheck %s
// The classify kernel ranks a warp's lanes with a shuffle scan over packed
// per-class counts; lanes 0-2 then reserve each class's positions with one
// atomic add, and every lane stores its segment ID at base + rank.
// CHECK-DAG: name = "ragged_0_classify", role = "classify", shared_bytes = 0 : i64, static_shared_bytes = 0 : i64
// CHECK-LABEL: gpu.func @ragged_0_classify(
// CHECK-NOT: workgroup
// CHECK: scf.if
// CHECK: memref.load {{.*}} : memref<?xi64>
// CHECK: arith.select
// CHECK: arith.shli
// CHECK-COUNT-5: gpu.shuffle up
// CHECK: gpu.shuffle idx
// CHECK: arith.minui
// CHECK: scf.if
// CHECK-NEXT: arith.index_cast
// CHECK-NEXT: arith.extui
// CHECK-NEXT: memref.atomic_rmw addi {{.*}} : (i64, memref<3xi64>) -> i64
// CHECK: gpu.shuffle idx
// CHECK: gpu.shuffle idx
// CHECK: scf.if
// CHECK: memref.store {{.*}} : memref<3x?xi64>
// CHECK: gpu.return
// CHECK-NOT: memref.atomic_rmw
// CHECK-LABEL: gpu.func @ragged_0_init
func.func @long_tail(%v: memref<?x7xf32>, %o: memref<?xi64>, %r: memref<?x7xf32>) {
  ragged.segment_reduce %v, %o into %r {kind = "sum"} : memref<?x7xf32>, memref<?xi64>, memref<?x7xf32>
  return
}
