// RUN: ragged-opt %s -ragged-pipeline | FileCheck %s
// CHECK-DAG: ragged.debug_visits = true
// CHECK-DAG: name = "visits", shape = ["B"]
// CHECK-DAG: name = "partials", shape = ["B_split", 56]
// CHECK-LABEL: gpu.func @ragged_0_split_combine_sum
// CHECK: gpu.grid_dim x
// CHECK: scf.for
// CHECK: memref.load {{.*}} : memref<3x?xi64>
// CHECK: scf.for {{.*}} iter_args
// CHECK: arith.subf
// CHECK: arith.addf
// CHECK: math.absf
// CHECK: arith.cmpf olt
// CHECK: arith.truncf
// CHECK: memref.store {{.*}} : memref<?x7xbf16>
// CHECK: memref.atomic_rmw addi {{.*}} : (i32, memref<?xi32>) -> i32
// CHECK: gpu.barrier
// CHECK-NEXT: }
// CHECK-LABEL: gpu.func @ragged_0_split_partial_sum
// CHECK: gpu.block_id z
// CHECK: arith.divui
// CHECK: arith.remui
// CHECK: arith.select
// CHECK: memref.store {{.*}} : memref<?x56xf32>
// CHECK: gpu.barrier
// CHECK-LABEL: gpu.func @ragged_0_warp_sum
// CHECK: gpu.shuffle down
// CHECK: gpu.shuffle idx
// CHECK: gpu.barrier
// CHECK-LABEL: gpu.func @ragged_0_classify
// CHECK: arith.cmpi ule
// CHECK: arith.cmpi ule
// CHECK: memref.atomic_rmw addi
// CHECK: memref.store {{.*}} : memref<3x?xi64>
// CHECK-LABEL: gpu.func @ragged_0_init
// CHECK: memref.store {{.*}} : memref<3xi64>
// CHECK: scf.for
// CHECK: memref.store {{.*}} : memref<?xi32>
func.func @sum(%v: memref<?x7xbf16>, %o: memref<?xi32>, %r: memref<?x7xbf16>) {
  ragged.segment_reduce %v, %o into %r {kind = "sum", debug_visits = true} : memref<?x7xbf16>, memref<?xi32>, memref<?x7xbf16>
  return
}
