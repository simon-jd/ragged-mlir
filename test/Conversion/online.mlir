// RUN: ragged-opt %s -ragged-pipeline | FileCheck %s
// CHECK: gpu.func @ragged_0_online
// CHECK: scf.for {{.*}} iter_args
// CHECK: arith.maximumf
// CHECK: math.exp
// CHECK: arith.select
// CHECK: arith.cmpf oeq
// CHECK: arith.constant 0x7FC00000
// CHECK: arith.select
// CHECK: gpu.shuffle idx
// CHECK: scf.for
// CHECK: arith.divf
// CHECK-NOT: gpu.func
func.func @softmax(%v: memref<?x16xbf16>, %o: memref<?xi32>, %r: memref<?x16xbf16>) {
  ragged.segment_softmax %v, %o into %r {max_len = 32 : i64, force_fusion = true} : memref<?x16xbf16>, memref<?xi32>, memref<?x16xbf16>
  return
}
