// RUN: ragged-opt %s -ragged-pipeline | FileCheck %s
// CHECK: ragged.abi = "mlir-memref64-v1"
// CHECK: gpu.module @ragged_kernels
// CHECK: gpu.func @ragged_0_sum
// CHECK-SAME: workgroup
// CHECK: scf.for
// CHECK: arith.extf
// CHECK: arith.addf
// CHECK: gpu.barrier
// CHECK: arith.truncf
// CHECK-NOT: vector.transfer_read
func.func @sum(%v: memref<?x7xbf16>, %o: memref<?xi32>, %r: memref<?x7xbf16>) {
  ragged.segment_reduce %v, %o into %r {kind = "sum", strategy = "naive"} : memref<?x7xbf16>, memref<?xi32>, memref<?x7xbf16>
  return
}
