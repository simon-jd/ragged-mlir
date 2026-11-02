// RUN: ragged-opt %s -ragged-pipeline -split-input-file | FileCheck %s

// D=64 with 32-feature tiles: 4 row lanes per 128-thread CTA. The main loop
// advances 8 rows per lane (32 rows) and issues all 8 loads before the first
// compensated add; a rolled loop with the lane stride covers the remainder.
// CHECK: pointwise_fusion = false, row_pipelining = true, shuffle = false
// CHECK: gpu.func @ragged_0_sum(
// CHECK: scf.for %{{.*}} step %c32{{(_[0-9]+)?}} iter_args
// CHECK-COUNT-8: memref.load %arg0[
// CHECK-NOT: memref.load %arg0[
// CHECK: arith.subf
// CHECK: scf.for %{{.*}} step %c4 iter_args
// CHECK: memref.load %arg0[
// CHECK-NOT: memref.load %arg0[
// CHECK: gpu.barrier
func.func @pipelined(%v: memref<?x64xf32>, %o: memref<?xi64>, %r: memref<?x64xf32>) {
  ragged.segment_reduce %v, %o into %r {kind = "sum", strategy = "naive", vectorize = false} : memref<?x64xf32>, memref<?xi64>, memref<?x64xf32>
  return
}

// -----

// Disabled: one rolled loop, one load per iteration.
// CHECK: pointwise_fusion = false, row_pipelining = false, shuffle = false
// CHECK: gpu.func @ragged_0_sum(
// CHECK-NOT: step %c32
// CHECK: scf.for %{{.*}} step %c4 iter_args
// CHECK: memref.load %arg0[
// CHECK-NOT: memref.load %arg0[
// CHECK: gpu.barrier
func.func @rolled(%v: memref<?x64xf32>, %o: memref<?xi64>, %r: memref<?x64xf32>) {
  ragged.segment_reduce %v, %o into %r {kind = "sum", strategy = "naive", row_pipelining = false, vectorize = false} : memref<?x64xf32>, memref<?xi64>, memref<?x64xf32>
  return
}
