// RUN: ragged-opt %s -ragged-pipeline | FileCheck %s
// Even static B=N=0 produces the init/check kernels for offsets[0].
// CHECK: ragged.validation = {abi = "offsets-check-v1"
// CHECK-SAME: max_len, min_len,
// CHECK-LABEL: gpu.func @ragged_0_offsets_check(
// CHECK-SAME: memref<0x1xbf16>, {{.*}}memref<1xi32>, {{.*}}memref<7xi64>) kernel
// CHECK: memref.dim
// CHECK: scf.if
// CHECK: memref.load
// CHECK-NEXT: arith.extsi {{.*}} : i32 to i64
// CHECK: memref.atomic_rmw mins
// CHECK: memref.atomic_rmw mins
// CHECK: memref.atomic_rmw mins
// CHECK: memref.atomic_rmw mins
// CHECK: memref.atomic_rmw mins
// CHECK-LABEL: gpu.func @ragged_0_offsets_init
func.func @empty(%v: memref<0x1xbf16>, %o: memref<1xi32>, %r: memref<0x1xbf16>) {
  ragged.segment_reduce %v, %o into %r {kind = "sum", strategy = "naive"} : memref<0x1xbf16>, memref<1xi32>, memref<0x1xbf16>
  return
}
