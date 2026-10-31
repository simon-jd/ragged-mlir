// RUN: ragged-opt %s -ragged-pipeline | FileCheck %s
// An offsets_checked promise does not suppress validation or static hints.
// CHECK: ragged.validation = {abi = "offsets-check-v1"
// CHECK-SAME: conditions = ["first_nonzero", "final_mismatch", "negative", "above_n", "nonmonotone", "below_min_len", "above_max_len"]
// CHECK-SAME: grid = ["ceil_B1_128", 1, 1]
// CHECK-SAME: max_len = 4096 : i64, min_len = 1 : i64, sentinel = 9223372036854775807 : i64
// CHECK-LABEL: gpu.func @ragged_0_offsets_check(
// CHECK-SAME: %[[VALUES:[a-z0-9]+]]: memref<?x7xf32>, %[[OFFSETS:[a-z0-9]+]]: memref<?xi64>, %[[STATUS:[a-z0-9]+]]: memref<7xi64>) kernel
// CHECK: %[[COUNT:.*]] = memref.dim %[[OFFSETS]]
// CHECK: %[[ACTIVE:.*]] = arith.cmpi ult, {{.*}}, %[[COUNT]]
// CHECK: scf.if %[[ACTIVE]]
// CHECK: %[[NINDEX:.*]] = memref.dim %[[VALUES]]
// CHECK: %[[N:.*]] = arith.index_cast %[[NINDEX]] : index to i64
// CHECK: %[[ZERO:.*]] = arith.constant 0 : i64
// CHECK: %[[CURRENT:.*]] = memref.load %[[OFFSETS]]
// CHECK: arith.cmpi ne, %[[CURRENT]]
// CHECK: memref.atomic_rmw mins {{.*}} %[[STATUS]]
// CHECK: arith.cmpi ne, %[[CURRENT]]
// CHECK: memref.atomic_rmw mins {{.*}} %[[STATUS]]
// CHECK: arith.cmpi slt, %[[CURRENT]]
// CHECK: arith.cmpi sgt, %[[CURRENT]]
// CHECK: memref.atomic_rmw mins {{.*}} %[[STATUS]]
// CHECK: memref.atomic_rmw mins {{.*}} %[[STATUS]]
// CHECK: %[[HASNEXT:.*]] = arith.cmpi ult
// CHECK: scf.if %[[HASNEXT]]
// CHECK: %[[NEXT:.*]] = memref.load %[[OFFSETS]]
// CHECK: arith.cmpi slt, %[[NEXT]], %[[CURRENT]]
// CHECK: memref.atomic_rmw mins {{.*}} %[[STATUS]]
// C++ argument evaluation may emit each independent bound pair in either order.
// Bind every predicate and conjunction so the full guard must protect subtraction.
// CHECK-DAG: %[[CURRENTLOW:.*]] = arith.cmpi sge, %[[CURRENT]], %[[ZERO]] : i64
// CHECK-DAG: %[[CURRENTHIGH:.*]] = arith.cmpi sle, %[[CURRENT]], %[[N]] : i64
// CHECK: %[[CURRENTVALID:.*]] = arith.andi %[[CURRENTLOW]], %[[CURRENTHIGH]] : i1
// CHECK-DAG: %[[NEXTLOW:.*]] = arith.cmpi sge, %[[NEXT]], %[[ZERO]] : i64
// CHECK-DAG: %[[NEXTHIGH:.*]] = arith.cmpi sle, %[[NEXT]], %[[N]] : i64
// CHECK: %[[NEXTVALID:.*]] = arith.andi %[[NEXTLOW]], %[[NEXTHIGH]] : i1
// CHECK: %[[MONOTONE:.*]] = arith.cmpi sge, %[[NEXT]], %[[CURRENT]] : i64
// CHECK: %[[RANGE:.*]] = arith.andi %[[CURRENTVALID]], %[[NEXTVALID]] : i1
// CHECK-NEXT: %[[SAFE:.*]] = arith.andi %[[RANGE]], %[[MONOTONE]] : i1
// CHECK-NEXT: scf.if %[[SAFE]]
// CHECK-NEXT: %[[LENGTH:.*]] = arith.subi %[[NEXT]], %[[CURRENT]] : i64
// CHECK: arith.cmpi slt, %[[LENGTH]]
// CHECK: memref.atomic_rmw mins {{.*}} %[[STATUS]]
// CHECK: arith.cmpi sgt, %[[LENGTH]]
// CHECK: memref.atomic_rmw mins {{.*}} %[[STATUS]]
// CHECK-LABEL: gpu.func @ragged_0_offsets_init(
// CHECK: arith.constant 7 : index
// CHECK: arith.cmpi ult
// CHECK: scf.if
// CHECK: arith.constant 9223372036854775807 : i64
// CHECK: memref.store {{.*}} : memref<7xi64>
func.func @softmax(%v: memref<?x7xf32>, %o: memref<?xi64>, %r: memref<?x7xf32>) {
  ragged.segment_softmax %v, %o into %r {min_len = 1 : i64, max_len = 4096 : i64, offsets_checked = true} : memref<?x7xf32>, memref<?xi64>, memref<?x7xf32>
  return
}
