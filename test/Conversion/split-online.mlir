// RUN: ragged-opt %s -ragged-pipeline | FileCheck %s
// RUN: ragged-opt %s -ragged-pipeline | FileCheck %s --check-prefix=ONLINE
// Without hints every class uses online statistics. Each split part writes its
// reference and mass; a combine kernel merges the eight parts before the
// normalization kernel reads the merged statistics.
// CHECK-DAG: ragged.dispatch_abi = "classified-v3"
// CHECK-DAG: name = "partials", shape = ["B_split", 1024]
// CHECK-DAG: name = "partial_sums", shape = ["B_split", 1024]
// CHECK-DAG: name = "max_stats", shape = ["B", 128]
// CHECK-DAG: name = "sum_stats", shape = ["B", 128]
// CHECK-DAG: role = "warp_online"
// CHECK-DAG: role = "cta_online"
// CHECK-DAG: grid = ["min_B_resident", 4, 8], name = "ragged_0_split_partial_online"
// CHECK-DAG: grid = ["min_B_resident", 4, 1], name = "ragged_0_split_combine_online"
// CHECK-DAG: grid = ["min_B_resident", 4, 8], name = "ragged_0_split_normalize"
// CHECK-LABEL: gpu.func @ragged_0_split_normalize
// CHECK: arith.divf
// CHECK-LABEL: gpu.func @ragged_0_split_combine_online
// CHECK: scf.for {{.*}} iter_args
// CHECK: arith.maximumf
// CHECK: scf.for {{.*}} iter_args
// CHECK: math.exp
// CHECK: arith.cmpf oeq
// CHECK: memref.store {{.*}} : memref<?x128xf32>
// CHECK: memref.store {{.*}} : memref<?x128xf32>
// CHECK-LABEL: gpu.func @ragged_0_split_partial_online
// CHECK: memref<256xf32, #gpu.address_space<workgroup>>
// CHECK: gpu.block_id z
// CHECK: math.exp
// CHECK: gpu.barrier
// CHECK: memref.store {{.*}} : memref<?x1024xf32>
// CHECK: memref.store {{.*}} : memref<?x1024xf32>
// CHECK-LABEL: gpu.func @ragged_0_cta_online
// CHECK: memref<256xf32, #gpu.address_space<workgroup>>
// CHECK-LABEL: gpu.func @ragged_0_warp_online
// CHECK: gpu.shuffle idx
// ONLINE-NOT: role = "{{[a-z]+}}_{{(max_stats|sum_stats|partial_max|partial_exp_sum|combine_max_stats|combine_sum_stats)}}"
func.func @long_tail(%v: memref<?x128xf32>, %o: memref<?xi64>, %r: memref<?x128xf32>) {
  ragged.segment_softmax %v, %o into %r : memref<?x128xf32>, memref<?xi64>, memref<?x128xf32>
  return
}
