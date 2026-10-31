// RUN: ragged-opt %s -ragged-legalize -ragged-plan -ragged-fuse | FileCheck %s
func.func @short(%v: memref<?x16xbf16>, %o: memref<?xi32>, %r: memref<?x16xbf16>) {
  // CHECK: fusion = "online"
  ragged.segment_softmax %v, %o into %r {max_len = 32 : i64} : memref<?x16xbf16>, memref<?xi32>, memref<?x16xbf16>
  return
}
func.func @inplace(%v: memref<?x16xf32>, %o: memref<?xi64>) {
  // CHECK: alias = "inplace"
  // CHECK-SAME: fusion = "three_pass"
  // CHECK-SAME: fusion_reason = "in-place output requires kernel-separated read and write phases"
  ragged.segment_softmax %v, %o into %v {max_len = 32 : i64} : memref<?x16xf32>, memref<?xi64>, memref<?x16xf32>
  return
}
func.func @resource_fallback(%v: memref<?x16xf32>, %o: memref<?xi64>, %r: memref<?x16xf32>) {
  // CHECK: shared_bytes = 512 : i64
  // CHECK: fusion = "three_pass"
  // CHECK-SAME: fusion_reason = "online statistics exceed shared-memory budget"
  ragged.segment_softmax %v, %o into %r {strategy = "naive", max_len = 128 : i64, smem_per_cta = 512 : i64} : memref<?x16xf32>, memref<?xi64>, memref<?x16xf32>
  return
}
func.func @long_tail(%v: memref<?x16xf32>, %o: memref<?xi64>, %r: memref<?x16xf32>) {
  // Online statistics need no length bound; split parts merge in a combine kernel.
  // CHECK: class_fusion = ["online", "online", "online"]
  // CHECK-SAME: fusion = "online"
  // CHECK-SAME: fusion_reason = "disjoint output permits online statistics"
  ragged.segment_softmax %v, %o into %r : memref<?x16xf32>, memref<?xi64>, memref<?x16xf32>
  return
}
func.func @forced_with_split(%v: memref<?x16xf32>, %o: memref<?xi64>, %r: memref<?x16xf32>) {
  // CHECK: class_fusion = ["online"]
  // CHECK-SAME: fusion = "online"
  ragged.segment_softmax %v, %o into %r {min_len = 4097 : i64, force_fusion = true} : memref<?x16xf32>, memref<?xi64>, memref<?x16xf32>
  return
}
func.func @cta_takes_the_remainder(%v: memref<?x16xf32>, %o: memref<?xi64>, %r: memref<?x16xf32>) {
  // CHECK: class_fusion = ["online", "online"]
  // CHECK-SAME: fusion = "online"
  ragged.segment_softmax %v, %o into %r {split_mapping = false} : memref<?x16xf32>, memref<?xi64>, memref<?x16xf32>
  return
}
func.func @shared_memory_per_class(%v: memref<?x16xf32>, %o: memref<?xi64>, %r: memref<?x16xf32>) {
  // A shuffle warp needs no shared memory; the 128-thread trees need 1 KiB.
  // CHECK: class_fusion = ["online", "three_pass", "three_pass"]
  // CHECK-SAME: fusion = "mixed"
  // CHECK-SAME: fusion_reason = "warp: disjoint output permits online statistics; cta, split: online statistics exceed shared-memory budget"
  ragged.segment_softmax %v, %o into %r {smem_per_cta = 512 : i64} : memref<?x16xf32>, memref<?xi64>, memref<?x16xf32>
  return
}
func.func @inplace_long_tail(%v: memref<?x16xf32>, %o: memref<?xi64>) {
  // CHECK: alias = "inplace"
  // CHECK-SAME: class_fusion = ["three_pass", "three_pass", "three_pass"]
  // CHECK-SAME: fusion = "three_pass"
  // CHECK-SAME: fusion_reason = "in-place output requires kernel-separated read and write phases"
  ragged.segment_softmax %v, %o into %v : memref<?x16xf32>, memref<?xi64>, memref<?x16xf32>
  return
}
func.func @disabled_long_tail(%v: memref<?x16xf32>, %o: memref<?xi64>, %r: memref<?x16xf32>) {
  // CHECK: class_fusion = ["three_pass", "three_pass", "three_pass"]
  // CHECK-SAME: fusion = "three_pass"
  // CHECK-SAME: fusion_reason = "online statistics fusion disabled"
  ragged.segment_softmax %v, %o into %r {online_fusion = false} : memref<?x16xf32>, memref<?xi64>, memref<?x16xf32>
  return
}
