// RUN: ragged-opt %s -ragged-legalize -ragged-plan -ragged-fuse -verify-diagnostics
func.func @resource(%v: memref<?x16xf32>, %o: memref<?xi64>, %r: memref<?x16xf32>) {
  // expected-error@+1 {{forced fusion is illegal: online statistics exceed shared-memory budget}}
  ragged.segment_softmax %v, %o into %r {strategy = "naive", max_len = 128 : i64, smem_per_cta = 512 : i64, force_fusion = true} : memref<?x16xf32>, memref<?xi64>, memref<?x16xf32>
  return
}
