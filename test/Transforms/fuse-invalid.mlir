// RUN: ragged-opt %s -ragged-legalize -ragged-plan -ragged-fuse -verify-diagnostics
func.func @inplace(%v: memref<?x16xf32>, %o: memref<?xi64>) {
  // expected-error@+1 {{forced fusion is illegal: in-place output requires kernel-separated read and write phases}}
  ragged.segment_softmax %v, %o into %v {max_len = 32 : i64, force_fusion = true} : memref<?x16xf32>, memref<?xi64>, memref<?x16xf32>
  return
}
