// RUN: ragged-opt %s --split-input-file --verify-diagnostics
func.func @checked_type(%v: memref<?x1xf32>, %o: memref<?xi64>, %r: memref<?x1xf32>) {
  // expected-error@+1 {{offsets_checked must be a boolean}}
  ragged.segment_reduce %v, %o into %r {kind = "sum", offsets_checked = "true"} : memref<?x1xf32>, memref<?xi64>, memref<?x1xf32>
  return
}
// -----
func.func @no_initial_offset(%v: memref<0x1xf32>, %o: memref<0xi64>, %r: memref<0x1xf32>) {
  // expected-error@+1 {{offsets must have B+1 entries, including the initial zero}}
  ragged.segment_softmax %v, %o into %r : memref<0x1xf32>, memref<0xi64>, memref<0x1xf32>
  return
}
// -----
func.func @negative_hint(%v: memref<?x1xf32>, %o: memref<?xi64>, %r: memref<?x1xf32>) {
  // expected-error@+1 {{min_len must be a nonnegative i64 integer}}
  ragged.segment_softmax %v, %o into %r {min_len = -1 : i64, offsets_checked = true} : memref<?x1xf32>, memref<?xi64>, memref<?x1xf32>
  return
}
