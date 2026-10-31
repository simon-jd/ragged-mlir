// RUN: ragged-opt %s -split-input-file -verify-diagnostics
func.func @dynamic_d(%v: memref<?x?xf32>, %o: memref<?xi32>, %r: memref<?x?xf32>) {
  // expected-error@+1 {{D must be static and in [1, 1024]}}
  ragged.segment_reduce %v, %o into %r {kind = "sum"} : memref<?x?xf32>, memref<?xi32>, memref<?x?xf32>
  return
}
// -----
func.func @dtype(%v: memref<?x16xf16>, %o: memref<?xi64>, %r: memref<?x16xf16>) {
  // expected-error@+1 {{values dtype must be f32 or bf16}}
  ragged.segment_softmax %v, %o into %r : memref<?x16xf16>, memref<?xi64>, memref<?x16xf16>
  return
}
// -----
func.func @offset_type(%v: memref<?x1xf32>, %o: memref<?xi16>, %r: memref<?x1xf32>) {
  // expected-error@+1 {{offsets dtype must be i32 or i64}}
  ragged.segment_reduce %v, %o into %r {kind = "max"} : memref<?x1xf32>, memref<?xi16>, memref<?x1xf32>
  return
}
// -----
func.func @shape(%v: memref<5x1xf32>, %o: memref<3xi32>, %r: memref<5x1xf32>) {
  // expected-error@+1 {{reduce out leading dimension must equal B}}
  ragged.segment_reduce %v, %o into %r {kind = "sum"} : memref<5x1xf32>, memref<3xi32>, memref<5x1xf32>
  return
}
// -----
func.func @alias(%v: memref<?x1xf32>, %o: memref<?xi64>) {
  // expected-error@+1 {{in-place reduction is unsafe}}
  ragged.segment_reduce %v, %o into %v {kind = "sum"} : memref<?x1xf32>, memref<?xi64>, memref<?x1xf32>
  return
}
// -----
func.func @hints(%v: memref<?x1xf32>, %o: memref<?xi64>, %r: memref<?x1xf32>) {
  // expected-error@+1 {{min_len must not exceed max_len}}
  ragged.segment_softmax %v, %o into %r {min_len = 64 : i64, max_len = 32 : i64} : memref<?x1xf32>, memref<?xi64>, memref<?x1xf32>
  return
}
// -----
func.func @wide_hint(%v: memref<?x1xf32>, %o: memref<?xi64>, %r: memref<?x1xf32>) {
  // expected-error@+1 {{max_len must be a nonnegative i64 integer}}
  ragged.segment_softmax %v, %o into %r {max_len = 18446744073709551616 : i128} : memref<?x1xf32>, memref<?xi64>, memref<?x1xf32>
  return
}
// -----
func.func @layout(%v: memref<?x1xf32, strided<[2, 1]>>, %o: memref<?xi64>, %r: memref<?x1xf32>) {
  // expected-error@+1 {{requires contiguous identity-layout memrefs}}
  ragged.segment_softmax %v, %o into %r : memref<?x1xf32, strided<[2, 1]>>, memref<?xi64>, memref<?x1xf32>
  return
}
