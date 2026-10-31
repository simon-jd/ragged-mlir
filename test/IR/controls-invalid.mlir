// RUN: ragged-opt %s --split-input-file --verify-diagnostics
func.func @typed(%v: memref<?x7xf32>, %o: memref<?xi64>, %r: memref<?x7xf32>) {
  // expected-error@+1 {{shuffle must be a boolean}}
  ragged.segment_softmax %v, %o into %r {shuffle = 0 : i64} : memref<?x7xf32>, memref<?xi64>, memref<?x7xf32>
  return
}
// -----
func.func @tile(%v: memref<?x7xf32>, %o: memref<?xi64>, %r: memref<?x7xf32>) {
  // expected-error@+1 {{feature_tile must be an i64 power of two in [1, 32]}}
  ragged.segment_softmax %v, %o into %r {feature_tile = 3 : i64} : memref<?x7xf32>, memref<?xi64>, memref<?x7xf32>
  return
}
// -----
func.func @force(%v: memref<?x7xf32>, %o: memref<?xi64>, %r: memref<?x7xf32>) {
  // expected-error@+1 {{force_fusion conflicts with online_fusion = false}}
  ragged.segment_softmax %v, %o into %r {force_fusion = true, online_fusion = false} : memref<?x7xf32>, memref<?xi64>, memref<?x7xf32>
  return
}
// -----
func.func @debug(%v: memref<?x7xf32>, %o: memref<?xi64>, %r: memref<?x7xf32>) {
  // expected-error@+1 {{debug_visits conflicts with classification = false}}
  ragged.segment_softmax %v, %o into %r {debug_visits = true, classification = false} : memref<?x7xf32>, memref<?xi64>, memref<?x7xf32>
  return
}
