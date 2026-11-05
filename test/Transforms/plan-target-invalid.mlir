// RUN: ragged-opt %s -ragged-legalize -ragged-plan -verify-diagnostics
func.func @sm90(%v: memref<?x1xf32>, %o: memref<?xi32>, %r: memref<?x1xf32>) {
  // expected-error@+1 {{target must be sm_86 or sm_89}}
  ragged.segment_reduce %v, %o into %r {kind = "sum", chip = "sm_90"} : memref<?x1xf32>, memref<?xi32>, memref<?x1xf32>
  return
}
