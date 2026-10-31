// RUN: ragged-opt %s -ragged-plan -verify-diagnostics
func.func @unlegalized(%v: memref<?x1xf32>, %o: memref<?xi32>, %r: memref<?x1xf32>) {
  // expected-error@+1 {{run ragged-legalize before ragged-plan}}
  ragged.segment_reduce %v, %o into %r {kind = "sum"} : memref<?x1xf32>, memref<?xi32>, memref<?x1xf32>
  return
}
