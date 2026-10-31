// RUN: ragged-opt %s --split-input-file --verify-diagnostics
func.func @reduce(%v: memref<?x7xf32>, %o: memref<?xi64>, %r: memref<?x7xf32>, %m: memref<?x7xf32>) {
  // expected-error@+1 {{additive mask applies only to softmax}}
  "ragged.segment_reduce"(%v, %o, %r, %m) {kind = "sum"} : (memref<?x7xf32>, memref<?xi64>, memref<?x7xf32>, memref<?x7xf32>) -> ()
  return
}
// -----
func.func @dtype(%v: memref<?x7xbf16>, %o: memref<?xi64>, %r: memref<?x7xbf16>, %m: memref<?x7xbf16>) {
  // expected-error@+1 {{mask must be a rank-2 f32 memref}}
  "ragged.segment_softmax"(%v, %o, %r, %m) : (memref<?x7xbf16>, memref<?xi64>, memref<?x7xbf16>, memref<?x7xbf16>) -> ()
  return
}
// -----
func.func @broadcast(%v: memref<5x7xf32>, %o: memref<?xi64>, %r: memref<5x7xf32>, %m: memref<1x7xf32>) {
  // expected-error@+1 {{mask shape must match values [N, D]; broadcasting is unsupported}}
  "ragged.segment_softmax"(%v, %o, %r, %m) : (memref<5x7xf32>, memref<?xi64>, memref<5x7xf32>, memref<1x7xf32>) -> ()
  return
}
// -----
func.func @layout(%v: memref<?x7xf32>, %o: memref<?xi64>, %r: memref<?x7xf32>, %m: memref<?x7xf32, strided<[14, 2]>>) {
  // expected-error@+1 {{mask requires a contiguous identity-layout memref}}
  "ragged.segment_softmax"(%v, %o, %r, %m) : (memref<?x7xf32>, memref<?xi64>, memref<?x7xf32>, memref<?x7xf32, strided<[14, 2]>>) -> ()
  return
}
// -----
func.func @alias(%v: memref<?x7xf32>, %o: memref<?xi64>, %r: memref<?x7xf32>) {
  // expected-error@+1 {{out must not alias mask}}
  "ragged.segment_softmax"(%v, %o, %r, %r) : (memref<?x7xf32>, memref<?xi64>, memref<?x7xf32>, memref<?x7xf32>) -> ()
  return
}
// -----
func.func @inplace_alias(%v: memref<?x7xf32>, %o: memref<?xi64>, %r: memref<?x7xf32>) {
  // expected-error@+1 {{in-place output must not alias mask through values}}
  "ragged.segment_softmax"(%v, %o, %r, %v) {alias = "inplace"} : (memref<?x7xf32>, memref<?xi64>, memref<?x7xf32>, memref<?x7xf32>) -> ()
  return
}
