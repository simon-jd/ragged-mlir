// RUN: ragged-opt %s | FileCheck %s
func.func @reduce(%values: memref<?x128xbf16>, %offsets: memref<?xi32>, %out: memref<?x128xbf16>) {
  // CHECK: ragged.segment_reduce {{.*}} {kind = "sum"}
  ragged.segment_reduce %values, %offsets into %out {kind = "sum"} : memref<?x128xbf16>, memref<?xi32>, memref<?x128xbf16>
  return
}
func.func @softmax(%values: memref<?x1xf32>, %offsets: memref<?xi64>, %out: memref<?x1xf32>) {
  // CHECK: ragged.segment_softmax
  ragged.segment_softmax %values, %offsets into %out : memref<?x1xf32>, memref<?xi64>, memref<?x1xf32>
  return
}
