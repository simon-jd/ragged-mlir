// RUN: ragged-opt %s -ragged-pipeline -split-input-file | FileCheck %s
// Each thread owns the largest power-of-two run of adjacent features that
// divides D and fits 16 bytes: four f32, eight bf16. Rows are read and
// outputs written with one aligned vector access. A feature tile still spans
// 32 features, so D=128 f32 has four tiles of 8 threads and 16 row lanes.
// CHECK: ragged.d_tile = 8 : i64
// CHECK-SAME: ragged.d_vector = 4 : i64
// CHECK: grid = ["B", 4, 1]
// CHECK: vectorize = true
// CHECK-LABEL: gpu.func @ragged_0_sum
// CHECK: memref<512xf32, #gpu.address_space<workgroup>>
// CHECK: vector.load {{.*}} {alignment = 16 : i64} : memref<?x128xf32>, vector<4xf32>
// CHECK: vector.store {{.*}} {alignment = 16 : i64} : memref<?x128xf32>, vector<4xf32>
func.func @sum(%v: memref<?x128xf32>, %o: memref<?xi64>, %r: memref<?x128xf32>) {
  ragged.segment_reduce %v, %o into %r {kind = "sum", strategy = "naive"} : memref<?x128xf32>, memref<?xi64>, memref<?x128xf32>
  return
}

// -----
// CHECK: ragged.d_tile = 4 : i64
// CHECK-SAME: ragged.d_vector = 8 : i64
// CHECK-LABEL: gpu.func @ragged_0_online
// CHECK: vector.load {{.*}} {alignment = 16 : i64} : memref<?x128xbf16>, vector<8xbf16>
// CHECK: vector.store {{.*}} {alignment = 16 : i64} : memref<?x128xbf16>, vector<8xbf16>
func.func @softmax(%v: memref<?x128xbf16>, %o: memref<?xi64>, %r: memref<?x128xbf16>) {
  ragged.segment_softmax %v, %o into %r {strategy = "naive"} : memref<?x128xbf16>, memref<?xi64>, memref<?x128xbf16>
  return
}

// -----
// D=6 takes pairs; D=7 and the explicit control keep one feature per thread.
// CHECK: ragged.d_tile = 4 : i64
// CHECK-SAME: ragged.d_vector = 2 : i64
// CHECK: vector.load {{.*}} {alignment = 8 : i64} : memref<?x6xf32>, vector<2xf32>
func.func @pairs(%v: memref<?x6xf32>, %o: memref<?xi64>, %r: memref<?x6xf32>) {
  ragged.segment_reduce %v, %o into %r {kind = "max", strategy = "naive"} : memref<?x6xf32>, memref<?xi64>, memref<?x6xf32>
  return
}

// -----
// CHECK: ragged.d_vector = 1 : i64
// CHECK-NOT: vector.load
func.func @odd(%v: memref<?x7xf32>, %o: memref<?xi64>, %r: memref<?x7xf32>) {
  ragged.segment_reduce %v, %o into %r {kind = "sum", strategy = "naive"} : memref<?x7xf32>, memref<?xi64>, memref<?x7xf32>
  return
}

// -----
// CHECK: ragged.d_tile = 32 : i64
// CHECK-SAME: ragged.d_vector = 1 : i64
// CHECK: grid = ["B", 4, 1]
// CHECK: vectorize = false
// CHECK-NOT: vector.load
func.func @scalar(%v: memref<?x128xf32>, %o: memref<?xi64>, %r: memref<?x128xf32>) {
  ragged.segment_reduce %v, %o into %r {kind = "sum", strategy = "naive", vectorize = false} : memref<?x128xf32>, memref<?xi64>, memref<?x128xf32>
  return
}
