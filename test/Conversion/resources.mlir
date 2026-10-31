// REQUIRES: cuda-ptxas
// RUN: %python %S/../Inputs/verify-resources.py ragged-compile %s %t
func.func @sum(%v: memref<?x128xf32>, %o: memref<?xi64>, %r: memref<?x128xf32>) {
  ragged.segment_reduce %v, %o into %r {kind = "sum", strategy = "naive", regs_budget = 65536 : i64, smem_per_cta = 49152 : i64} : memref<?x128xf32>, memref<?xi64>, memref<?x128xf32>
  return
}
