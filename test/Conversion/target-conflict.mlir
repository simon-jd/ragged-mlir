// RUN: not ragged-compile %s --chip=sm_86 --emit=plan -o - 2>&1 | FileCheck %s
// CHECK: the operation's chip attribute differs from --chip=sm_86
func.func @main(%v: memref<?x7xf32>, %o: memref<?xi64>, %r: memref<?x7xf32>) {
  ragged.segment_reduce %v, %o into %r {kind = "sum", chip = "sm_89"} : memref<?x7xf32>, memref<?xi64>, memref<?x7xf32>
  return
}
