// RUN: ragged-opt %s -ragged-pipeline -mlir-print-op-generic | FileCheck %s
// The kernel marker belongs to inherent properties. A legacy discardable
// gpu.kernel is duplicated by LLVM 23 GPUFuncOpLowering under assertions.
// CHECK: "gpu.func"() <{function_type = {{.*}}, kernel,
// CHECK-NOT: gpu.kernel
// CHECK: "gpu.return"
// CHECK-NOT: gpu.kernel
func.func @sum(%v: memref<?x1xf32>, %o: memref<?xi64>, %r: memref<?x1xf32>) {
  ragged.segment_reduce %v, %o into %r {kind = "sum", strategy = "naive"} : memref<?x1xf32>, memref<?xi64>, memref<?x1xf32>
  return
}
