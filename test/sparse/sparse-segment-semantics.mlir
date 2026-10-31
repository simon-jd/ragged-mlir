// CPU feasibility probe, not a ragged compiler kernel or GPU performance claim.
// Reproduce with scripts/run_sparse_probe.py. Explicit CSR assembly preserves
// stored zero, which dense-to-sparse conversion is allowed to discard.
#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed),
  posWidth = 32, crdWidth = 32
}>
#reduce = {
  indexing_maps = [affine_map<(i,j) -> (i,j)>, affine_map<(i,j) -> (i)>],
  iterator_types = ["parallel", "reduction"]
}
#broadcast = {
  indexing_maps = [affine_map<(i,j) -> (i,j)>, affine_map<(i,j) -> (i)>, affine_map<(i,j) -> (i,j)>],
  iterator_types = ["parallel", "parallel"]
}

module {
  func.func @segment_max(%a: tensor<6x4xf64, #CSR>) -> tensor<6xf64> {
    %identity = arith.constant 0xFFF0000000000000 : f64
    %empty = tensor.empty() : tensor<6xf64>
    %init = linalg.fill ins(%identity : f64) outs(%empty : tensor<6xf64>) -> tensor<6xf64>
    %max = linalg.generic #reduce ins(%a : tensor<6x4xf64, #CSR>) outs(%init : tensor<6xf64>) {
      ^bb0(%x: f64, %acc: f64):
        %present = sparse_tensor.unary %x : f64 to f64
          present={ ^bb1(%v: f64): sparse_tensor.yield %v : f64 }
          absent={}
        %next = sparse_tensor.reduce %present, %acc, %identity : f64 {
          ^bb2(%lhs: f64, %rhs: f64):
            %value = arith.maximumf %lhs, %rhs : f64
            sparse_tensor.yield %value : f64
        }
        linalg.yield %next : f64
    } -> tensor<6xf64>
    return %max : tensor<6xf64>
  }

  func.func @softmax(%a: tensor<6x4xf64, #CSR>, %max: tensor<6xf64>) -> tensor<6x4xf64, #CSR> {
    %empty = tensor.empty() : tensor<6x4xf64, #CSR>
    %weights = linalg.generic #broadcast ins(%a, %max : tensor<6x4xf64, #CSR>, tensor<6xf64>) outs(%empty : tensor<6x4xf64, #CSR>) {
      ^bb0(%x: f64, %m: f64, %unused: f64):
        %weight = sparse_tensor.unary %x : f64 to f64
          present={
            ^bb1(%v: f64):
              %shift = arith.subf %v, %m : f64
              %e = math.exp %shift : f64
              sparse_tensor.yield %e : f64
          }
          absent={}
        linalg.yield %weight : f64
    } -> tensor<6x4xf64, #CSR>
    %zero = arith.constant 0.0 : f64
    %sum_empty = tensor.empty() : tensor<6xf64>
    %sum_init = linalg.fill ins(%zero : f64) outs(%sum_empty : tensor<6xf64>) -> tensor<6xf64>
    %total = linalg.generic #reduce ins(%weights : tensor<6x4xf64, #CSR>) outs(%sum_init : tensor<6xf64>) {
      ^bb0(%x: f64, %acc: f64):
        %next = sparse_tensor.reduce %x, %acc, %zero : f64 {
          ^bb1(%lhs: f64, %rhs: f64):
            %value = arith.addf %lhs, %rhs : f64
            sparse_tensor.yield %value : f64
        }
        linalg.yield %next : f64
    } -> tensor<6xf64>
    %out_empty = tensor.empty() : tensor<6x4xf64, #CSR>
    %out = linalg.generic #broadcast ins(%weights, %total : tensor<6x4xf64, #CSR>, tensor<6xf64>) outs(%out_empty : tensor<6x4xf64, #CSR>) {
      ^bb0(%x: f64, %sum: f64, %unused: f64):
        %prob = sparse_tensor.unary %x : f64 to f64
          present={
            ^bb1(%v: f64):
              %p = arith.divf %v, %sum : f64
              sparse_tensor.yield %p : f64
          }
          absent={}
        linalg.yield %prob : f64
    } -> tensor<6x4xf64, #CSR>
    bufferization.dealloc_tensor %weights : tensor<6x4xf64, #CSR>
    bufferization.dealloc_tensor %total : tensor<6xf64>
    return %out : tensor<6x4xf64, #CSR>
  }

  func.func @main() {
    %positions = arith.constant dense<[0, 2, 2, 4, 6, 8, 10]> : tensor<7xi32>
    %coordinates = arith.constant dense<[0, 1, 0, 1, 0, 1, 0, 1, 0, 1]> : tensor<10xi32>
    %values = arith.constant dense<[-2.0, -4.0, 0x7FF8000000000000, 1.0, 0xFFF0000000000000, 0xFFF0000000000000, 0.0, 2.0, 0x7FF0000000000000, 1.0]> : tensor<10xf64>
    %input = sparse_tensor.assemble (%positions, %coordinates), %values : (tensor<7xi32>, tensor<10xi32>), tensor<10xf64> to tensor<6x4xf64, #CSR>
    %max = call @segment_max(%input) : (tensor<6x4xf64, #CSR>) -> tensor<6xf64>
    %out = call @softmax(%input, %max) : (tensor<6x4xf64, #CSR>, tensor<6xf64>) -> tensor<6x4xf64, #CSR>
    %c0 = arith.constant 0 : index
    %pad = arith.constant 0.0 : f64
    %max_vector = vector.transfer_read %max[%c0], %pad : tensor<6xf64>, vector<6xf64>
    vector.print %max_vector : vector<6xf64>
    sparse_tensor.print %input : tensor<6x4xf64, #CSR>
    sparse_tensor.print %out : tensor<6x4xf64, #CSR>
    // assemble borrows the constant buffers in direct-codegen mode; the
    // runtime-library path copies them. Match upstream sparse_pack ownership.
    %has_runtime = sparse_tensor.has_runtime_library
    scf.if %has_runtime {
      bufferization.dealloc_tensor %input : tensor<6x4xf64, #CSR>
    }
    bufferization.dealloc_tensor %max : tensor<6xf64>
    bufferization.dealloc_tensor %out : tensor<6x4xf64, #CSR>
    return
  }
}
