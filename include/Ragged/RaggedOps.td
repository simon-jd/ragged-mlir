#ifndef RAGGED_OPS
#define RAGGED_OPS
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Ragged_Dialect : Dialect {
  let name = "ragged";
  let cppNamespace = "::ragged";
  let summary = "Explicit, contiguous segmented tensor operations";
}
class Ragged_Op<string mnemonic, list<Trait> traits = []>
    : Op<Ragged_Dialect, mnemonic, traits>;
class Ragged_DataOp<string mnemonic> : Ragged_Op<mnemonic> {
  let arguments = (ins AnyMemRef:$values, AnyMemRef:$offsets, AnyMemRef:$out,
                       Optional<AnyMemRef>:$mask);
  let assemblyFormat = "$values `,` $offsets `into` $out (`mask` $mask^ `:` type($mask))? attr-dict `:` type($values) `,` type($offsets) `,` type($out)";
  let hasVerifier = 1;
}
def Ragged_SegmentReduceOp : Ragged_DataOp<"segment_reduce"> {
  let summary = "Sum or maximum over each segment, accumulating in f32";
}
def Ragged_SegmentSoftmaxOp : Ragged_DataOp<"segment_softmax"> {
  let summary = "Stable per-feature segment softmax, accumulating in f32";
}
def Ragged_MapSegmentsOp : Ragged_DataOp<"map_segments"> {
  let summary = "Legalized segment operation with explicit semantic contract";
}
def Ragged_YieldOp : Ragged_Op<"yield", [Terminator, Pure]> {
  let assemblyFormat = "attr-dict";
}
def Ragged_DispatchOp : Ragged_Op<"dispatch", [SingleBlockImplicitTerminator<"YieldOp">]> {
  let regions = (region AnyRegion:$body);
  let assemblyFormat = "attr-dict-with-keyword $body";
  let hasVerifier = 1;
}
def Ragged_ClassifyOp : Ragged_Op<"classify"> {
  let arguments = (ins AnyMemRef:$offsets);
  let assemblyFormat = "$offsets attr-dict `:` type($offsets)";
  let hasVerifier = 1;
}
#endif
