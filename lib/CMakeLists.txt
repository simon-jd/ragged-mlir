add_mlir_dialect_library(RaggedIR IR/Ragged.cpp
  DEPENDS RaggedIncGen
  LINK_LIBS PUBLIC MLIRIR MLIRSideEffectInterfaces)
add_mlir_library(RaggedTransforms Transforms/Passes.cpp Conversion/ToGPU.cpp
  DEPENDS RaggedIncGen
  LINK_LIBS PUBLIC RaggedIR MLIRPass MLIRTransforms MLIRArithDialect
  MLIRFuncDialect MLIRGPUDialect MLIRMathDialect MLIRMemRefDialect MLIRSCFDialect)
