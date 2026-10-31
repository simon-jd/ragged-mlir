#include "Ragged/Passes.h"
#include "Ragged/Ragged.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/TargetSelect.h"

int main(int argc, char **argv) {
  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmPrinters();
  mlir::registerAllPasses();
  ragged::registerRaggedPasses();
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  mlir::registerAllToLLVMIRTranslations(registry);
  registry.insert<ragged::RaggedDialect>();
  return mlir::asMainReturnCode(mlir::MlirOptMain(argc, argv,
      "Ragged segmented tensor optimizer\n", registry));
}
