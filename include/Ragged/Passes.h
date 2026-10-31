#ifndef RAGGED_PASSES_H
#define RAGGED_PASSES_H
#include "mlir/Pass/Pass.h"
#include <memory>
namespace ragged {
std::unique_ptr<mlir::Pass> createLegalizePass();
std::unique_ptr<mlir::Pass> createPlanPass();
std::unique_ptr<mlir::Pass> createFusePass();
std::unique_ptr<mlir::Pass> createToGPUPass();
void registerRaggedPasses();
}
#endif
