#ifndef RAGGED_PASSES_H
#define RAGGED_PASSES_H
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"
#include <cstdint>
#include <memory>
namespace ragged {
// A target chip and the per-CTA limits the compiler plans within (CUDA C++
// Programming Guide, compute capabilities 8.6 and 8.9): static shared memory
// in bytes, 32-bit registers and threads. ragged-plan plans within the
// smaller of these and a program's smem_per_cta and regs_budget, which
// default to them. How many CTAs are resident and the L2 size depend on the
// device, so the runtime queries them.
struct ChipLimits {
  llvm::StringRef chip;
  int64_t sharedPerCta;
  int64_t registersPerCta;
  int64_t threadsPerCta;
};
inline constexpr llvm::StringRef kDefaultChip = "sm_89";
inline constexpr ChipLimits kChips[] = {{"sm_86", 49152, 65536, 1024}, {"sm_89", 49152, 65536, 1024}};
inline const ChipLimits *lookupChip(llvm::StringRef chip) {
  for (const ChipLimits &limits : kChips)
    if (limits.chip == chip) return &limits;
  return nullptr;
}

std::unique_ptr<mlir::Pass> createLegalizePass();
std::unique_ptr<mlir::Pass> createPlanPass();
std::unique_ptr<mlir::Pass> createFusePass();
std::unique_ptr<mlir::Pass> createToGPUPass();
void registerRaggedPasses();
}
#endif
