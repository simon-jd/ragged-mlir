#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ragged {

// One kernel launch of a launch plan. Lane 0 is the caller's stream; lane
// k > 0 is the caller stream's k-th side stream.
struct PlanStep {
  std::string kernel;
  std::array<unsigned, 3> grid, block;
  unsigned sharedBytes, lane;
};

// No CUDA runtime or LLVM dependency. The driver library is resolved at runtime.
// Arguments are the flattened, 64-bit-index MLIR memref descriptor ABI.
class DriverModule {
public:
  explicit DriverModule(const std::string &cubin);
  ~DriverModule();
  DriverModule(const DriverModule &) = delete;
  DriverModule &operator=(const DriverModule &) = delete;

  void launch(const std::string &kernel, const std::array<unsigned, 3> &grid,
              const std::array<unsigned, 3> &block, unsigned sharedBytes,
              std::uintptr_t stream, const std::vector<std::uint64_t> &arguments);
  int device() const;
  // Blocks of this kernel that can be resident on the device at once: the
  // occupancy limit per multiprocessor times the multiprocessor count.
  int residentBlocks(const std::string &kernel, unsigned blockThreads, unsigned sharedBytes);
  // A launch plan fixes the kernels, grids and lanes of one problem size,
  // and the scratch descriptors with the indices of their pointer fields,
  // which hold offsets from the scratch base. Running it launches every
  // step with the inputs' descriptors followed by the relocated scratch
  // descriptors as arguments: side lanes are forked from the caller's
  // stream before their first step and joined to it after the last.
  int addPlan(const std::vector<PlanStep> &steps, const std::vector<std::uint64_t> &scratch,
              const std::vector<std::uint32_t> &relocations);
  void runPlan(int plan, std::uintptr_t stream, const std::vector<std::uint64_t> &inputs,
               std::uint64_t scratchBase);
  void clearPlans();
  void close();

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};

} // namespace ragged
