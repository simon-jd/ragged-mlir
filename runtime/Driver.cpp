#include "Driver.h"

#include <algorithm>
#include <cstddef>
#include <dlfcn.h>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace ragged {
namespace {
using CUresult = int;
using CUdevice = int;
using CUcontext = void *;
using CUmodule = void *;
using CUfunction = void *;
using CUstream = void *;
using CUevent = void *;

struct Driver {
  void *library = nullptr;
  CUresult (*init)(unsigned) = nullptr;
  CUresult (*getCurrent)(CUcontext *) = nullptr;
  CUresult (*getDevice)(CUdevice *) = nullptr;
  CUresult (*getAttribute)(int *, int, CUdevice) = nullptr;
  CUresult (*moduleLoad)(CUmodule *, const void *) = nullptr;
  CUresult (*moduleUnload)(CUmodule) = nullptr;
  CUresult (*getFunction)(CUfunction *, CUmodule, const char *) = nullptr;
  CUresult (*launch)(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned,
                     unsigned, unsigned, CUstream, void **, void **) = nullptr;
  CUresult (*occupancy)(int *, CUfunction, int, std::size_t) = nullptr;
  CUresult (*streamCreate)(CUstream *, unsigned, int) = nullptr;
  CUresult (*streamPriority)(CUstream, int *) = nullptr;
  CUresult (*streamWait)(CUstream, CUevent, unsigned) = nullptr;
  CUresult (*eventCreate)(CUevent *, unsigned) = nullptr;
  CUresult (*eventRecord)(CUevent, CUstream) = nullptr;
  CUresult (*getErrorName)(CUresult, const char **) = nullptr;
  CUresult (*getErrorString)(CUresult, const char **) = nullptr;
  CUresult (*pushCurrent)(CUcontext) = nullptr;
  CUresult (*popCurrent)(CUcontext *) = nullptr;

  template <typename T> void symbol(T &target, const char *name) {
    target = reinterpret_cast<T>(dlsym(library, name));
    if (!target)
      throw std::runtime_error(std::string("CUDA driver is missing ") + name);
  }

  Driver() {
    library = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!library)
      throw std::runtime_error("CUDA Driver API unavailable: cannot load libcuda.so.1");
    try {
      symbol(init, "cuInit");
      symbol(getCurrent, "cuCtxGetCurrent");
      symbol(getDevice, "cuCtxGetDevice");
      symbol(getAttribute, "cuDeviceGetAttribute");
      symbol(moduleLoad, "cuModuleLoadData");
      symbol(moduleUnload, "cuModuleUnload");
      symbol(getFunction, "cuModuleGetFunction");
      symbol(launch, "cuLaunchKernel");
      symbol(occupancy, "cuOccupancyMaxActiveBlocksPerMultiprocessor");
      symbol(streamCreate, "cuStreamCreateWithPriority");
      symbol(streamPriority, "cuStreamGetPriority");
      symbol(streamWait, "cuStreamWaitEvent");
      symbol(eventCreate, "cuEventCreate");
      symbol(eventRecord, "cuEventRecord");
      symbol(getErrorName, "cuGetErrorName");
      symbol(getErrorString, "cuGetErrorString");
      symbol(pushCurrent, "cuCtxPushCurrent_v2");
      symbol(popCurrent, "cuCtxPopCurrent_v2");
      check(init(0), "cuInit");
    } catch (...) {
      dlclose(library);
      library = nullptr;
      throw;
    }
  }

  // The library stays loaded for the process: modules may be unloaded from
  // destructors that run during interpreter shutdown.

  void check(CUresult status, const char *operation) const {
    if (status == 0) return;
    const char *name = nullptr;
    const char *description = nullptr;
    getErrorName(status, &name);
    getErrorString(status, &description);
    throw std::runtime_error(std::string(operation) + ": " +
                             (name ? name : "unknown CUDA error") + " (" +
                             std::to_string(status) + "): " +
                             (description ? description : "no description"));
  }

  // Side streams of a caller stream, for launch plans with more than one
  // lane, with the events that fork them from and join them to the caller.
  // They inherit the caller's priority, never block on the legacy default
  // stream, and live for the process (two streams and three events per
  // caller stream that ever ran a concurrent plan).
  struct Lanes {
    CUevent fork = nullptr;
    std::vector<CUstream> streams;
    std::vector<CUevent> joins;
  };
  std::mutex lanesMutex;
  std::map<std::pair<CUcontext, CUstream>, Lanes> lanes;

  Lanes lanesFor(CUcontext context, CUstream caller, std::size_t count) {
    std::lock_guard<std::mutex> lock(lanesMutex);
    Lanes &entry = lanes[{context, caller}];
    // CU_EVENT_DISABLE_TIMING and CU_STREAM_NON_BLOCKING.
    if (!entry.fork) check(eventCreate(&entry.fork, 2), "cuEventCreate");
    while (entry.streams.size() < count) {
      int priority = 0;
      check(streamPriority(caller, &priority), "cuStreamGetPriority");
      CUstream stream = nullptr;
      check(streamCreate(&stream, 1, priority), "cuStreamCreateWithPriority");
      entry.streams.push_back(stream);
      CUevent join = nullptr;
      check(eventCreate(&join, 2), "cuEventCreate");
      entry.joins.push_back(join);
    }
    return entry;
  }
};
// One dlopen and cuInit per process, shared by every module.
Driver &driver() {
  static Driver instance;
  return instance;
}

struct Step {
  CUfunction function;
  std::array<unsigned, 3> grid, block;
  unsigned sharedBytes, lane;
};
struct Plan {
  std::vector<Step> steps;
  std::size_t lanes = 0;
  std::vector<std::uint64_t> scratch;
  std::vector<std::uint32_t> relocations;
};

void checkBlock(const std::array<unsigned, 3> &block) {
  std::uint64_t threads = 1;
  for (unsigned dimension : block) {
    if (!dimension || dimension > 1024)
      throw std::runtime_error("block dimensions must be in [1, 1024]");
    threads *= dimension;
  }
  if (threads > 1024) throw std::runtime_error("block exceeds 1024 CUDA threads");
}
} // namespace

struct DriverModule::Impl {
  Driver &driver = ragged::driver();
  CUcontext context = nullptr;
  CUmodule module = nullptr;
  CUdevice ordinal = -1;
  int multiprocessors = 0;
  // Kernel handles by name, resolved on first launch.
  std::mutex mutex;
  std::unordered_map<std::string, CUfunction> functions;
  std::vector<Plan> plans;

  CUfunction function(const std::string &name) {
    std::lock_guard<std::mutex> lock(mutex);
    auto found = functions.find(name);
    if (found != functions.end()) return found->second;
    CUfunction handle = nullptr;
    driver.check(driver.getFunction(&handle, module, name.c_str()), "cuModuleGetFunction");
    functions.emplace(name, handle);
    return handle;
  }

  explicit Impl(const std::string &cubin) {
    driver.check(driver.getCurrent(&context), "cuCtxGetCurrent");
    if (!context)
      throw std::runtime_error("no current CUDA context; initialize PyTorch CUDA before loading a cubin");
    driver.check(driver.getDevice(&ordinal), "cuCtxGetDevice");
    int major = 0, minor = 0;
    // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR/MINOR from the Driver API.
    driver.check(driver.getAttribute(&major, 75, ordinal), "cuDeviceGetAttribute(major)");
    driver.check(driver.getAttribute(&minor, 76, ordinal), "cuDeviceGetAttribute(minor)");
    if (major != 8 || minor != 9)
      throw std::runtime_error("sm_89 cubin requires an sm_89 device; current device is sm_" +
                               std::to_string(major) + std::to_string(minor));
    // CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT.
    driver.check(driver.getAttribute(&multiprocessors, 16, ordinal), "cuDeviceGetAttribute(multiprocessors)");
    driver.check(driver.moduleLoad(&module, cubin.data()), "cuModuleLoadData");
  }

  void requireContext() const {
    if (!module) throw std::runtime_error("CUDA module is closed");
    CUcontext current = nullptr;
    driver.check(driver.getCurrent(&current), "cuCtxGetCurrent");
    if (current != context)
      throw std::runtime_error("CUDA context differs from the context that loaded this module");
  }
};

DriverModule::DriverModule(const std::string &cubin) {
  if (cubin.size() < 4 || cubin.compare(0, 4, "\x7f" "ELF") != 0)
    throw std::runtime_error("expected a compiled ELF cubin; PTX and empty input are rejected");
  impl = std::make_unique<Impl>(cubin);
}

DriverModule::~DriverModule() {
  if (!impl || !impl->module) return;
  // Destruction may occur on a different Python thread. Restore its context.
  if (impl->driver.pushCurrent(impl->context) == 0) {
    impl->driver.moduleUnload(impl->module);
    CUcontext popped = nullptr;
    impl->driver.popCurrent(&popped);
  }
}

void DriverModule::close() {
  if (!impl->module) return;
  impl->requireContext();
  impl->driver.check(impl->driver.moduleUnload(impl->module), "cuModuleUnload");
  impl->module = nullptr;
  std::lock_guard<std::mutex> lock(impl->mutex);
  impl->functions.clear();
  impl->plans.clear();
}

int DriverModule::device() const { return impl->ordinal; }

int DriverModule::residentBlocks(const std::string &kernel, unsigned blockThreads,
                                 unsigned sharedBytes) {
  impl->requireContext();
  if (!blockThreads || blockThreads > 1024)
    throw std::runtime_error("block threads must be in [1, 1024]");
  int perMultiprocessor = 0;
  impl->driver.check(impl->driver.occupancy(&perMultiprocessor, impl->function(kernel),
      static_cast<int>(blockThreads), sharedBytes), "cuOccupancyMaxActiveBlocksPerMultiprocessor");
  return perMultiprocessor * impl->multiprocessors;
}

void DriverModule::launch(const std::string &kernel,
                          const std::array<unsigned, 3> &grid,
                          const std::array<unsigned, 3> &block,
                          unsigned sharedBytes, std::uintptr_t stream,
                          const std::vector<std::uint64_t> &arguments) {
  impl->requireContext();
  checkBlock(block);
  for (unsigned dimension : grid)
    if (!dimension) throw std::runtime_error("grid dimensions must be nonzero");
  if (kernel.empty() || kernel.find('\0') != std::string::npos)
    throw std::runtime_error("kernel name is empty or contains NUL");
  CUfunction function = impl->function(kernel);
  // The mlir-memref64-v1 ABI flattens every memref into 64-bit fields
  // (allocated and aligned pointers, offset, sizes, strides), so each kernel
  // parameter is one 8-byte value; cuLaunchKernel reads it through its address.
  static_assert(sizeof(void *) == sizeof(std::uint64_t), "64-bit host required");
  std::vector<void *> parameters;
  parameters.reserve(arguments.size());
  for (const std::uint64_t &argument : arguments)
    parameters.push_back(const_cast<std::uint64_t *>(&argument));
  impl->driver.check(impl->driver.launch(function, grid[0], grid[1], grid[2],
      block[0], block[1], block[2], sharedBytes, reinterpret_cast<CUstream>(stream),
      parameters.data(), nullptr), "cuLaunchKernel");
}

int DriverModule::addPlan(const std::vector<PlanStep> &steps,
                          const std::vector<std::uint64_t> &scratch,
                          const std::vector<std::uint32_t> &relocations) {
  impl->requireContext();
  Plan plan;
  for (const PlanStep &step : steps) {
    checkBlock(step.block);
    for (unsigned dimension : step.grid)
      if (!dimension) throw std::runtime_error("grid dimensions must be nonzero");
    if (step.lane > 8) throw std::runtime_error("a launch plan has at most eight side lanes");
    plan.steps.push_back({impl->function(step.kernel), step.grid, step.block, step.sharedBytes, step.lane});
    plan.lanes = std::max<std::size_t>(plan.lanes, step.lane);
  }
  for (std::uint32_t index : relocations)
    if (index >= scratch.size()) throw std::runtime_error("scratch relocation is out of range");
  plan.scratch = scratch;
  plan.relocations = relocations;
  std::lock_guard<std::mutex> lock(impl->mutex);
  impl->plans.push_back(std::move(plan));
  return static_cast<int>(impl->plans.size() - 1);
}

void DriverModule::clearPlans() {
  std::lock_guard<std::mutex> lock(impl->mutex);
  impl->plans.clear();
}

void DriverModule::runPlan(int id, std::uintptr_t stream, const std::vector<std::uint64_t> &inputs,
                           std::uint64_t scratchBase) {
  impl->requireContext();
  Driver &driver = impl->driver;
  std::unique_lock<std::mutex> lock(impl->mutex);
  if (id < 0 || static_cast<std::size_t>(id) >= impl->plans.size())
    throw std::runtime_error("unknown launch plan");
  const Plan &plan = impl->plans[id];
  // Inputs then scratch descriptors; scratch pointers are offsets from its base.
  std::vector<std::uint64_t> arguments(inputs);
  arguments.insert(arguments.end(), plan.scratch.begin(), plan.scratch.end());
  for (std::uint32_t index : plan.relocations) arguments[inputs.size() + index] += scratchBase;
  std::vector<void *> parameters;
  parameters.reserve(arguments.size());
  for (std::uint64_t &argument : arguments) parameters.push_back(&argument);
  CUstream caller = reinterpret_cast<CUstream>(stream);
  Driver::Lanes lanes;
  if (plan.lanes) lanes = driver.lanesFor(impl->context, caller, plan.lanes);
  // Each side lane waits for the caller stream at its first step; the caller
  // waits for every forked lane after the last step, also when a launch
  // fails, so later work on the caller stream stays ordered after the plan.
  std::vector<bool> forked(plan.lanes + 1, false);
  bool recorded = false;
  auto join = [&](bool strict) {
    for (std::size_t lane = 1; lane <= plan.lanes; ++lane) {
      if (!forked[lane]) continue;
      CUresult status = driver.eventRecord(lanes.joins[lane - 1], lanes.streams[lane - 1]);
      if (status == 0) status = driver.streamWait(caller, lanes.joins[lane - 1], 0);
      if (strict) driver.check(status, "join of a side lane");
    }
  };
  try {
    for (const Step &step : plan.steps) {
      CUstream target = caller;
      if (step.lane) {
        target = lanes.streams[step.lane - 1];
        if (!forked[step.lane]) {
          if (!recorded) {
            driver.check(driver.eventRecord(lanes.fork, caller), "cuEventRecord");
            recorded = true;
          }
          driver.check(driver.streamWait(target, lanes.fork, 0), "cuStreamWaitEvent");
          forked[step.lane] = true;
        }
      }
      driver.check(driver.launch(step.function, step.grid[0], step.grid[1], step.grid[2],
          step.block[0], step.block[1], step.block[2], step.sharedBytes, target,
          parameters.data(), nullptr), "cuLaunchKernel");
    }
  } catch (...) {
    join(false);
    throw;
  }
  join(true);
}
} // namespace ragged
