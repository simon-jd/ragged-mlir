#include "Ragged/Passes.h"
#include "Ragged/Ragged.h"
#include "Ragged/ToolchainConfig.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVM/NVVM/Target.h"
#include "mlir/Target/LLVM/NVVM/Utils.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/VCSRevision.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Regex.h"
#include <limits>

using namespace mlir;
namespace {
llvm::cl::opt<std::string> input(llvm::cl::Positional, llvm::cl::desc("input MLIR"), llvm::cl::init("-"));
llvm::cl::opt<std::string> output("o", llvm::cl::desc("output path"), llvm::cl::init("-"));
llvm::cl::opt<std::string> emit("emit", llvm::cl::desc("legal|plan|gpu|nvvm|ptx|cubin"), llvm::cl::init("gpu"));
llvm::cl::opt<bool> explain("explain", llvm::cl::desc("print dispatch decisions to stderr"));
llvm::cl::opt<std::string> toolkit("cuda-toolkit", llvm::cl::desc("CUDA toolkit root for libdevice and ptxas"), llvm::cl::init(""));
llvm::cl::opt<std::string> chip("chip", llvm::cl::desc("target chip (sm_89)"), llvm::cl::init("sm_89"));
llvm::cl::opt<std::string> features("features", llvm::cl::desc("PTX target feature"), llvm::cl::init("+ptx80"));
llvm::cl::opt<bool> toolchainJSON("toolchain-json", llvm::cl::desc("print actual compiler and ptxas provenance"));

std::string ptxasVersion() {
  std::string path;
  if (!toolkit.empty()) {
    llvm::SmallString<256> candidate(toolkit.getValue());
    llvm::sys::path::append(candidate, "bin", "ptxas");
    if (llvm::sys::fs::can_execute(candidate)) path = candidate.str().str();
  }
  if (path.empty()) {
    auto found = llvm::sys::findProgramByName("ptxas");
    if (found) path = *found;
  }
  if (path.empty()) {
    llvm::SmallString<256> candidate(NVVM::getCUDAToolkitPath());
    llvm::sys::path::append(candidate, "bin", "ptxas");
    if (llvm::sys::fs::can_execute(candidate)) path = candidate.str().str();
  }
  if (path.empty()) return "unavailable";
  llvm::SmallString<128> temporary;
  if (llvm::sys::fs::createTemporaryFile("ragged-ptxas-version", "txt", temporary))
    return "unavailable";
  llvm::FileRemover remover(temporary);
  std::optional<StringRef> redirects[3] = {StringRef(""), temporary.str(), temporary.str()};
  if (llvm::sys::ExecuteAndWait(path, {path, "--version"}, std::nullopt, redirects, 10) != 0)
    return "unavailable";
  auto data = llvm::MemoryBuffer::getFile(temporary);
  return data ? (*data)->getBuffer().trim().str() : "unavailable";
}
llvm::json::Object provenance() {
#ifdef LLVM_REVISION
  constexpr StringRef revision = LLVM_REVISION;
  constexpr StringRef revisionSource = "llvm-vcs-header";
#else
  constexpr StringRef revision = RAGGED_VERIFIED_LLVM_REVISION;
  constexpr StringRef revisionSource = "verified-source-archive";
#endif
  llvm::json::Object result{{"llvm_commit", revision.str()}, {"llvm_version", LLVM_VERSION_STRING},
          {"llvm_revision_source", revisionSource.str()},
          {"ptxas_version", ptxasVersion()}, {"dialect_version", "1"},
          {"chip", chip.getValue()}, {"features", features.getValue()}};
#ifndef LLVM_REVISION
  result["llvm_source_archive_sha256"] = RAGGED_LLVM_SOURCE_ARCHIVE_SHA256;
#endif
  return result;
}

llvm::json::Value attrToJSON(Attribute attr) {
  if (auto s = dyn_cast<StringAttr>(attr)) return s.getValue().str();
  if (auto value = dyn_cast<BoolAttr>(attr)) return value.getValue();
  if (auto i = dyn_cast<IntegerAttr>(attr)) return i.getInt();
  if (auto a = dyn_cast<ArrayAttr>(attr)) {
    llvm::json::Array result;
    for (Attribute item : a) result.push_back(attrToJSON(item));
    return result;
  }
  if (auto a = dyn_cast<DenseI64ArrayAttr>(attr)) {
    llvm::json::Array result;
    for (int64_t item : a.asArrayRef()) result.push_back(item);
    return result;
  }
  if (auto d = dyn_cast<DictionaryAttr>(attr)) {
    llvm::json::Object result;
    for (NamedAttribute item : d) result[item.getName().str()] = attrToJSON(item.getValue());
    return result;
  }
  return nullptr;
}

llvm::json::Object resourceReport(int64_t registers, int64_t shared) {
  return llvm::json::Object{{"status", "unmeasured"}, {"source", nullptr},
          {"requested", llvm::json::Object{{"regs_budget", registers}, {"smem_per_cta", shared}}},
          {"kernels", llvm::json::Array{}}};
}

// Read diagnostics from the very ptxas invocation that serialized this cubin.
// Missing data never means zero: only an otherwise complete Used line may omit
// its optional smem clause. Every main and validation entry must appear once.
FailureOr<llvm::json::Object> measuredResources(StringRef log, ArrayAttr main,
                                               DictionaryAttr validation,
                                               int64_t regsBudget, int64_t sharedBudget) {
  struct Measurement {
    int64_t threads = 1, dynamicShared = 0;
    int64_t registers = 0, shared = 0, stack = 0, stores = 0, loads = 0;
    bool entry = false, properties = false, usage = false, frame = false;
  };
  llvm::StringMap<Measurement> measurements;
  SmallVector<StringRef> order;
  auto reject = [&](const Twine &message) -> FailureOr<llvm::json::Object> {
    llvm::errs() << "ptxas resource validation failed: " << message << "\n";
    return failure();
  };
  if (!main || !validation || !validation.getAs<ArrayAttr>("kernels"))
    return reject("missing main or validation launch manifest");
  for (ArrayAttr kernels : {main, validation.getAs<ArrayAttr>("kernels")})
    for (Attribute attr : kernels) {
      auto kernel = cast<DictionaryAttr>(attr);
      StringRef name = kernel.getAs<StringAttr>("name").getValue();
      auto inserted = measurements.try_emplace(name);
      if (!inserted.second) return reject("duplicate manifest entry " + name);
      auto &item = inserted.first->second;
      auto block = kernel.getAs<DenseI64ArrayAttr>("block");
      if (!block || block.size() != 3) return reject("invalid launch block for " + name);
      for (int64_t dimension : block.asArrayRef()) {
        if (dimension <= 0 || dimension > 1024 || item.threads > 1024 / dimension)
          return reject("invalid thread count for " + name);
        item.threads *= dimension;
      }
      item.dynamicShared = kernel.getAs<IntegerAttr>("shared_bytes").getInt();
      if (item.dynamicShared < 0) return reject("negative dynamic shared memory for " + name);
      order.push_back(name);
    }
  llvm::Regex entry("^ptxas info[ ]*: Compiling entry function '([^']+)' for '([^']+)'[ ]*$");
  llvm::Regex properties("^ptxas info[ ]*: Function properties for (.+)$");
  llvm::Regex usage("^ptxas info[ ]*: Used ([0-9]+) registers(,.*)?$");
  llvm::Regex shared("^([0-9]+) bytes smem$");
  llvm::Regex barriers("^used ([0-9]+) barriers$");
  llvm::Regex constant("^([0-9]+) bytes cmem\\[([0-9]+)\\]$");
  llvm::Regex frame("^[ ]*([0-9]+) bytes stack frame, ([0-9]+) bytes spill stores, ([0-9]+) bytes spill loads[ ]*$");
  SmallVector<StringRef> lines, match;
  log.split(lines, '\n');
  Measurement *current = nullptr;
  StringRef currentName;
  auto number = [](StringRef text, int64_t &result) { return !text.getAsInteger(10, result) && result >= 0; };
  for (StringRef line : lines) {
    line = line.rtrim("\r");
    if (entry.match(line, &match)) {
      auto found = measurements.find(match[1]);
      if (found == measurements.end()) return reject("unknown entry " + match[1]);
      if (found->second.entry) return reject("duplicate entry " + match[1]);
      if (match[2] != chip) return reject("target mismatch for " + match[1]);
      current = &found->second;
      currentName = found->first();
      current->entry = true;
    } else if (properties.match(line, &match)) {
      if (!current || current->properties || match[1] != currentName)
        return reject("duplicate or unmatched function properties record");
      current->properties = true;
    } else if (frame.match(line, &match)) {
      if (!current || current->frame || !current->properties)
        return reject("duplicate or unmatched stack/spill record");
      if (!number(match[1], current->stack) || !number(match[2], current->stores) || !number(match[3], current->loads))
        return reject("invalid stack/spill integer for " + currentName);
      current->frame = true;
    } else if (usage.match(line, &match)) {
      if (!current || current->usage || !current->properties)
        return reject("duplicate or unmatched register record");
      if (!number(match[1], current->registers)) return reject("invalid register integer for " + currentName);
      SmallVector<StringRef> clauses, clauseMatch;
      line.split(clauses, ',');
      bool seenShared = false, seenBarriers = false;
      llvm::SmallDenseSet<int64_t> constantBanks;
      for (StringRef clause : llvm::drop_begin(clauses)) {
        clause = clause.trim();
        int64_t ignored = 0, bank = 0;
        if (shared.match(clause, &clauseMatch)) {
          if (seenShared || !number(clauseMatch[1], current->shared))
            return reject("duplicate or invalid shared memory record for " + currentName);
          seenShared = true;
        } else if (barriers.match(clause, &clauseMatch)) {
          if (seenBarriers || !number(clauseMatch[1], ignored))
            return reject("duplicate or invalid barrier record for " + currentName);
          seenBarriers = true;
        } else if (constant.match(clause, &clauseMatch)) {
          if (!number(clauseMatch[1], ignored) || !number(clauseMatch[2], bank) || !constantBanks.insert(bank).second)
            return reject("duplicate or invalid constant memory record for " + currentName);
        } else {
          return reject("unrecognized resource clause for " + currentName + ": " + clause);
        }
      }
      current->usage = true;
    } else if (line.contains("Compiling entry function") || line.contains("Function properties for") ||
               line.contains("bytes stack frame") || line.contains("Used ")) {
      return reject("unrecognized resource record: " + line);
    }
  }
  llvm::json::Array kernels;
  for (StringRef name : order) {
    const Measurement &item = measurements.find(name)->second;
    if (!item.entry || !item.properties || !item.usage || !item.frame)
      return reject("incomplete measurement for " + name);
    if (item.registers > std::numeric_limits<int64_t>::max() / item.threads ||
        item.shared > std::numeric_limits<int64_t>::max() - item.dynamicShared)
      return reject("resource arithmetic overflow for " + name);
    int64_t registers = item.registers * item.threads;
    int64_t sharedBytes = item.shared + item.dynamicShared;
    if (registers > regsBudget)
      return reject(name + " uses " + Twine(registers) + " registers per CTA (" + Twine(item.registers) +
                    " per thread * " + Twine(item.threads) + "), exceeding regs_budget=" + Twine(regsBudget));
    if (sharedBytes > sharedBudget)
      return reject(name + " uses " + Twine(sharedBytes) + " shared bytes per CTA, exceeding smem_per_cta=" + Twine(sharedBudget));
    kernels.push_back(llvm::json::Object{
        {"name", name.str()}, {"registers_per_thread", item.registers}, {"threads_per_cta", item.threads},
        {"registers_per_cta", registers}, {"static_shared_bytes", item.shared}, {"dynamic_shared_bytes", item.dynamicShared},
        {"shared_bytes_per_cta", sharedBytes}, {"stack_frame_bytes", item.stack},
        {"spill_store_bytes", item.stores}, {"spill_load_bytes", item.loads}});
  }
  auto report = resourceReport(regsBudget, sharedBudget);
  report["status"] = "measured";
  report["source"] = "ptxas";
  report["kernels"] = std::move(kernels);
  return report;
}
}

int main(int argc, char **argv) {
  llvm::InitLLVM init(argc, argv);
  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmPrinters();
  mlir::registerAllPasses();
  ragged::registerRaggedPasses();
  llvm::cl::ParseCommandLineOptions(argc, argv, "Compile ragged operations to sm_89 GPU kernels\n");
#ifdef LLVM_REVISION
  if (StringRef(LLVM_REVISION).empty()) {
    llvm::errs() << "LLVM_REVISION is empty; compiler toolchain provenance is unavailable\n";
    return 1;
  }
#else
  if (StringRef(RAGGED_VERIFIED_LLVM_REVISION).empty()) {
    llvm::errs() << "LLVM revision is unavailable; reconfigure with a verified source archive\n";
    return 1;
  }
#endif
  if (chip != "sm_89" || !StringRef(features).starts_with("+ptx") ||
      StringRef(features).drop_front(4).empty() ||
      !llvm::all_of(StringRef(features).drop_front(4), [](char c) { return c >= '0' && c <= '9'; })) {
    llvm::errs() << "only --chip=sm_89 and a single +ptxNN feature are supported\n"; return 1;
  }
  // LLVM warns and ignores unknown feature names by default. Validate against
  // the linked backend's actual feature table before a cache key or sidecar can
  // claim that the ignored feature was used to compile this program.
  std::string targetError;
  llvm::Triple triple("nvptx64-nvidia-cuda");
  const llvm::Target *target = llvm::TargetRegistry::lookupTarget(triple, targetError);
  std::unique_ptr<llvm::MCSubtargetInfo> subtarget(
      target ? target->createMCSubtargetInfo(triple, chip, "") : nullptr);
  if (!subtarget) {
    llvm::errs() << "cannot inspect NVPTX target features: " << targetError << "\n"; return 1;
  }
  StringRef requestedFeature = StringRef(features).drop_front();
  if (!llvm::any_of(subtarget->getAllProcessorFeatures(), [&](const llvm::SubtargetFeatureKV &feature) {
        return requestedFeature == feature.key();
      })) {
    llvm::errs() << "unsupported NVPTX target feature: " << features << "\n"; return 1;
  }
  if (toolchainJSON) {
    llvm::outs() << llvm::formatv("{0:2}\n", llvm::json::Value(provenance())); return 0;
  }
  if (emit != "legal" && emit != "plan" && emit != "gpu" && emit != "nvvm" && emit != "ptx" && emit != "cubin") {
    llvm::errs() << "unsupported --emit value: " << emit << "\n"; return 1;
  }
  if ((emit == "cubin" || emit == "ptx") && output == "-") {
    llvm::errs() << "binary/assembly output requires -o so its ABI sidecar can be written\n"; return 1;
  }
  DialectRegistry registry;
  registerAllDialects(registry); registerAllExtensions(registry);
  registerAllToLLVMIRTranslations(registry);
  registry.insert<ragged::RaggedDialect>();
  MLIRContext context(registry);
  std::string error;
  auto source = openInputFile(input, &error);
  if (!source) { llvm::errs() << error << "\n"; return 1; }
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(source), llvm::SMLoc());
  auto module = parseSourceFile<ModuleOp>(sourceMgr, &context);
  if (!module || failed(verify(*module))) return 1;
  int64_t d = 0;
  bool softmax = false;
  bool additiveMask = false;
  int64_t regsBudget = 65536, sharedBudget = 49152;
  unsigned opCount = 0;
  module->walk([&](Operation *op) {
    if (isa<ragged::SegmentReduceOp, ragged::SegmentSoftmaxOp>(op)) {
      ++opCount; softmax = isa<ragged::SegmentSoftmaxOp>(op);
      additiveMask = op->getNumOperands() == 4;
      d = cast<MemRefType>(op->getOperand(0).getType()).getDimSize(1);
      if (auto attr = op->getAttrOfType<IntegerAttr>("regs_budget")) regsBudget = attr.getInt();
      if (auto attr = op->getAttrOfType<IntegerAttr>("smem_per_cta")) sharedBudget = attr.getInt();
    }
  });
  if (opCount != 1) { llvm::errs() << "ragged-compile requires exactly one source ragged operation\n"; return 1; }
  Builder builder(&context);
  module->getOperation()->setAttr("ragged.resources", builder.getDictionaryAttr({
      builder.getNamedAttr("status", builder.getStringAttr("unmeasured")),
      builder.getNamedAttr("source", builder.getUnitAttr()),
      builder.getNamedAttr("requested", builder.getDictionaryAttr({
          builder.getNamedAttr("regs_budget", builder.getI64IntegerAttr(regsBudget)),
          builder.getNamedAttr("smem_per_cta", builder.getI64IntegerAttr(sharedBudget))})),
      builder.getNamedAttr("kernels", builder.getArrayAttr({}))}));
  auto functions = module->getOps<func::FuncOp>();
  if (!llvm::hasSingleElement(functions)) {
    llvm::errs() << "ragged-compile requires one entry function\n"; return 1;
  }
  auto function = *functions.begin();
  for (Operation &op : module->getBody()->getOperations())
    if (&op != function.getOperation()) {
      op.emitError("ragged-compile only supports one entry function at module scope"); return 1;
    }
  unsigned inputCount = additiveMask ? 4 : 3;
  if (!function.getBody().hasOneBlock() || function.getNumArguments() != inputCount || function.getNumResults() != 0) {
    llvm::errs() << "entry must have one block, " << inputCount << " memref arguments, and no results\n"; return 1;
  }
  for (Operation &op : function.getBody().front()) {
    if (isa<func::ReturnOp>(op)) continue;
    if (!isa<ragged::SegmentReduceOp, ragged::SegmentSoftmaxOp>(op) ||
        op.getOperand(0) != function.getArgument(0) || op.getOperand(1) != function.getArgument(1) ||
        op.getOperand(2) != function.getArgument(2) ||
        (additiveMask && op.getOperand(3) != function.getArgument(3))) {
      op.emitError("entry must contain only one ragged operation on values, offsets, out and optional mask arguments, followed by return"); return 1;
    }
  }
  PassManager pm(&context);
  pm.addPass(ragged::createLegalizePass());
  if (emit != "legal") { pm.addPass(ragged::createPlanPass()); pm.addPass(ragged::createFusePass()); }
  if (failed(pm.run(*module))) return 1;
  llvm::json::Value classes = nullptr;
  llvm::json::Value thresholds = nullptr;
  std::string reason;
  std::string fusion = "not_applicable";
  llvm::json::Value classFusion = nullptr;
  module->walk([&](ragged::DispatchOp dispatch) {
    classes = attrToJSON(dispatch->getAttr("classes"));
    thresholds = attrToJSON(dispatch->getAttr("thresholds"));
    reason = dispatch->getAttrOfType<StringAttr>("reason").getValue().str();
    if (explain) {
      llvm::errs() << "ragged-plan: " << reason << "; classes=";
      dispatch->getAttr("classes").print(llvm::errs());
      llvm::errs() << "; d_tile=";
      dispatch->getAttr("d_tile").print(llvm::errs());
      llvm::errs() << "\n";
    }
  });
  module->walk([&](ragged::MapSegmentsOp map) {
    if (auto attr = map->getAttrOfType<StringAttr>("fusion")) {
      fusion = attr.getValue().str();
      if (Attribute schedule = map->getAttr("class_fusion")) classFusion = attrToJSON(schedule);
      if (explain)
        llvm::errs() << "ragged-fuse: " << fusion << "; "
                     << map->getAttrOfType<StringAttr>("fusion_reason").getValue() << "\n";
    }
  });
  if (emit != "legal" && emit != "plan") {
    pm.clear(); pm.addPass(ragged::createToGPUPass());
    if (failed(pm.run(*module))) return 1;
  }
  Attribute kernelManifest = module->getOperation()->getAttr("ragged.kernels");
  Attribute workspaceManifest = module->getOperation()->getAttr("ragged.workspace");
  Attribute validationManifest = module->getOperation()->getAttr("ragged.validation");
  Attribute optimizationManifest = module->getOperation()->getAttr("ragged.optimizations");
  Attribute resolvedTile = module->getOperation()->getAttr("ragged.d_tile");
  Attribute materializedPointwise = module->getOperation()->getAttr("ragged.pointwise_materialized");
  if (explain && optimizationManifest) {
    llvm::errs() << "ragged-controls: " << llvm::formatv("{0}", attrToJSON(optimizationManifest))
                 << "; d_tile=" << cast<IntegerAttr>(resolvedTile).getInt()
                 << "; pointwise_materialized=" << (cast<BoolAttr>(materializedPointwise).getValue() ? "true" : "false") << "\n";
  }
  auto dispatchABI = module->getOperation()->getAttrOfType<StringAttr>("ragged.dispatch_abi");
  auto backend = module->getOperation()->getAttrOfType<StringAttr>("ragged.backend");
  auto debugVisits = module->getOperation()->getAttrOfType<BoolAttr>("ragged.debug_visits");
  bool needsWorkspace = false;
  if (auto kernels = dyn_cast_or_null<ArrayAttr>(kernelManifest))
    for (Attribute kernel : kernels)
      for (Attribute arg : cast<DictionaryAttr>(kernel).getAs<ArrayAttr>("args"))
        needsWorkspace |= cast<StringAttr>(arg).getValue() == "max_stats";
  if (emit == "nvvm" || emit == "ptx" || emit == "cubin") {
    pm.clear();
    std::string pipeline = "gpu.module(convert-scf-to-cf,convert-gpu-to-nvvm,canonicalize,cse,reconcile-unrealized-casts),nvvm-attach-target{chip=" + chip + " features=" + features + " collect-compiler-diagnostics=true}";
    if (emit != "nvvm") {
      pipeline += ",gpu-module-to-binary{format=";
      pipeline += emit == "ptx" ? "isa" : "bin";
      if (emit == "cubin") pipeline += " opts=-v";
      if (!toolkit.empty()) pipeline += " toolkit=" + toolkit;
      pipeline += "}";
    }
    if (failed(parsePassPipeline(pipeline, pm, llvm::errs())) || failed(pm.run(*module))) return 1;
  }
  llvm::json::Object resources = resourceReport(regsBudget, sharedBudget);
  if (emit == "cubin" || emit == "ptx") {
    gpu::BinaryOp binary;
    module->walk([&](gpu::BinaryOp op) { binary = op; });
    if (!binary || binary.getObjects().size() != 1) {
      llvm::errs() << "expected one serialized GPU object\n"; return 1;
    }
    auto object = cast<gpu::ObjectAttr>(binary.getObjects()[0]);
    StringRef compilerLog;
    if (auto properties = object.getProperties())
      if (auto log = properties.getAs<StringAttr>("ISACompilerLog")) compilerLog = log.getValue();
    if (emit == "cubin") {
      llvm::errs() << compilerLog;
      auto measured = measuredResources(compilerLog, cast<ArrayAttr>(kernelManifest),
                                        cast<DictionaryAttr>(validationManifest), regsBudget, sharedBudget);
      if (failed(measured)) return 1;
      resources = std::move(*measured);
    }
    if (explain) llvm::errs() << "ragged-resources: " << llvm::formatv("{0}", llvm::json::Value(llvm::json::Object(resources))) << "\n";
    llvm::json::Array workspace;
    if (needsWorkspace) {
      for (StringRef name : {"max_stats", "sum_stats"})
        workspace.push_back(llvm::json::Object{{"name", name.str()}, {"shape", llvm::json::Array{"B", d}}, {"dtype", "f32"}});
    }
    llvm::json::Object manifest{
      {"schema", 1}, {"abi", "mlir-memref64-v1"}, {"chip", "sm_89"},
      {"mask", additiveMask},
      {"resources", std::move(resources)},
      {"optimizations", attrToJSON(optimizationManifest)},
      {"d_tile", attrToJSON(resolvedTile)}, {"pointwise_materialized", attrToJSON(materializedPointwise)},
      {"validation", attrToJSON(validationManifest)},
      {"kernels", attrToJSON(kernelManifest)}, {"classes", std::move(classes)},
      {"thresholds", std::move(thresholds)},
      {"workspace", workspaceManifest ? attrToJSON(workspaceManifest) : llvm::json::Value(std::move(workspace))},
      {"fallback_reason", reason},
      {"backend", backend.getValue().str()}, {"fusion", fusion},
      {"alias_safe", softmax && fusion == "three_pass"}, {"toolchain", provenance()}};
    if (softmax) manifest["class_fusion"] = std::move(classFusion);
    if (dispatchABI) {
      manifest["dispatch_abi"] = dispatchABI.getValue().str();
      manifest["split_parts"] = 8;
      manifest["debug_visits"] = debugVisits.getValue();
    }
    // No output path is opened until compilation, complete measurement, and
    // all budget checks have succeeded. In particular rejection preserves a
    // previously valid artifact at the same path.
    auto file = openOutputFile(output, &error);
    if (!file) { llvm::errs() << error << "\n"; return 1; }
    auto sidecar = openOutputFile(output + ".json", &error);
    if (!sidecar) { llvm::errs() << error << "\n"; return 1; }
    std::unique_ptr<llvm::ToolOutputFile> logFile;
    if (emit == "cubin") {
      logFile = openOutputFile(output + ".ptxas.log", &error);
      if (!logFile) { llvm::errs() << error << "\n"; return 1; }
      logFile->os() << compilerLog;
    }
    file->os() << object.getObject().getValue();
    sidecar->os() << llvm::formatv("{0:2}\n", llvm::json::Value(std::move(manifest)));
    if (logFile) logFile->keep();
    sidecar->keep();
    file->keep();
  } else {
    if (explain) llvm::errs() << "ragged-resources: unmeasured (no ptxas assembly)\n";
    auto file = openOutputFile(output, &error);
    if (!file) { llvm::errs() << error << "\n"; return 1; }
    module->print(file->os()); file->os() << "\n";
    file->keep();
  }
  return 0;
}
