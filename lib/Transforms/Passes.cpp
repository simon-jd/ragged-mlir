#include "Ragged/Passes.h"
#include "Ragged/Ragged.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
namespace ragged {
namespace {
// Segment classes by length (rows). A warp-class segment is reduced by one
// warp with a shuffle tree and no shared memory; a CTA-class segment by one
// 128-thread block; longer segments are split over 8 blocks and combined. The
// bounds are the dispatch's `thresholds` attribute, which the classify kernel
// and the runtime's classified ABI (python/ragged/runtime.py) read.
constexpr int64_t kWarpMaxLength = 32;
constexpr int64_t kCtaMaxLength = 4096;
static StringRef stringAttr(Operation *op, StringRef name, StringRef fallback = "") {
  if (auto attr = op->getAttrOfType<StringAttr>(name))
    return attr.getValue();
  return fallback;
}
static bool boolAttr(Operation *op, StringRef name, bool fallback = false) {
  auto attr = op->getAttrOfType<BoolAttr>(name);
  return attr ? attr.getValue() : fallback;
}
static int64_t intAttr(Operation *op, StringRef name, int64_t fallback) {
  auto attr = op->getAttrOfType<IntegerAttr>(name);
  return attr ? attr.getInt() : fallback;
}

struct LegalizePass : PassWrapper<LegalizePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LegalizePass)
  StringRef getArgument() const final { return "ragged-legalize"; }
  StringRef getDescription() const final {
    return "Make ragged accumulation, identity and NaN semantics explicit";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<RaggedDialect>();
  }
  void runOnOperation() override {
    SmallVector<Operation *> source;
    getOperation().walk([&](Operation *op) {
      if (isa<SegmentReduceOp, SegmentSoftmaxOp>(op)) source.push_back(op);
    });
    for (Operation *op : source) {
      OpBuilder builder(op);
      StringRef kind = isa<SegmentSoftmaxOp>(op) ? "softmax" : stringAttr(op, "kind");
      OperationState state(op->getLoc(), MapSegmentsOp::getOperationName());
      state.addOperands(op->getOperands());
      state.addAttributes(op->getAttrs());
      auto *map = builder.create(state);
      map->removeAttr("kind");
      map->setAttr("operation", builder.getStringAttr(kind));
      map->setAttr("accumulator", TypeAttr::get(builder.getF32Type()));
      map->setAttr("identity", builder.getStringAttr(kind == "sum" ? "zero" : "negative_infinity"));
      map->setAttr("nan", builder.getStringAttr("propagate"));
      map->setAttr("semantics_version", builder.getI64IntegerAttr(1));
      if (op->getOperand(0) == op->getOperand(2))
        map->setAttr("alias", builder.getStringAttr("inplace"));
      op->erase();
    }
  }
};

struct PlanPass : PassWrapper<PlanPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PlanPass)
  StringRef getArgument() const final { return "ragged-plan"; }
  StringRef getDescription() const final {
    return "Select segment length classes and feature tiles within sm_89 budgets";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<RaggedDialect>();
  }
  void runOnOperation() override {
    bool bad = false;
    getOperation().walk([&](Operation *op) {
      if (isa<SegmentReduceOp, SegmentSoftmaxOp>(op)) {
        op->emitOpError("run ragged-legalize before ragged-plan"); bad = true;
      }
    });
    if (bad) return signalPassFailure();
    SmallVector<MapSegmentsOp> maps;
    getOperation().walk([&](MapSegmentsOp op) {
      if (!op->getParentOfType<DispatchOp>()) maps.push_back(op);
    });
    for (MapSegmentsOp map : maps) {
      OpBuilder b(map);
      auto chip = stringAttr(map, "chip", "sm_89");
      if (chip != "sm_89") {
        map.emitOpError("target must be sm_89");
        signalPassFailure(); return;
      }
      int64_t d = cast<MemRefType>(map.getValues().getType()).getDimSize(1);
      // Each thread owns `vec` adjacent features, read and written with one
      // access of at most 16 bytes; vec divides D so no vector straddles the
      // end of a row. An explicit feature tile keeps one feature per thread.
      int64_t elementBytes = cast<MemRefType>(map.getValues().getType()).getElementTypeBitWidth() / 8;
      int64_t vec = 1;
      if (boolAttr(map, "vectorize", true) && !map->hasAttr("feature_tile"))
        // The block tree keeps one f32 per thread and feature in shared memory.
        while (2 * vec * elementBytes <= 16 && d % (2 * vec) == 0 &&
               128 * 2 * vec * 4 <= intAttr(map, "smem_per_cta", 49152)) vec *= 2;
      // A power-of-two feature tile keeps reduction groups aligned at warp
      // boundaries. A tile spans the next power of two of D, at most 32
      // features, with or without vectors: d_tile = features / vec threads
      // cover it, so wider accesses add row lanes rather than removing tiles.
      int64_t features = 1;
      while (features < d && features < 32) features *= 2;
      vec = std::min(vec, features);
      int64_t tile = intAttr(map, "feature_tile", features / vec);
      int64_t max = intAttr(map, "max_len", -1);
      int64_t min = intAttr(map, "min_len", 0);
      bool naive = stringAttr(map, "strategy", "auto") == "naive";
      bool classify = boolAttr(map, "classification", true);
      bool warp = boolAttr(map, "warp_mapping", true);
      bool split = boolAttr(map, "split_mapping", true);
      SmallVector<Attribute> classes;
      StringRef reason;
      if (naive || !classify) {
        classes.push_back(b.getStringAttr("cta"));
        reason = naive ? "naive strategy requested" : "device classification disabled; direct CTA mapping";
      } else {
        if (min <= kWarpMaxLength && warp) classes.push_back(b.getStringAttr("warp"));
        if (((max < 0 || max > kWarpMaxLength) && min <= kCtaMaxLength) || (!warp && min <= kWarpMaxLength) ||
            (!split && (max < 0 || max > kCtaMaxLength))) classes.push_back(b.getStringAttr("cta"));
        if ((max < 0 || max > kCtaMaxLength) && split) classes.push_back(b.getStringAttr("split"));
        if (classes.empty()) classes.push_back(b.getStringAttr("cta"));
        reason = classes.size() == 1 ? "static length bounds select one class" : "length bounds require device classification";
        if (!warp || !split) reason = "disabled mappings fall back to CTA coverage";
      }
      OperationState state(map.getLoc(), DispatchOp::getOperationName());
      state.addAttribute("classes", b.getArrayAttr(classes));
      bool deviceClassification = classes.size() > 1 ||
          cast<StringAttr>(classes.front()).getValue() == "split" || boolAttr(map, "debug_visits");
      state.addAttribute("device_classification", b.getBoolAttr(deviceClassification));
      state.addAttribute("thresholds", b.getDenseI64ArrayAttr({kWarpMaxLength, kCtaMaxLength}));
      state.addAttribute("chip", b.getStringAttr(chip));
      state.addAttribute("reason", b.getStringAttr(reason));
      state.addAttribute("d_tile", b.getI64IntegerAttr(tile));
      state.addAttribute("d_vector", b.getI64IntegerAttr(vec));
      int64_t threads = classes.size() == 1 && cast<StringAttr>(classes.front()).getValue() == "warp" ? 32 : 128;
      state.addAttribute("threads", b.getI64IntegerAttr(threads));
      state.addAttribute("shared_bytes", b.getI64IntegerAttr(threads == 32 && boolAttr(map, "shuffle", true) ? 0 : threads * vec * 4));
      state.addRegion();
      auto dispatch = cast<DispatchOp>(b.create(state));
      auto *body = new Block();
      dispatch.getBody().push_back(body);
      b.setInsertionPointToStart(body);
      if (deviceClassification) {
        OperationState classifier(map.getLoc(), ClassifyOp::getOperationName());
        classifier.addOperands(map.getOffsets());
        classifier.addAttribute("thresholds", b.getDenseI64ArrayAttr({kWarpMaxLength, kCtaMaxLength}));
        b.create(classifier);
      }
      map->moveBefore(body, body->end());
      map->setAttr("d_tile", b.getI64IntegerAttr(tile));
      map->setAttr("d_vector", b.getI64IntegerAttr(vec));
      map->setAttr("mapping", classes.size() == 1 ? classes.front() : b.getStringAttr("classified"));
      b.setInsertionPointToEnd(body);
      b.create<YieldOp>(map.getLoc());
    }
  }
};

struct FusePass : PassWrapper<FusePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FusePass)
  StringRef getArgument() const final { return "ragged-fuse"; }
  StringRef getDescription() const final {
    return "Choose legal softmax statistics fusion and diagnose unsafe forced fusion";
  }
  void runOnOperation() override {
    bool failed = false;
    getOperation().walk([&](MapSegmentsOp map) {
      if (stringAttr(map, "operation") != "softmax") return;
      OpBuilder b(map);
      auto dispatch = map->getParentOfType<DispatchOp>();
      if (!dispatch) {
        map.emitOpError("run ragged-plan before ragged-fuse"); failed = true; return;
      }
      bool alias = stringAttr(map, "alias", "none") == "inplace" || map.getValues() == map.getOut();
      auto classes = dispatch->getAttrOfType<ArrayAttr>("classes");
      bool onlineEnabled = boolAttr(map, "online_fusion", true);
      // Each class gets its own schedule. Online statistics are exact for
      // any segment length (compensated lane sums against a reference that
      // lags the maximum by at most one), so only the output alias and each
      // class's shared memory decide. Split parts keep a kernel boundary:
      // their per-part statistics merge in a combine kernel.
      SmallVector<Attribute> schedule;
      SmallVector<std::pair<std::string, StringRef>> reasons;
      StringRef rejection;
      int64_t sharedBytes = 0;
      for (Attribute attr : classes) {
        StringRef segmentClass = cast<StringAttr>(attr).getValue();
        // One f32 reference and one f32 mass per thread and feature for the block tree.
        int64_t vec = intAttr(map, "d_vector", 1);
        int64_t classBytes = segmentClass == "warp" && boolAttr(map, "shuffle", true)
            ? 0 : (segmentClass == "warp" ? 32 : 128) * vec * 2 * 4;
        StringRef reason;
        if (!onlineEnabled) reason = "online statistics fusion disabled";
        else if (alias) reason = "in-place output requires kernel-separated read and write phases";
        else if (classBytes > intAttr(map, "smem_per_cta", 49152)) reason = "online statistics exceed shared-memory budget";
        bool fused = reason.empty();
        if (fused) {
          reason = "disjoint output permits online statistics";
          sharedBytes = std::max(sharedBytes, classBytes);
        } else if (rejection.empty()) {
          rejection = reason;
        }
        schedule.push_back(b.getStringAttr(fused ? "online" : "three_pass"));
        if (!reasons.empty() && reasons.back().second == reason)
          reasons.back().first += ", " + segmentClass.str();
        else
          reasons.push_back({segmentClass.str(), reason});
      }
      if (!rejection.empty() && boolAttr(map, "force_fusion")) {
        map.emitOpError() << "forced fusion is illegal: " << rejection;
        failed = true; return;
      }
      bool anyOnline = llvm::any_of(schedule, [](Attribute item) { return cast<StringAttr>(item).getValue() == "online"; });
      StringRef fusion = rejection.empty() ? "online" : anyOnline ? "mixed" : "three_pass";
      std::string reason = reasons.front().second.str();
      if (reasons.size() > 1) {
        reason.clear();
        for (auto &[names, text] : reasons)
          reason += (reason.empty() ? "" : "; ") + names + ": " + text.str();
      }
      map->setAttr("fusion", b.getStringAttr(fusion));
      map->setAttr("class_fusion", b.getArrayAttr(schedule));
      map->setAttr("fusion_reason", b.getStringAttr(reason));
      if (anyOnline) dispatch->setAttr("shared_bytes", b.getI64IntegerAttr(sharedBytes));
    });
    if (failed) signalPassFailure();
  }
};
}
std::unique_ptr<Pass> createLegalizePass() { return std::make_unique<LegalizePass>(); }
std::unique_ptr<Pass> createPlanPass() { return std::make_unique<PlanPass>(); }
std::unique_ptr<Pass> createFusePass() { return std::make_unique<FusePass>(); }
void registerRaggedPasses() {
  PassRegistration<LegalizePass>();
  PassRegistration<PlanPass>();
  PassRegistration<FusePass>();
  registerPass([] { return createToGPUPass(); });
  PassPipelineRegistration<>("ragged-pipeline", "Legalize, plan, fuse and lower ragged operations",
    [](OpPassManager &pm) {
      pm.addPass(createLegalizePass()); pm.addPass(createPlanPass());
      pm.addPass(createFusePass()); pm.addPass(createToGPUPass());
    });
}
}
