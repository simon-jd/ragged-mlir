#include "Ragged/Ragged.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/StringSwitch.h"

using namespace mlir;
#include "Ragged/RaggedDialect.cpp.inc"
void ragged::RaggedDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Ragged/RaggedOps.cpp.inc"
      >();
}

namespace {
LogicalResult verifyDataOp(Operation *op, bool softmax, bool legalized) {
  for (NamedAttribute item : op->getAttrs()) {
    StringRef name = item.getName().getValue();
    bool known = llvm::StringSwitch<bool>(name)
        .Cases({"kind", "max_len", "min_len", "offsets_checked"}, true)
        .Cases({"strategy", "alias", "force_fusion", "scale"}, true)
        .Case("debug_visits", true)
        .Cases({"classification", "online_fusion", "warp_mapping", "split_mapping"}, true)
        .Cases({"shuffle", "pointwise_fusion", "row_pipelining", "feature_tile"}, true)
        .Case("vectorize", true)
        .Cases({"chip", "smem_per_cta", "regs_budget"}, true)
        .Default(false);
    if (legalized)
      known |= llvm::StringSwitch<bool>(name)
          .Cases({"operation", "accumulator", "identity", "nan"}, true)
          .Cases({"semantics_version", "d_tile", "d_vector", "mapping"}, true)
          .Cases({"fusion", "class_fusion", "fusion_reason"}, true).Default(false);
    if (!known) return op->emitOpError() << "unsupported attribute '" << name << "'";
  }
  auto values = dyn_cast<MemRefType>(op->getOperand(0).getType());
  auto offsets = dyn_cast<MemRefType>(op->getOperand(1).getType());
  auto out = dyn_cast<MemRefType>(op->getOperand(2).getType());
  if (!values || !out || values.getRank() != 2 || out.getRank() != 2)
    return op->emitOpError("values and out must be rank-2 memrefs");
  if (!offsets || offsets.getRank() != 1)
    return op->emitOpError("offsets must be a rank-1 memref");
  if (!values.getLayout().isIdentity() || !offsets.getLayout().isIdentity() ||
      !out.getLayout().isIdentity())
    return op->emitOpError("requires contiguous identity-layout memrefs");
  for (MemRefType type : {values, offsets, out})
    if (auto space = type.getMemorySpace()) {
      auto integer = dyn_cast<IntegerAttr>(space);
      if (!integer || !integer.getValue().isZero())
        return op->emitOpError("input and output buffers must use the default memory space");
    }
  Type dtype = values.getElementType();
  if (!dtype.isF32() && !dtype.isBF16())
    return op->emitOpError("values dtype must be f32 or bf16");
  if (dtype != out.getElementType())
    return op->emitOpError("out dtype must match values dtype");
  if (!offsets.getElementType().isInteger(32) &&
      !offsets.getElementType().isInteger(64))
    return op->emitOpError("offsets dtype must be i32 or i64");
  int64_t d = values.getDimSize(1);
  if (ShapedType::isDynamic(d) || d < 1 || d > 1024)
    return op->emitOpError("D must be static and in [1, 1024]");
  if (out.getDimSize(1) != d)
    return op->emitOpError("out feature dimension must equal D");
  auto sameIfStatic = [](int64_t a, int64_t b) {
    return ShapedType::isDynamic(a) || ShapedType::isDynamic(b) || a == b;
  };
  if (softmax && !sameIfStatic(values.getDimSize(0), out.getDimSize(0)))
    return op->emitOpError("softmax out leading dimension must equal N");
  if (op->getNumOperands() == 4) {
    if (!softmax)
      return op->emitOpError("additive mask applies only to softmax");
    auto mask = dyn_cast<MemRefType>(op->getOperand(3).getType());
    if (!mask || mask.getRank() != 2 || !mask.getElementType().isF32())
      return op->emitOpError("mask must be a rank-2 f32 memref");
    if (!mask.getLayout().isIdentity())
      return op->emitOpError("mask requires a contiguous identity-layout memref");
    if (Attribute space = mask.getMemorySpace()) {
      auto integer = dyn_cast<IntegerAttr>(space);
      if (!integer || !integer.getValue().isZero())
        return op->emitOpError("mask must use the default memory space");
    }
    if (mask.getDimSize(1) != d || !sameIfStatic(mask.getDimSize(0), values.getDimSize(0)))
      return op->emitOpError("mask shape must match values [N, D]; broadcasting is unsupported");
    if (op->getOperand(3) == op->getOperand(2))
      return op->emitOpError("out must not alias mask");
  }
  int64_t offsetsSize = offsets.getDimSize(0);
  if (!ShapedType::isDynamic(offsetsSize) && offsetsSize < 1)
    return op->emitOpError("offsets must have B+1 entries, including the initial zero");
  if (!softmax && !ShapedType::isDynamic(offsetsSize) &&
      !sameIfStatic(out.getDimSize(0), offsetsSize - 1))
    return op->emitOpError("reduce out leading dimension must equal B");
  for (StringRef name : {"min_len", "max_len"}) {
    if (Attribute attr = op->getAttr(name)) {
      auto integer = dyn_cast<IntegerAttr>(attr);
      if (!integer || !integer.getType().isInteger(64) || integer.getInt() < 0)
        return op->emitOpError() << name << " must be a nonnegative i64 integer";
    }
  }
  auto min = op->getAttrOfType<IntegerAttr>("min_len");
  auto max = op->getAttrOfType<IntegerAttr>("max_len");
  if (min && max && min.getInt() > max.getInt())
    return op->emitOpError("min_len must not exceed max_len");
  for (StringRef name : {"smem_per_cta", "regs_budget"})
    if (Attribute attr = op->getAttr(name)) {
      auto integer = dyn_cast<IntegerAttr>(attr);
      if (!integer || !integer.getType().isInteger(64) || integer.getInt() <= 0)
        return op->emitOpError() << name << " must be a positive i64 integer";
    }
  if (Attribute attr = op->getAttr("chip"); attr && !isa<StringAttr>(attr))
    return op->emitOpError("chip must be a string");
  for (StringRef name : {"offsets_checked", "force_fusion", "debug_visits", "classification",
                         "online_fusion", "warp_mapping", "split_mapping", "shuffle", "pointwise_fusion",
                         "row_pipelining", "vectorize"})
    if (Attribute attr = op->getAttr(name); attr && !isa<BoolAttr>(attr))
      return op->emitOpError() << name << " must be a boolean";
  if (Attribute attr = op->getAttr("feature_tile")) {
    auto tile = dyn_cast<IntegerAttr>(attr);
    if (!tile || !tile.getType().isInteger(64) || tile.getInt() < 1 ||
        tile.getInt() > 32 || (tile.getInt() & (tile.getInt() - 1)))
      return op->emitOpError("feature_tile must be an i64 power of two in [1, 32]");
  }
  auto enabled = [&](StringRef name, bool fallback) {
    auto attr = op->getAttrOfType<BoolAttr>(name);
    return attr ? attr.getValue() : fallback;
  };
  if (enabled("force_fusion", false) && !enabled("online_fusion", true))
    return op->emitOpError("force_fusion conflicts with online_fusion = false");
  if (enabled("debug_visits", false) && !enabled("classification", true))
    return op->emitOpError("debug_visits conflicts with classification = false");
  if (auto force = op->getAttrOfType<BoolAttr>("force_fusion"); force && force.getValue() && !softmax)
    return op->emitOpError("force_fusion applies only to softmax");
  auto alias = op->getAttrOfType<StringAttr>("alias");
  if (op->hasAttr("alias") && (!alias || (alias.getValue() != "none" && alias.getValue() != "inplace")))
    return op->emitOpError("alias must be 'none' or 'inplace'");
  bool inplace = op->getOperand(0) == op->getOperand(2) || (alias && alias.getValue() == "inplace");
  if (op->getNumOperands() == 4 && inplace && op->getOperand(3) == op->getOperand(0))
    return op->emitOpError("in-place output must not alias mask through values");
  if (inplace && !softmax)
    return op->emitOpError("in-place reduction is unsafe; output must not alias values");
  if (op->getOperand(1) == op->getOperand(2))
    return op->emitOpError("out must not alias offsets");
  auto strategy = op->getAttrOfType<StringAttr>("strategy");
  if (op->hasAttr("strategy") &&
      (!strategy || (strategy.getValue() != "auto" && strategy.getValue() != "naive")))
    return op->emitOpError("strategy must be 'auto' or 'naive'");
  if (Attribute attr = op->getAttr("scale")) {
    auto scale = dyn_cast<FloatAttr>(attr);
    if (!scale || !scale.getType().isF32())
      return op->emitOpError("scale must be an f32 floating-point attribute");
  }
  if (legalized) {
    auto accumulator = op->getAttrOfType<TypeAttr>("accumulator");
    if (!accumulator || !accumulator.getValue().isF32())
      return op->emitOpError("legalized accumulator must be f32");
    auto nan = op->getAttrOfType<StringAttr>("nan");
    if (!nan || nan.getValue() != "propagate")
      return op->emitOpError("legalized NaN rule must be 'propagate'");
    auto identity = op->getAttrOfType<StringAttr>("identity");
    auto operation = op->getAttrOfType<StringAttr>("operation");
    StringRef expected = operation.getValue() == "sum" ? "zero" : "negative_infinity";
    if (!identity || identity.getValue() != expected)
      return op->emitOpError("legalized identity does not match the operation");
    auto version = op->getAttrOfType<IntegerAttr>("semantics_version");
    if (!version || !version.getType().isInteger(64) || version.getInt() != 1)
      return op->emitOpError("semantics_version must be 1 : i64");
    if (Attribute attr = op->getAttr("fusion")) {
      auto fusion = dyn_cast<StringAttr>(attr);
      if (!fusion || (fusion.getValue() != "online" && fusion.getValue() != "three_pass" &&
                      fusion.getValue() != "mixed"))
        return op->emitOpError("fusion must be 'online', 'three_pass' or 'mixed'");
    }
    if (Attribute attr = op->getAttr("class_fusion")) {
      auto schedule = dyn_cast<ArrayAttr>(attr);
      if (!schedule || schedule.empty() || !op->getAttr("fusion") ||
          llvm::any_of(schedule, [](Attribute item) {
            auto name = dyn_cast<StringAttr>(item);
            return !name || (name.getValue() != "online" && name.getValue() != "three_pass");
          }))
        return op->emitOpError("class_fusion must list 'online' or 'three_pass' per class, with fusion");
      bool online = llvm::any_of(schedule, [](Attribute item) { return cast<StringAttr>(item).getValue() == "online"; });
      bool threePass = llvm::any_of(schedule, [](Attribute item) { return cast<StringAttr>(item).getValue() == "three_pass"; });
      StringRef expected = online && threePass ? "mixed" : online ? "online" : "three_pass";
      if (op->getAttrOfType<StringAttr>("fusion").getValue() != expected)
        return op->emitOpError("fusion must summarize class_fusion");
    } else if (auto fusion = op->getAttrOfType<StringAttr>("fusion"); fusion && fusion.getValue() == "mixed") {
      return op->emitOpError("mixed fusion requires class_fusion");
    }
  }
  return success();
}
}

LogicalResult ragged::SegmentReduceOp::verify() {
  auto kind = (*this)->getAttrOfType<StringAttr>("kind");
  if (!kind || (kind.getValue() != "sum" && kind.getValue() != "max"))
    return emitOpError("kind must be 'sum' or 'max'");
  return verifyDataOp(*this, false, false);
}
LogicalResult ragged::SegmentSoftmaxOp::verify() {
  return verifyDataOp(*this, true, false);
}
LogicalResult ragged::MapSegmentsOp::verify() {
  auto kind = (*this)->getAttrOfType<StringAttr>("operation");
  if (!kind || (kind.getValue() != "sum" && kind.getValue() != "max" && kind.getValue() != "softmax"))
    return emitOpError("operation must be 'sum', 'max', or 'softmax'");
  if (failed(verifyDataOp(*this, kind.getValue() == "softmax", true)))
    return failure();
  auto schedule = (*this)->getAttrOfType<ArrayAttr>("class_fusion");
  auto dispatch = (*this)->getParentOfType<DispatchOp>();
  if (!schedule || !dispatch) return success();
  auto classes = dispatch->getAttrOfType<ArrayAttr>("classes");
  if (!classes || classes.size() != schedule.size())
    return emitOpError("class_fusion must have one entry per dispatch class");
  return success();
}
LogicalResult ragged::ClassifyOp::verify() {
  auto type = dyn_cast<MemRefType>(getOffsets().getType());
  if (!type || type.getRank() != 1 ||
      (!type.getElementType().isInteger(32) && !type.getElementType().isInteger(64)))
    return emitOpError("offsets must be a rank-1 i32 or i64 memref");
  auto thresholds = (*this)->getAttrOfType<DenseI64ArrayAttr>("thresholds");
  if (!thresholds || thresholds.size() != 2 || thresholds[0] <= 0 ||
      thresholds[0] >= thresholds[1])
    return emitOpError("thresholds must contain two strictly increasing positive lengths");
  return success();
}
LogicalResult ragged::DispatchOp::verify() {
  if (!getBody().hasOneBlock())
    return emitOpError("requires one block");
  auto classes = (*this)->getAttrOfType<ArrayAttr>("classes");
  if (!classes || classes.empty())
    return emitOpError("requires nonempty classes");
  int previous = -1;
  for (Attribute item : classes) {
    auto name = dyn_cast<StringAttr>(item);
    int ordinal = name ? llvm::StringSwitch<int>(name.getValue())
        .Case("warp", 0).Case("cta", 1).Case("split", 2).Default(-1) : -1;
    if (ordinal <= previous)
      return emitOpError("classes must be unique strings ordered warp, cta, split");
    previous = ordinal;
  }
  auto thresholds = (*this)->getAttrOfType<DenseI64ArrayAttr>("thresholds");
  if (!thresholds || thresholds.size() != 2 || thresholds[0] <= 0 || thresholds[0] >= thresholds[1])
    return emitOpError("thresholds must contain two strictly increasing positive lengths");
  for (StringRef name : {"threads", "d_tile", "shared_bytes"}) {
    auto integer = (*this)->getAttrOfType<IntegerAttr>(name);
    if (!integer || !integer.getType().isInteger(64) ||
        (name == "shared_bytes" ? integer.getInt() < 0 : integer.getInt() <= 0))
      return emitOpError() << name << " must be " << (name == "shared_bytes" ? "a nonnegative" : "a positive") << " i64 integer";
  }
  int64_t threads = (*this)->getAttrOfType<IntegerAttr>("threads").getInt();
  int64_t tile = (*this)->getAttrOfType<IntegerAttr>("d_tile").getInt();
  if (threads < 32 || threads > 1024 || (threads & (threads - 1)) ||
      tile > 32 || (tile & (tile - 1)) || threads % tile)
    return emitOpError("threads must be a power of two in [32, 1024]; d_tile must be a power of two in [1, 32] dividing threads");
  if (Attribute attr = (*this)->getAttr("d_vector")) {
    auto vec = dyn_cast<IntegerAttr>(attr);
    if (!vec || !vec.getType().isInteger(64) || vec.getInt() < 1 || vec.getInt() > 8 ||
        (vec.getInt() & (vec.getInt() - 1)))
      return emitOpError("d_vector must be an i64 power of two in [1, 8]");
  }
  if (classes.size() == 1 && cast<StringAttr>(classes[0]).getValue() == "warp" && threads != 32)
    return emitOpError("a direct warp mapping requires exactly 32 threads");
  unsigned maps = 0, classifiers = 0;
  for (Operation &op : getBody().front()) {
    maps += isa<MapSegmentsOp>(op);
    classifiers += isa<ClassifyOp>(op);
    if (!isa<MapSegmentsOp, ClassifyOp, YieldOp>(op))
      return emitOpError("body only permits classify, map_segments and yield");
  }
  if (maps != 1)
    return emitOpError("requires one map and a classifier exactly when device classification is active");
  auto map = *getBody().front().getOps<MapSegmentsOp>().begin();
  auto debug = map->getAttrOfType<BoolAttr>("debug_visits");
  bool deviceClassification = classes.size() > 1 ||
      cast<StringAttr>(classes[0]).getValue() == "split" || (debug && debug.getValue());
  if (classifiers != (deviceClassification ? 1u : 0u))
    return emitOpError("requires one map and a classifier exactly when device classification is active");
  if (Attribute attr = (*this)->getAttr("device_classification")) {
    auto value = dyn_cast<BoolAttr>(attr);
    if (!value || value.getValue() != deviceClassification)
      return emitOpError("device_classification must match classes and debug_visits");
  }
  for (auto classifier : getBody().front().getOps<ClassifyOp>())
    if (classifier.getOffsets() != map.getOffsets() || classifier->getAttr("thresholds") != thresholds)
      return emitOpError("classify must use the mapped offsets and dispatch thresholds");
  return success();
}

#define GET_OP_CLASSES
#include "Ragged/RaggedOps.cpp.inc"
