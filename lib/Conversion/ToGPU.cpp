#include "Ragged/Passes.h"
#include "Ragged/Ragged.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSwitch.h"
#include <limits>

using namespace mlir;
namespace ragged {
namespace {
StringRef str(Operation *op, StringRef name, StringRef fallback = "") {
  auto attr = op->getAttrOfType<StringAttr>(name);
  return attr ? attr.getValue() : fallback;
}
Value index(OpBuilder &b, Location loc, int64_t value) {
  return b.create<arith::ConstantIndexOp>(loc, value);
}
Value fp(OpBuilder &b, Location loc, float value) {
  return b.create<arith::ConstantFloatOp>(loc, b.getF32Type(), APFloat(value));
}
bool enabled(Operation *op, StringRef name) {
  auto attr = op->getAttrOfType<BoolAttr>(name);
  return !attr || attr.getValue();
}
bool hasPointwise(MapSegmentsOp map) {
  auto scale = map->getAttrOfType<FloatAttr>("scale");
  return bool(map.getMask()) || (scale && scale.getValueAsDouble() != 1.0);
}
bool materializesPointwise(MapSegmentsOp map) {
  return hasPointwise(map) && !enabled(map, "pointwise_fusion");
}
Value transformedInput(OpBuilder &b, MapSegmentsOp map, Value values, Value mask,
                       Value row, Value feature) {
  Location loc = map.getLoc();
  Value x = b.create<memref::LoadOp>(loc, values, ValueRange{row, feature});
  if (!x.getType().isF32()) x = b.create<arith::ExtFOp>(loc, b.getF32Type(), x);
  if (auto scale = map->getAttrOfType<FloatAttr>("scale"))
    x = b.create<arith::MulFOp>(loc, x, fp(b, loc, scale.getValueAsDouble()));
  if (mask) {
    Value bias = b.create<memref::LoadOp>(loc, mask, ValueRange{row, feature});
    x = b.create<arith::AddFOp>(loc, x, bias);
  }
  return x;
}
template <typename Fn>
void when(OpBuilder &b, Location loc, Value cond, Fn body) {
  auto ifOp = b.create<scf::IfOp>(loc, cond, false);
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(&ifOp.getThenRegion().front());
  body();
}
template <typename Fn>
Value choose(OpBuilder &b, Location loc, Value cond, Value otherwise, Fn body) {
  auto ifOp = b.create<scf::IfOp>(loc, TypeRange{otherwise.getType()}, cond, true);
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(&ifOp.getThenRegion().front());
  b.create<scf::YieldOp>(loc, body());
  b.setInsertionPointToStart(&ifOp.getElseRegion().front());
  b.create<scf::YieldOp>(loc, otherwise);
  return ifOp.getResult(0);
}

DictionaryAttr emitMaterialize(OpBuilder &b, gpu::GPUModuleOp gpuModule,
                               MapSegmentsOp map, unsigned id, ArrayRef<Type> types,
                               ArrayAttr names, unsigned logitsIndex) {
  Location loc = map.getLoc();
  b.setInsertionPointToStart(gpuModule.getBody());
  std::string name = "ragged_" + std::to_string(id) + "_materialize";
  auto function = b.create<gpu::GPUFuncOp>(loc, name, b.getFunctionType(types, {}));
  function.setKernelAttr(b.getUnitAttr());
  b.setInsertionPointToStart(&function.getBody().front());
  Value values = function.getArgument(0), logits = function.getArgument(logitsIndex);
  Value mask = map.getMask() ? Value(function.getArgument(3)) : Value();
  Value d = index(b, loc, cast<MemRefType>(values.getType()).getDimSize(1));
  Value n = b.create<memref::DimOp>(loc, values, 0);
  Value count = b.create<arith::MulIOp>(loc, n, d);
  Value tid = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value block = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value blockSize = index(b, loc, 128);
  Value first = b.create<arith::AddIOp>(loc, tid, b.create<arith::MulIOp>(loc, block, blockSize));
  Value grid = b.create<gpu::GridDimOp>(loc, gpu::Dimension::x);
  Value step = b.create<arith::MulIOp>(loc, grid, blockSize);
  b.create<scf::ForOp>(loc, first, count, step, ValueRange{},
    [&](OpBuilder &nested, Location at, Value linear, ValueRange) {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPoint(nested.getInsertionBlock(), nested.getInsertionPoint());
      Value row = b.create<arith::DivUIOp>(loc, linear, d);
      Value feature = b.create<arith::RemUIOp>(loc, linear, d);
      Value value = transformedInput(b, map, values, mask, row, feature);
      b.create<memref::StoreOp>(loc, value, logits, ValueRange{row, feature});
      b.create<scf::YieldOp>(loc);
    });
  b.create<gpu::ReturnOp>(loc);
  return b.getDictionaryAttr({
      b.getNamedAttr("name", b.getStringAttr(name)),
      b.getNamedAttr("role", b.getStringAttr("materialize")),
      b.getNamedAttr("grid", b.getArrayAttr({b.getI64IntegerAttr(256), b.getI64IntegerAttr(1), b.getI64IntegerAttr(1)})),
      b.getNamedAttr("block", b.getDenseI64ArrayAttr({128, 1, 1})),
      b.getNamedAttr("args", names),
      b.getNamedAttr("shared_bytes", b.getI64IntegerAttr(0)),
      b.getNamedAttr("static_shared_bytes", b.getI64IntegerAttr(0))});
}

// The baseline maps one CTA to one (segment, feature tile). Every barrier is
// reached by every CTA thread, including threads in a partial feature tile.
class CTAEmitter {
public:
  CTAEmitter(OpBuilder &builder, MapSegmentsOp source, int64_t threads,
             int64_t tile, ValueRange args, Value shared,
             Value segmentOverride = {}, Value maximumStats = {},
             Value totalStats = {}, Value partialBuffer = {},
             Value partId = {}, bool useShuffle = false, Value mask = {}, Value logits = {},
             Value partialSumBuffer = {})
      : b(builder), loc(source.getLoc()), source(source), threads(threads),
        tile(tile), useShuffle(useShuffle), values(args[0]), offsets(args[1]),
        out(args[2]), shared(shared), maxStats(maximumStats), sumStats(totalStats),
        partials(partialBuffer), partialSums(partialSumBuffer), part(partId), mask(mask), logits(logits) {
    zero = index(b, loc, 0); one = index(b, loc, 1);
    tid = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
    segment = segmentOverride ? segmentOverride : Value(b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x));
    Value dBlock = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::y);
    Value tileValue = index(b, loc, tile);
    featureLane = b.create<arith::RemUIOp>(loc, tid, tileValue);
    rowLane = b.create<arith::DivUIOp>(loc, tid, tileValue);
    feature = b.create<arith::AddIOp>(loc, featureLane,
        b.create<arith::MulIOp>(loc, dBlock, tileValue));
    int64_t d = cast<MemRefType>(values.getType()).getDimSize(1);
    // When the tile divides D every launched feature tile is full, so the
    // guard is a constant and loads need no branch around them.
    validFeature = d % tile == 0 ? Value(b.create<arith::ConstantIntOp>(loc, 1, 1))
        : Value(b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, feature, index(b, loc, d)));
    unroll = enabled(source, "row_pipelining") ? kRowUnroll : 1;
    Value next = b.create<arith::AddIOp>(loc, segment, one);
    Value startOffset = b.create<memref::LoadOp>(loc, offsets, ValueRange{segment});
    Value endOffset = b.create<memref::LoadOp>(loc, offsets, ValueRange{next});
    start = b.create<arith::IndexCastOp>(loc, b.getIndexType(), startOffset);
    end = b.create<arith::IndexCastOp>(loc, b.getIndexType(), endOffset);
    if (part) {
      // Quotient/remainder partitioning avoids overflowing length + 7 and
      // covers every row exactly once, including segments shorter than eight.
      Value length = b.create<arith::SubIOp>(loc, end, start);
      Value eight = index(b, loc, 8);
      Value quotient = b.create<arith::DivUIOp>(loc, length, eight);
      Value remainder = b.create<arith::RemUIOp>(loc, length, eight);
      Value extra = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, part, remainder);
      Value prefix = b.create<arith::SelectOp>(loc, extra, part, remainder);
      Value lengthPart = b.create<arith::AddIOp>(loc, quotient,
          b.create<arith::SelectOp>(loc, extra, one, zero));
      Value offsetPart = b.create<arith::AddIOp>(loc,
          b.create<arith::MulIOp>(loc, part, quotient), prefix);
      start = b.create<arith::AddIOp>(loc, start, offsetPart);
      end = b.create<arith::AddIOp>(loc, start, lengthPart);
    }
    firstRow = b.create<arith::AddIOp>(loc, start, rowLane);
    rowStep = index(b, loc, threads / tile);
  }

  void emit(StringRef phase) {
    if (phase == "partial_online") {
      // A split part's reference and mass, merged by combine_online.
      auto stats = onlineStatistics();
      int64_t d = cast<MemRefType>(values.getType()).getDimSize(1);
      Value column = b.create<arith::AddIOp>(loc, feature,
          b.create<arith::MulIOp>(loc, part, index(b, loc, d)));
      storeLeader(stats.first, partials, false, column);
      storeLeader(stats.second, partialSums, false, column);
    } else if (phase == "combine_online") {
      combineOnline();
    } else if (phase.starts_with("combine_")) {
      combinePartials(phase.drop_front(8));
    } else if (phase.starts_with("partial_")) {
      StringRef kind = phase.drop_front(8);
      Value maximum = kind == "exp_sum" ? loadStat(maxStats) : Value();
      Value result = reduce(kind, maximum);
      int64_t d = cast<MemRefType>(values.getType()).getDimSize(1);
      Value column = b.create<arith::AddIOp>(loc, feature,
          b.create<arith::MulIOp>(loc, part, index(b, loc, d)));
      storeLeader(result, partials, false, column);
    } else if (phase == "sum" || phase == "max") {
      Value result = reduce(phase, Value());
      storeLeader(result, out, true);
    } else if (phase == "max_stats") {
      storeLeader(reduce("max", Value()), maxStats, false);
    } else if (phase == "sum_stats") {
      Value maximum = loadStat(maxStats);
      storeLeader(reduce("exp_sum", maximum), sumStats, false);
    } else if (phase == "normalize") {
      normalize(loadStat(maxStats), loadStat(sumStats));
    } else if (phase == "online") {
      auto stats = onlineStatistics();
      normalize(stats.first, stats.second);
    } else if (phase == "softmax") {
      Value maximum = reduce("max", Value());
      // All consumers finish the max read before shared storage is reused.
      b.create<gpu::BarrierOp>(loc);
      Value total = reduce("exp_sum", maximum);
      normalize(maximum, total);
    }
  }

private:
  static constexpr int64_t kRowUnroll = 8;

  // A lane visits rows firstRow, firstRow + rowStep, ... below end. Each
  // main-loop iteration issues `unroll` loads before folding them in visit
  // order, so the arithmetic is that of the rolled loop while several loads
  // are in flight per lane; a rolled loop covers the last count % unroll rows.
  SmallVector<Value> forRows(ValueRange initial, function_ref<Value(Value)> fetch,
                             function_ref<SmallVector<Value>(ValueRange, Value, Value)> fold) {
    int64_t step = threads / tile;
    auto body = [&](int64_t width) {
      return [&, width](OpBuilder &nested, Location at, Value row, ValueRange iter) {
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPoint(nested.getInsertionBlock(), nested.getInsertionPoint());
        SmallVector<Value> rows{row}, fetched;
        for (int64_t k = 1; k < width; ++k)
          rows.push_back(b.create<arith::AddIOp>(loc, row, index(b, loc, k * step)));
        for (Value r : rows) fetched.push_back(fetch(r));
        SmallVector<Value> state(iter.begin(), iter.end());
        for (int64_t k = 0; k < width; ++k) state = fold(state, fetched[k], rows[k]);
        b.create<scf::YieldOp>(loc, state);
      };
    };
    if (unroll == 1)
      return b.create<scf::ForOp>(loc, firstRow, end, rowStep, initial, body(1)).getResults();
    Value length = b.create<arith::MaxSIOp>(loc, b.create<arith::SubIOp>(loc, end, firstRow), zero);
    Value count = b.create<arith::AddIOp>(loc, b.create<arith::DivUIOp>(loc, length, rowStep),
        b.create<arith::SelectOp>(loc, b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne,
            b.create<arith::RemUIOp>(loc, length, rowStep), zero), one, zero));
    Value mainEnd = b.create<arith::AddIOp>(loc, firstRow, b.create<arith::MulIOp>(loc,
        b.create<arith::DivUIOp>(loc, count, index(b, loc, unroll)), index(b, loc, unroll * step)));
    auto main = b.create<scf::ForOp>(loc, firstRow, mainEnd, index(b, loc, unroll * step),
                                     initial, body(unroll));
    return b.create<scf::ForOp>(loc, mainEnd, end, rowStep, main.getResults(), body(1)).getResults();
  }
  std::pair<Value, Value> compensatedAdd(Value total, Value error, Value value) {
    Value corrected = b.create<arith::SubFOp>(loc, value, error);
    Value next = b.create<arith::AddFOp>(loc, total, corrected);
    Value roundedIncrement = b.create<arith::SubFOp>(loc, next, total);
    Value rawError = b.create<arith::SubFOp>(loc, roundedIncrement, corrected);
    Value magnitude = b.create<math::AbsFOp>(loc, next);
    Value finite = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OLT,
        magnitude, fp(b, loc, std::numeric_limits<float>::infinity()));
    Value correction = b.create<arith::SelectOp>(loc, finite, rawError, fp(b, loc, 0.0f));
    return {next, correction};
  }
  void combinePartials(StringRef phase) {
    bool maximum = phase == "max" || phase == "max_stats";
    bool stats = phase.ends_with("_stats");
    Value destination = stats ? (maximum ? maxStats : sumStats) : out;
    Value leader = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, rowLane, zero);
    Value active = b.create<arith::AndIOp>(loc, leader, validFeature);
    when(b, loc, active, [&] {
      Value identity = fp(b, loc, maximum ? -std::numeric_limits<float>::infinity() : 0.0f);
      SmallVector<Value> initial{identity};
      if (!maximum) initial.push_back(identity);
      int64_t d = cast<MemRefType>(values.getType()).getDimSize(1);
      auto loop = b.create<scf::ForOp>(loc, zero, index(b, loc, 8), one, initial,
        [&](OpBuilder &nested, Location at, Value p, ValueRange iter) {
          OpBuilder::InsertionGuard guard(b);
          b.setInsertionPoint(nested.getInsertionBlock(), nested.getInsertionPoint());
          Value column = b.create<arith::AddIOp>(loc, feature,
              b.create<arith::MulIOp>(loc, p, index(b, loc, d)));
          Value value = b.create<memref::LoadOp>(loc, partials, ValueRange{segment, column});
          if (maximum) {
            b.create<scf::YieldOp>(loc, b.create<arith::MaximumFOp>(loc, iter[0], value).getResult());
          } else {
            auto result = compensatedAdd(iter[0], iter[1], value);
            b.create<scf::YieldOp>(loc, ValueRange{result.first, result.second});
          }
        });
      Value result = loop.getResult(0);
      Type dtype = cast<MemRefType>(destination.getType()).getElementType();
      if (!dtype.isF32()) result = b.create<arith::TruncFOp>(loc, dtype, result);
      b.create<memref::StoreOp>(loc, result, destination, ValueRange{segment, feature});
    });
  }
  // Merge the eight parts' (reference, mass) pairs: the largest reference
  // first, then the compensated sum of the masses rescaled to it. Parts with
  // a -inf reference (empty or all -inf) have no mass; a NaN reference makes
  // the sum NaN and a +inf one is a NaN feature, as in the CTA tree.
  void combineOnline() {
    Value leader = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, rowLane, zero);
    Value active = b.create<arith::AndIOp>(loc, leader, validFeature);
    when(b, loc, active, [&] {
      int64_t d = cast<MemRefType>(values.getType()).getDimSize(1);
      Value minusInf = fp(b, loc, -std::numeric_limits<float>::infinity());
      Value zeroFloat = fp(b, loc, 0.0f);
      auto column = [&](Value p) {
        return b.create<arith::AddIOp>(loc, feature, b.create<arith::MulIOp>(loc, p, index(b, loc, d)));
      };
      auto overParts = [&](ValueRange initial, function_ref<SmallVector<Value>(ValueRange, Value)> fold) {
        return b.create<scf::ForOp>(loc, zero, index(b, loc, 8), one, initial,
          [&](OpBuilder &nested, Location at, Value p, ValueRange iter) {
            OpBuilder::InsertionGuard guard(b);
            b.setInsertionPoint(nested.getInsertionBlock(), nested.getInsertionPoint());
            b.create<scf::YieldOp>(loc, fold(iter, column(p)));
          }).getResults();
      };
      Value reference = overParts(ValueRange{minusInf}, [&](ValueRange iter, Value at) -> SmallVector<Value> {
        Value partReference = b.create<memref::LoadOp>(loc, partials, ValueRange{segment, at});
        return {b.create<arith::MaximumFOp>(loc, iter[0], partReference)};
      })[0];
      Value total = overParts(ValueRange{zeroFloat, zeroFloat}, [&](ValueRange iter, Value at) -> SmallVector<Value> {
        Value partReference = b.create<memref::LoadOp>(loc, partials, ValueRange{segment, at});
        Value mass = b.create<memref::LoadOp>(loc, partialSums, ValueRange{segment, at});
        Value scaled = b.create<arith::MulFOp>(loc, mass, b.create<math::ExpOp>(loc,
            b.create<arith::SubFOp>(loc, partReference, reference)));
        Value empty = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OEQ, partReference, minusInf);
        auto sum = compensatedAdd(iter[0], iter[1], b.create<arith::SelectOp>(loc, empty, zeroFloat, scaled));
        return {sum.first, sum.second};
      })[0];
      Value positiveInfinity = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OEQ,
          reference, fp(b, loc, std::numeric_limits<float>::infinity()));
      total = b.create<arith::SelectOp>(loc, positiveInfinity,
          fp(b, loc, std::numeric_limits<float>::quiet_NaN()), total);
      b.create<memref::StoreOp>(loc, reference, maxStats, ValueRange{segment, feature});
      b.create<memref::StoreOp>(loc, total, sumStats, ValueRange{segment, feature});
    });
  }
  Value broadcastWarp(Value value) {
    Value lane = b.create<arith::IndexCastOp>(loc, b.getI32Type(), featureLane);
    Value width = b.create<arith::ConstantIntOp>(loc, 32, 32);
    return b.create<gpu::ShuffleOp>(loc, value, lane, width, gpu::ShuffleMode::IDX).getShuffleResult();
  }
  std::pair<Value, Value> mergeStatistics(Value leftMax, Value leftSum,
                                        Value rightMax, Value rightSum) {
    Value minusInf = fp(b, loc, -std::numeric_limits<float>::infinity());
    Value mergedMax = b.create<arith::MaximumFOp>(loc, leftMax, rightMax);
    Value leftScale = b.create<math::ExpOp>(loc,
        b.create<arith::SubFOp>(loc, leftMax, mergedMax));
    Value rightScale = b.create<math::ExpOp>(loc,
        b.create<arith::SubFOp>(loc, rightMax, mergedMax));
    Value mergedSum = b.create<arith::AddFOp>(loc,
        b.create<arith::MulFOp>(loc, leftSum, leftScale),
        b.create<arith::MulFOp>(loc, rightSum, rightScale));
    // -inf elements have zero mass. Avoid (-inf)-(-inf) contaminating a
    // later finite element; an entirely -inf feature still normalizes to NaN.
    Value leftEmpty = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OEQ, leftMax, minusInf);
    Value rightEmpty = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OEQ, rightMax, minusInf);
    mergedSum = b.create<arith::SelectOp>(loc, leftEmpty, rightSum, mergedSum);
    mergedSum = b.create<arith::SelectOp>(loc, rightEmpty, leftSum, mergedSum);
    Value positiveInfinity = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OEQ,
        mergedMax, fp(b, loc, std::numeric_limits<float>::infinity()));
    mergedSum = b.create<arith::SelectOp>(loc, positiveInfinity,
        fp(b, loc, std::numeric_limits<float>::quiet_NaN()), mergedSum);
    return {mergedMax, mergedSum};
  }
  std::pair<Value, Value> onlineStatistics() {
    Value minusInf = fp(b, loc, -std::numeric_limits<float>::infinity());
    Value zeroFloat = fp(b, loc, 0.0f);
    Value oneFloat = fp(b, loc, 1.0f);
    auto loop = forRows(ValueRange{minusInf, zeroFloat, zeroFloat},
      [&](Value row) { return choose(b, loc, validFeature, minusInf, [&] { return loadInput(row); }); },
      [&](ValueRange iter, Value x, Value) -> SmallVector<Value> {
        // Keep a stable exponential reference instead of rescaling at every
        // new maximum. Repeated, nearly-unit exp(oldMax-newMax) factors lose
        // accuracy even with compensated addition. A reference may lag the
        // maximum by at most one: individual masses are bounded by exp(1),
        // and every actual rescaling damps earlier rounding errors by < 1/e.
        Value maximum = b.create<arith::MaximumFOp>(loc, iter[0], x);
        Value empty = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OEQ, iter[0], minusInf);
        Value advance = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT,
            b.create<arith::SubFOp>(loc, maximum, iter[0]), oneFloat);
        Value unordered = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::UNO, iter[0], x);
        Value recenter = b.create<arith::OrIOp>(loc, empty,
            b.create<arith::OrIOp>(loc, advance, unordered));
        Value anchor = b.create<arith::SelectOp>(loc, recenter, maximum, iter[0]);
        Value scale = b.create<math::ExpOp>(loc,
            b.create<arith::SubFOp>(loc, iter[0], anchor));
        Value total = b.create<arith::SelectOp>(loc, empty, zeroFloat,
            b.create<arith::MulFOp>(loc, iter[1], scale));
        Value error = b.create<arith::SelectOp>(loc, empty, zeroFloat,
            b.create<arith::MulFOp>(loc, iter[2], scale));
        Value absent = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OEQ, x, minusInf);
        Value mass = b.create<arith::SelectOp>(loc, absent, zeroFloat,
            b.create<math::ExpOp>(loc, b.create<arith::SubFOp>(loc, x, anchor)));
        auto accumulated = compensatedAdd(total, error, mass);
        Value positiveInfinity = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OEQ,
            anchor, fp(b, loc, std::numeric_limits<float>::infinity()));
        total = b.create<arith::SelectOp>(loc, positiveInfinity,
            fp(b, loc, std::numeric_limits<float>::quiet_NaN()), accumulated.first);
        error = b.create<arith::SelectOp>(loc, positiveInfinity, zeroFloat, accumulated.second);
        return {anchor, total, error};
      });
    if (useShuffle) {
      Value maximum = loop[0], total = loop[1];
      for (int64_t distance = (threads / tile) / 2; distance > 0; distance /= 2) {
        Value otherMax = b.create<gpu::ShuffleOp>(loc, maximum,
            static_cast<int32_t>(distance * tile), 32, gpu::ShuffleMode::DOWN).getShuffleResult();
        Value otherSum = b.create<gpu::ShuffleOp>(loc, total,
            static_cast<int32_t>(distance * tile), 32, gpu::ShuffleMode::DOWN).getShuffleResult();
        Value active = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult,
                                            rowLane, index(b, loc, distance));
        auto merged = mergeStatistics(maximum, total, otherMax, otherSum);
        maximum = b.create<arith::SelectOp>(loc, active, merged.first, maximum);
        total = b.create<arith::SelectOp>(loc, active, merged.second, total);
      }
      return {broadcastWarp(maximum), broadcastWarp(total)};
    }
    Value sumIndex = b.create<arith::AddIOp>(loc, tid, index(b, loc, threads));
    b.create<memref::StoreOp>(loc, loop[0], shared, ValueRange{tid});
    b.create<memref::StoreOp>(loc, loop[1], shared, ValueRange{sumIndex});
    b.create<gpu::BarrierOp>(loc);
    for (int64_t distance = (threads / tile) / 2; distance > 0; distance /= 2) {
      Value active = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult,
                                          rowLane, index(b, loc, distance));
      when(b, loc, active, [&] {
        Value other = b.create<arith::AddIOp>(loc, tid, index(b, loc, distance * tile));
        Value otherSum = b.create<arith::AddIOp>(loc, other, index(b, loc, threads));
        Value lhsM = b.create<memref::LoadOp>(loc, shared, ValueRange{tid});
        Value lhsS = b.create<memref::LoadOp>(loc, shared, ValueRange{sumIndex});
        Value rhsM = b.create<memref::LoadOp>(loc, shared, ValueRange{other});
        Value rhsS = b.create<memref::LoadOp>(loc, shared, ValueRange{otherSum});
        auto merged = mergeStatistics(lhsM, lhsS, rhsM, rhsS);
        b.create<memref::StoreOp>(loc, merged.first, shared, ValueRange{tid});
        b.create<memref::StoreOp>(loc, merged.second, shared, ValueRange{sumIndex});
      });
      b.create<gpu::BarrierOp>(loc);
    }
    Value leaderSum = b.create<arith::AddIOp>(loc, featureLane, index(b, loc, threads));
    return {b.create<memref::LoadOp>(loc, shared, ValueRange{featureLane}),
            b.create<memref::LoadOp>(loc, shared, ValueRange{leaderSum})};
  }
  Value loadInput(Value row) {
    if (logits) return b.create<memref::LoadOp>(loc, logits, ValueRange{row, feature});
    return transformedInput(b, source, values, mask, row, feature);
  }
  Value loadStat(Value buffer) {
    return choose(b, loc, validFeature, fp(b, loc, 0), [&] {
      return Value(b.create<memref::LoadOp>(loc, buffer, ValueRange{segment, feature}));
    });
  }
  Value combine(StringRef kind, Value lhs, Value rhs) {
    if (kind == "max") return b.create<arith::MaximumFOp>(loc, lhs, rhs);
    return b.create<arith::AddFOp>(loc, lhs, rhs);
  }
  Value reduce(StringRef kind, Value maximum) {
    Value identity = fp(b, loc, kind == "max" ? -std::numeric_limits<float>::infinity() : 0.0f);
    // A lane may visit thousands of rows before the CTA tree. Ordinary serial
    // f32 addition can exceed the shared oracle tolerance even for repeated
    // 0.1 inputs, so carry the rounding error through the lane traversal.
    bool compensated = kind != "max";
    SmallVector<Value> initial{identity};
    Value positiveInfinity;
    if (compensated) {
      initial.push_back(identity);
      positiveInfinity = fp(b, loc, std::numeric_limits<float>::infinity());
    }
    auto loop = forRows(initial,
      [&](Value row) {
        return choose(b, loc, validFeature, identity, [&] {
          Value x = loadInput(row);
          if (kind == "exp_sum") {
            x = b.create<arith::SubFOp>(loc, x, maximum);
            x = b.create<math::ExpOp>(loc, x);
          }
          return x;
        });
      },
      [&](ValueRange iter, Value value, Value) -> SmallVector<Value> {
        if (!compensated) return {combine(kind, iter[0], value)};
        Value corrected = b.create<arith::SubFOp>(loc, value, iter[1]);
        Value total = b.create<arith::AddFOp>(loc, iter[0], corrected);
        Value roundedIncrement = b.create<arith::SubFOp>(loc, total, iter[0]);
        Value error = b.create<arith::SubFOp>(loc, roundedIncrement, corrected);
        Value magnitude = b.create<math::AbsFOp>(loc, total);
        Value finite = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OLT,
                                             magnitude, positiveInfinity);
        // inf - inf in the compensation expression is NaN. Clearing the
        // correction for non-finite totals preserves +inf + finite = +inf,
        // while +inf + -inf and NaN inputs still poison the total itself.
        Value compensation = b.create<arith::SelectOp>(loc, finite, error, identity);
        return {total, compensation};
      });
    if (useShuffle) {
      Value result = loop[0];
      for (int64_t distance = (threads / tile) / 2; distance > 0; distance /= 2) {
        Value other = b.create<gpu::ShuffleOp>(loc, result,
            static_cast<int32_t>(distance * tile), 32, gpu::ShuffleMode::DOWN).getShuffleResult();
        Value active = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult,
                                            rowLane, index(b, loc, distance));
        result = b.create<arith::SelectOp>(loc, active, combine(kind, result, other), result);
      }
      return broadcastWarp(result);
    }
    b.create<memref::StoreOp>(loc, loop[0], shared, ValueRange{tid});
    b.create<gpu::BarrierOp>(loc);
    for (int64_t distance = (threads / tile) / 2; distance > 0; distance /= 2) {
      Value active = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult,
                                          rowLane, index(b, loc, distance));
      when(b, loc, active, [&] {
        Value other = b.create<arith::AddIOp>(loc, tid, index(b, loc, distance * tile));
        Value lhs = b.create<memref::LoadOp>(loc, shared, ValueRange{tid});
        Value rhs = b.create<memref::LoadOp>(loc, shared, ValueRange{other});
        b.create<memref::StoreOp>(loc, combine(kind, lhs, rhs), shared, ValueRange{tid});
      });
      b.create<gpu::BarrierOp>(loc);
    }
    return b.create<memref::LoadOp>(loc, shared, ValueRange{featureLane});
  }
  void storeLeader(Value result, Value buffer, bool castOutput, Value column = {}) {
    Value leader = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, rowLane, zero);
    Value active = b.create<arith::AndIOp>(loc, leader, validFeature);
    when(b, loc, active, [&] {
      if (castOutput && !cast<MemRefType>(out.getType()).getElementType().isF32())
        result = b.create<arith::TruncFOp>(loc, cast<MemRefType>(out.getType()).getElementType(), result);
      b.create<memref::StoreOp>(loc, result, buffer, ValueRange{segment, column ? column : feature});
    });
  }
  void normalize(Value maximum, Value total) {
    // Every element is read and then written by the same lane, so loading
    // later rows before storing earlier ones keeps in-place softmax exact.
    forRows(ValueRange{},
      [&](Value row) { return choose(b, loc, validFeature, fp(b, loc, 0), [&] { return loadInput(row); }); },
      [&](ValueRange, Value input, Value row) -> SmallVector<Value> {
        when(b, loc, validFeature, [&] {
          Value value = b.create<arith::SubFOp>(loc, input, maximum);
          value = b.create<math::ExpOp>(loc, value);
          value = b.create<arith::DivFOp>(loc, value, total);
          Type dtype = cast<MemRefType>(out.getType()).getElementType();
          if (!dtype.isF32()) value = b.create<arith::TruncFOp>(loc, dtype, value);
          b.create<memref::StoreOp>(loc, value, out, ValueRange{row, feature});
        });
        return {};
      });
  }
  OpBuilder &b;
  Location loc;
  MapSegmentsOp source;
  int64_t threads, tile, unroll;
  bool useShuffle;
  Value values, offsets, out, shared, maxStats, sumStats, partials, partialSums, part, mask, logits;
  Value zero, one, tid, segment, featureLane, rowLane, feature, validFeature;
  Value start, end, firstRow, rowStep;
};

// Scheduling and workspace ownership are separated from the arithmetic
// emitter. Every persistent CTA consumes one immutable, device-built queue;
// all inter-CTA dependencies are expressed by manifest kernel order.
class ClassifiedEmitter {
public:
  ClassifiedEmitter(OpBuilder &builder, gpu::GPUModuleOp gpuModule,
                    DispatchOp dispatch, unsigned id,
                    SmallVectorImpl<Attribute> &manifest)
      : b(builder), gpuModule(gpuModule), dispatch(dispatch), loc(dispatch.getLoc()),
        id(id), manifest(manifest) {
    map = *dispatch.getBody().front().getOps<MapSegmentsOp>().begin();
    loc = map.getLoc();
    d = cast<MemRefType>(map.getValues().getType()).getDimSize(1);
    tile = dispatch->getAttrOfType<IntegerAttr>("d_tile").getInt();
    operation = str(map, "operation");
    auto debug = map->getAttrOfType<BoolAttr>("debug_visits");
    debugVisits = debug && debug.getValue();
    auto schedule = map->getAttrOfType<ArrayAttr>("class_fusion");
    for (auto [index, attr] : llvm::enumerate(dispatch->getAttrOfType<ArrayAttr>("classes"))) {
      classes.push_back(cast<StringAttr>(attr).getValue());
      StringRef fusion = schedule ? cast<StringAttr>(schedule[index]).getValue() : str(map, "fusion");
      online.push_back(operation == "softmax" && fusion == "online");
    }
    hasSplit = llvm::is_contained(classes, "split");
    types.append(map->getOperandTypes().begin(), map->getOperandTypes().end());
    names = {b.getStringAttr("values"), b.getStringAttr("offsets"), b.getStringAttr("out")};
    if (map.getMask()) names.push_back(b.getStringAttr("mask"));
    addWorkspace("class_counts", "i64", b.getI64Type(), {3});
    addWorkspace("segment_ids", "i64", b.getI64Type(), {3, ShapedType::kDynamic});
    if (hasSplit) addWorkspace("partials", "f32", b.getF32Type(), {ShapedType::kDynamic, 8 * d});
    // Split is the last class; its online parts keep a mass beside each reference.
    if (operation == "softmax" && hasSplit && online.back())
      addWorkspace("partial_sums", "f32", b.getF32Type(), {ShapedType::kDynamic, 8 * d});
    if (operation == "softmax" && (hasSplit || llvm::is_contained(online, false))) {
      addWorkspace("max_stats", "f32", b.getF32Type(), {ShapedType::kDynamic, d});
      addWorkspace("sum_stats", "f32", b.getF32Type(), {ShapedType::kDynamic, d});
    }
    if (debugVisits) addWorkspace("visits", "i32", b.getI32Type(), {ShapedType::kDynamic});
    if (materializesPointwise(map))
      addWorkspace("logits", "f32", b.getF32Type(), {ShapedType::kDynamic, d}, "N");
  }

  ArrayAttr emit() {
    if (materializesPointwise(map))
      manifest.push_back(emitMaterialize(b, gpuModule, map, id, types, b.getArrayAttr(names),
                                        argumentIndices.lookup("logits")));
    emitKernel("init", "", "init", 128, 1, 0);
    emitKernel("classify", "", "classify", 128, 1, 0);
    for (auto [segmentClass, fused] : llvm::zip(classes, online)) {
      if (segmentClass == "split") {
        if (operation == "softmax" && fused) {
          emitWorker(segmentClass, "partial_online");
          emitWorker(segmentClass, "combine_online");
          emitWorker(segmentClass, "normalize");
        } else if (operation == "softmax") {
          emitWorker(segmentClass, "partial_max");
          emitWorker(segmentClass, "combine_max_stats");
          emitWorker(segmentClass, "partial_exp_sum");
          emitWorker(segmentClass, "combine_sum_stats");
          emitWorker(segmentClass, "normalize");
        } else {
          emitWorker(segmentClass, ("partial_" + operation).str());
          emitWorker(segmentClass, ("combine_" + operation).str());
        }
      } else if (operation == "softmax") {
        if (fused) emitWorker(segmentClass, "online");
        else {
          emitWorker(segmentClass, "max_stats");
          emitWorker(segmentClass, "sum_stats");
          emitWorker(segmentClass, "normalize");
        }
      } else {
        emitWorker(segmentClass, operation);
      }
    }
    return b.getArrayAttr(workspace);
  }

private:
  void addWorkspace(StringRef name, StringRef dtype, Type element,
                    ArrayRef<int64_t> shape, StringRef dynamicDimension = "B") {
    argumentIndices[name] = types.size();
    names.push_back(b.getStringAttr(name));
    types.push_back(MemRefType::get(shape, element));
    SmallVector<Attribute> dimensions;
    for (int64_t size : shape)
      dimensions.push_back(size == ShapedType::kDynamic ? Attribute(b.getStringAttr(dynamicDimension))
                                                       : Attribute(b.getI64IntegerAttr(size)));
    workspace.push_back(b.getDictionaryAttr({
        b.getNamedAttr("name", b.getStringAttr(name)),
        b.getNamedAttr("dtype", b.getStringAttr(dtype)),
        b.getNamedAttr("shape", b.getArrayAttr(dimensions))}));
  }
  Value argument(StringRef name) {
    auto found = argumentIndices.find(name);
    return found == argumentIndices.end() ? Value() : function.getArgument(found->second);
  }
  Value segmentCount() {
    Value size = b.create<memref::DimOp>(loc, function.getArgument(1), 0);
    return b.create<arith::SubIOp>(loc, size, index(b, loc, 1));
  }
  static int64_t classId(StringRef name) {
    return llvm::StringSwitch<int64_t>(name).Case("warp", 0).Case("cta", 1).Case("split", 2).Default(-1);
  }
  void emitWorker(StringRef segmentClass, StringRef phase) {
    int64_t threads = segmentClass == "warp" ? 32 : 128;
    bool splitRange = segmentClass == "split" &&
                      (phase.starts_with("partial_") || phase == "normalize");
    int64_t shared = (segmentClass == "warp" && enabled(map, "shuffle")) || phase.starts_with("combine_") || phase == "normalize"
        ? 0 : threads * (phase == "online" || phase == "partial_online" ? 2 : 1);
    std::string role = (segmentClass + "_" + phase).str();
    emitKernel(role, segmentClass, phase, threads, splitRange ? 8 : 1, shared);
  }
  void emitKernel(StringRef role, StringRef segmentClass, StringRef phase,
                  int64_t threads, int64_t parts, int64_t sharedElements) {
    b.setInsertionPointToStart(gpuModule.getBody());
    std::string name = "ragged_" + std::to_string(id) + "_" + role.str();
    function = b.create<gpu::GPUFuncOp>(loc, name, b.getFunctionType(types, {}));
    // LLVM 23 models this as an inherent property. The legacy discardable
    // gpu.kernel is copied then appended again by GPUFuncOpLowering, producing
    // duplicate dictionary keys in an assertions-enabled SDK.
    function.setKernelAttr(b.getUnitAttr());
    Value shared;
    if (sharedElements) {
      auto space = gpu::AddressSpaceAttr::get(b.getContext(), gpu::AddressSpace::Workgroup);
      auto type = MemRefType::get({sharedElements}, b.getF32Type(), MemRefLayoutAttrInterface{}, space);
      shared = function.addWorkgroupAttribution(type, loc);
    }
    b.setInsertionPointToStart(&function.getBody().front());
    if (phase == "init") emitInit();
    else if (phase == "classify") emitClassify();
    else emitPersistent(segmentClass, phase, threads, parts, shared);
    b.create<gpu::ReturnOp>(loc);
    int64_t staticShared = 0;
    for (BlockArgument attribution : function.getWorkgroupAttributionBBArgs()) {
      auto type = cast<MemRefType>(attribution.getType());
      staticShared += type.getNumElements() * type.getElementTypeBitWidth() / 8;
    }
    Attribute gridX = phase == "init" ? Attribute(b.getI64IntegerAttr(1)) :
        Attribute(b.getStringAttr(phase == "classify" ? "ceil_B_128" : "min_B_resident"));
    SmallVector<NamedAttribute> fields{
      b.getNamedAttr("name", b.getStringAttr(name)),
      b.getNamedAttr("role", b.getStringAttr(role)),
      b.getNamedAttr("grid", b.getArrayAttr({gridX,
          b.getI64IntegerAttr(segmentClass.empty() ? 1 : (d + tile - 1) / tile),
          b.getI64IntegerAttr(parts)})),
      b.getNamedAttr("block", b.getDenseI64ArrayAttr({threads, 1, 1})),
      b.getNamedAttr("args", b.getArrayAttr(names)),
      b.getNamedAttr("shared_bytes", b.getI64IntegerAttr(0)),
      b.getNamedAttr("static_shared_bytes", b.getI64IntegerAttr(staticShared))};
    if (!segmentClass.empty()) fields.push_back(b.getNamedAttr("segment_class", b.getStringAttr(segmentClass)));
    manifest.push_back(b.getDictionaryAttr(fields));
  }
  void emitInit() {
    Value tid = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
    Value active = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, tid, index(b, loc, 3));
    when(b, loc, active, [&] {
      Value zero64 = b.create<arith::ConstantIntOp>(loc, 0, 64);
      b.create<memref::StoreOp>(loc, zero64, argument("class_counts"), ValueRange{tid});
    });
    if (debugVisits) {
      b.create<scf::ForOp>(loc, tid, segmentCount(), index(b, loc, 128), ValueRange{},
        [&](OpBuilder &nested, Location at, Value row, ValueRange) {
          Value zero32 = nested.create<arith::ConstantIntOp>(at, 0, 32);
          nested.create<memref::StoreOp>(at, zero32, argument("visits"), ValueRange{row});
          nested.create<scf::YieldOp>(at);
        });
    }
  }
  // Each warp ranks its lanes within their classes with one shuffle scan over
  // three packed 8-bit counts, then lanes 0-2 reserve each class's positions
  // with one atomic add apiece. The counters see at most three atomics per
  // warp instead of one per segment, and a warp's segments of one class land
  // in consecutive queue positions in segment order. Every lane of the block
  // reaches the shuffles; only segment loads and the final store are guarded.
  void emitClassify() {
    Value block = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
    Value tid = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
    Value segment = b.create<arith::AddIOp>(loc, tid,
        b.create<arith::MulIOp>(loc, block, index(b, loc, 128)));
    Value active = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, segment, segmentCount());
    Value selected = choose(b, loc, active, index(b, loc, 0), [&] {
      Value next = b.create<arith::AddIOp>(loc, segment, index(b, loc, 1));
      Value begin = b.create<memref::LoadOp>(loc, function.getArgument(1), ValueRange{segment});
      Value end = b.create<memref::LoadOp>(loc, function.getArgument(1), ValueRange{next});
      begin = b.create<arith::IndexCastOp>(loc, b.getIndexType(), begin);
      end = b.create<arith::IndexCastOp>(loc, b.getIndexType(), end);
      Value length = b.create<arith::SubIOp>(loc, end, begin);
      Value selected = index(b, loc, classId(classes.back()));
      auto thresholds = dispatch->getAttrOfType<DenseI64ArrayAttr>("thresholds");
      if (llvm::is_contained(classes, "cta")) {
        Value fits = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ule, length, index(b, loc, thresholds[1]));
        selected = b.create<arith::SelectOp>(loc, fits, index(b, loc, 1), selected);
      }
      if (llvm::is_contained(classes, "warp")) {
        Value fits = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ule, length, index(b, loc, thresholds[0]));
        selected = b.create<arith::SelectOp>(loc, fits, index(b, loc, 0), selected);
      }
      return selected;
    });
    auto i32 = [&](int64_t value) { return Value(b.create<arith::ConstantIntOp>(loc, value, 32)); };
    Value width = i32(32);
    Value classShift = b.create<arith::MulIOp>(loc,
        b.create<arith::IndexCastOp>(loc, b.getI32Type(), selected), i32(8));
    Value packed = b.create<arith::SelectOp>(loc, active,
        b.create<arith::ShLIOp>(loc, i32(1), classShift), i32(0));
    Value lane = b.create<arith::IndexCastOp>(loc, b.getI32Type(),
        b.create<arith::RemUIOp>(loc, tid, index(b, loc, 32)));
    // Inclusive scan; a lane below the distance keeps its value.
    Value inclusive = packed;
    for (int64_t distance = 1; distance < 32; distance *= 2) {
      Value lower = b.create<gpu::ShuffleOp>(loc, inclusive, i32(distance), width,
                                             gpu::ShuffleMode::UP).getShuffleResult();
      Value has = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::uge, lane, i32(distance));
      inclusive = b.create<arith::SelectOp>(loc, has,
          b.create<arith::AddIOp>(loc, inclusive, lower), inclusive);
    }
    Value total = b.create<gpu::ShuffleOp>(loc, inclusive, i32(31), width,
                                           gpu::ShuffleMode::IDX).getShuffleResult();
    Value rank = b.create<arith::AndIOp>(loc, b.create<arith::ShRUIOp>(loc,
        b.create<arith::SubIOp>(loc, inclusive, packed), classShift), i32(0xff));
    // Lane c < 3 reserves class c; lanes from 3 read the zero top byte.
    Value laneShift = b.create<arith::MulIOp>(loc, b.create<arith::MinUIOp>(loc, lane, i32(3)), i32(8));
    Value count = b.create<arith::AndIOp>(loc, b.create<arith::ShRUIOp>(loc, total, laneShift), i32(0xff));
    Value reserves = b.create<arith::AndIOp>(loc,
        b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, lane, i32(3)),
        b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, count, i32(0)));
    Value zero64 = b.create<arith::ConstantIntOp>(loc, 0, 64);
    Value base = choose(b, loc, reserves, zero64, [&] {
      Value queue = b.create<arith::IndexCastOp>(loc, b.getIndexType(), lane);
      return Value(b.create<memref::AtomicRMWOp>(loc, arith::AtomicRMWKind::addi,
          b.create<arith::ExtUIOp>(loc, b.getI64Type(), count), argument("class_counts"), ValueRange{queue}));
    });
    // Broadcast the reserving lane's i64 base as two 32-bit halves.
    Value from = b.create<arith::IndexCastOp>(loc, b.getI32Type(), selected);
    Value low = b.create<gpu::ShuffleOp>(loc, b.create<arith::TruncIOp>(loc, b.getI32Type(), base),
                                         from, width, gpu::ShuffleMode::IDX).getShuffleResult();
    Value high = b.create<gpu::ShuffleOp>(loc, b.create<arith::TruncIOp>(loc, b.getI32Type(),
        b.create<arith::ShRUIOp>(loc, base, b.create<arith::ConstantIntOp>(loc, 32, 64))),
        from, width, gpu::ShuffleMode::IDX).getShuffleResult();
    Value classBase = b.create<arith::OrIOp>(loc,
        b.create<arith::ShLIOp>(loc, b.create<arith::ExtUIOp>(loc, b.getI64Type(), high),
                                b.create<arith::ConstantIntOp>(loc, 32, 64)),
        b.create<arith::ExtUIOp>(loc, b.getI64Type(), low));
    Value position = b.create<arith::AddIOp>(loc, classBase, b.create<arith::ExtUIOp>(loc, b.getI64Type(), rank));
    when(b, loc, active, [&] {
      Value id64 = b.create<arith::IndexCastOp>(loc, b.getI64Type(), segment);
      b.create<memref::StoreOp>(loc, id64, argument("segment_ids"),
          ValueRange{selected, b.create<arith::IndexCastOp>(loc, b.getIndexType(), position)});
    });
  }
  void emitPersistent(StringRef segmentClass, StringRef phase, int64_t threads,
                      int64_t parts, Value shared) {
    Value queue = index(b, loc, classId(segmentClass));
    Value count = b.create<memref::LoadOp>(loc, argument("class_counts"), ValueRange{queue});
    count = b.create<arith::IndexCastOp>(loc, b.getIndexType(), count);
    Value first = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
    Value step = b.create<gpu::GridDimOp>(loc, gpu::Dimension::x);
    b.create<scf::ForOp>(loc, first, count, step, ValueRange{},
      [&](OpBuilder &nested, Location at, Value position, ValueRange) {
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPoint(nested.getInsertionBlock(), nested.getInsertionPoint());
        Value segment = b.create<memref::LoadOp>(loc, argument("segment_ids"), ValueRange{queue, position});
        segment = b.create<arith::IndexCastOp>(loc, b.getIndexType(), segment);
        Value part = parts == 8 ? Value(b.create<gpu::BlockIdOp>(loc, gpu::Dimension::z)) : Value();
        CTAEmitter emitter(b, map, threads, tile, function.getArguments().take_front(3), shared,
                           segment, argument("max_stats"), argument("sum_stats"),
                           argument("partials"), part, segmentClass == "warp" && enabled(map, "shuffle"),
                           map.getMask() ? Value(function.getArgument(3)) : Value(), argument("logits"),
                           argument("partial_sums"));
        emitter.emit(phase);
        bool finalOutput = phase == "sum" || phase == "max" || phase == "online" ||
                           phase == "normalize" || phase == "combine_sum" || phase == "combine_max";
        if (debugVisits && finalOutput) {
          Value tid = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
          Value featureBlock = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::y);
          Value zero = index(b, loc, 0);
          Value leader = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, tid, zero);
          Value firstTile = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, featureBlock, zero);
          Value countVisit = b.create<arith::AndIOp>(loc, leader, firstTile);
          if (part) countVisit = b.create<arith::AndIOp>(loc, countVisit,
              b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, part, zero));
          when(b, loc, countVisit, [&] {
            Value one32 = b.create<arith::ConstantIntOp>(loc, 1, 32);
            b.create<memref::AtomicRMWOp>(loc, arith::AtomicRMWKind::addi,
                one32, argument("visits"), ValueRange{segment});
          });
        }
        // All old consumers must finish before any lane begins the next queue
        // item's shared-memory stores. Queue position and count are CTA-uniform.
        b.create<gpu::BarrierOp>(loc);
        b.create<scf::YieldOp>(loc);
      });
  }
  OpBuilder &b;
  gpu::GPUModuleOp gpuModule;
  DispatchOp dispatch;
  MapSegmentsOp map;
  Location loc;
  unsigned id;
  int64_t d, tile;
  StringRef operation;
  bool debugVisits, hasSplit;
  SmallVector<StringRef> classes;
  SmallVector<bool> online;  // per class, aligned with classes
  SmallVector<Type> types;
  SmallVector<Attribute> names, workspace;
  llvm::StringMap<unsigned> argumentIndices;
  gpu::GPUFuncOp function;
  SmallVectorImpl<Attribute> &manifest;
};

// Validation is a separate launch sequence. These kernels have no out argument,
// and values contributes only its descriptor's N dimension. The runtime invokes
// them only in checked mode, before allocating/launching computation workspace.
DictionaryAttr emitOffsetsValidation(OpBuilder &b, gpu::GPUModuleOp gpuModule,
                                     MapSegmentsOp map, unsigned id) {
  Location loc = map.getLoc();
  auto statusType = MemRefType::get({7}, b.getI64Type());
  SmallVector<Type> types{map.getValues().getType(), map.getOffsets().getType(), statusType};
  auto names = b.getArrayAttr({b.getStringAttr("values"), b.getStringAttr("offsets"),
                             b.getStringAttr("validation_status")});
  SmallVector<Attribute> kernels;
  for (StringRef role : {"init", "check"}) {
    b.setInsertionPointToStart(gpuModule.getBody());
    std::string name = "ragged_" + std::to_string(id) + "_offsets_" + role.str();
    auto function = b.create<gpu::GPUFuncOp>(loc, name, b.getFunctionType(types, {}));
    function.setKernelAttr(b.getUnitAttr());
    b.setInsertionPointToStart(&function.getBody().front());
    Value values = function.getArgument(0), offsets = function.getArgument(1);
    Value status = function.getArgument(2);
    Value tid = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
    Value zero = index(b, loc, 0), one = index(b, loc, 1);
    if (role == "init") {
      Value active = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult,
                                           tid, index(b, loc, 7));
      when(b, loc, active, [&] {
        Value sentinel = b.create<arith::ConstantIntOp>(loc, std::numeric_limits<int64_t>::max(), 64);
        b.create<memref::StoreOp>(loc, sentinel, status, ValueRange{tid});
      });
    } else {
      Value block = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
      Value position = b.create<arith::AddIOp>(loc, tid,
          b.create<arith::MulIOp>(loc, block, index(b, loc, 128)));
      Value offsetCount = b.create<memref::DimOp>(loc, offsets, 0);
      Value active = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, position, offsetCount);
      // This guard also permits B=0: offsets has one entry and one lane checks it.
      when(b, loc, active, [&] {
        Value n = b.create<memref::DimOp>(loc, values, 0);
        n = b.create<arith::IndexCastOp>(loc, b.getI64Type(), n);
        Value zero64 = b.create<arith::ConstantIntOp>(loc, 0, 64);
        Value index64 = b.create<arith::IndexCastOp>(loc, b.getI64Type(), position);
        auto loadOffset = [&](Value at) -> Value {
          Value value = b.create<memref::LoadOp>(loc, offsets, ValueRange{at});
          if (value.getType().isInteger(32))
            value = b.create<arith::ExtSIOp>(loc, b.getI64Type(), value);
          return value;
        };
        auto report = [&](unsigned slot, Value condition) {
          when(b, loc, condition, [&] {
            b.create<memref::AtomicRMWOp>(loc, arith::AtomicRMWKind::mins,
                index64, status, ValueRange{index(b, loc, slot)});
          });
        };
        auto compare = [&](arith::CmpIPredicate predicate, Value lhs, Value rhs) -> Value {
          return b.create<arith::CmpIOp>(loc, predicate, lhs, rhs);
        };
        auto both = [&](Value lhs, Value rhs) -> Value {
          return b.create<arith::AndIOp>(loc, lhs, rhs);
        };
        Value current = loadOffset(position);
        Value first = compare(arith::CmpIPredicate::eq, position, zero);
        Value lastIndex = b.create<arith::SubIOp>(loc, offsetCount, one);
        Value last = compare(arith::CmpIPredicate::eq, position, lastIndex);
        report(0, both(first, compare(arith::CmpIPredicate::ne, current, zero64)));
        report(1, both(last, compare(arith::CmpIPredicate::ne, current, n)));
        Value negative = compare(arith::CmpIPredicate::slt, current, zero64);
        Value above = compare(arith::CmpIPredicate::sgt, current, n);
        report(2, negative);
        report(3, above);
        Value hasNext = compare(arith::CmpIPredicate::ult, position, lastIndex);
        when(b, loc, hasNext, [&] {
          Value nextPosition = b.create<arith::AddIOp>(loc, position, one);
          Value next = loadOffset(nextPosition);
          report(4, compare(arith::CmpIPredicate::slt, next, current));
          auto minLength = map->getAttrOfType<IntegerAttr>("min_len");
          auto maxLength = map->getAttrOfType<IntegerAttr>("max_len");
          if (!minLength && !maxLength) return;
          Value currentValid = both(compare(arith::CmpIPredicate::sge, current, zero64),
                                    compare(arith::CmpIPredicate::sle, current, n));
          Value nextValid = both(compare(arith::CmpIPredicate::sge, next, zero64),
                                 compare(arith::CmpIPredicate::sle, next, n));
          Value monotone = compare(arith::CmpIPredicate::sge, next, current);
          // Only valid nonnegative endpoints reach subtraction, eliminating
          // signed overflow even for inputs containing INT64_MIN/INT64_MAX.
          when(b, loc, both(both(currentValid, nextValid), monotone), [&] {
            Value length = b.create<arith::SubIOp>(loc, next, current);
            if (minLength) {
              Value bound = b.create<arith::ConstantIntOp>(loc, minLength.getInt(), 64);
              report(5, compare(arith::CmpIPredicate::slt, length, bound));
            }
            if (maxLength) {
              Value bound = b.create<arith::ConstantIntOp>(loc, maxLength.getInt(), 64);
              report(6, compare(arith::CmpIPredicate::sgt, length, bound));
            }
          });
        });
      });
    }
    b.create<gpu::ReturnOp>(loc);
    Attribute gridX = role == "init" ? Attribute(b.getI64IntegerAttr(1)) :
                                      Attribute(b.getStringAttr("ceil_B1_128"));
    kernels.push_back(b.getDictionaryAttr({
        b.getNamedAttr("name", b.getStringAttr(name)),
        b.getNamedAttr("role", b.getStringAttr(role)),
        b.getNamedAttr("grid", b.getArrayAttr({gridX, b.getI64IntegerAttr(1), b.getI64IntegerAttr(1)})),
        b.getNamedAttr("block", b.getDenseI64ArrayAttr({128, 1, 1})),
        b.getNamedAttr("args", names),
        b.getNamedAttr("shared_bytes", b.getI64IntegerAttr(0)),
        b.getNamedAttr("static_shared_bytes", b.getI64IntegerAttr(0))}));
  }
  auto workspace = b.getDictionaryAttr({
      b.getNamedAttr("name", b.getStringAttr("validation_status")),
      b.getNamedAttr("dtype", b.getStringAttr("i64")),
      b.getNamedAttr("shape", b.getArrayAttr({b.getI64IntegerAttr(7)}))});
  SmallVector<Attribute> conditions;
  for (StringRef name : {"first_nonzero", "final_mismatch", "negative", "above_n",
                         "nonmonotone", "below_min_len", "above_max_len"})
    conditions.push_back(b.getStringAttr(name));
  return b.getDictionaryAttr({
      b.getNamedAttr("abi", b.getStringAttr("offsets-check-v1")),
      b.getNamedAttr("workspace", b.getArrayAttr({workspace})),
      b.getNamedAttr("conditions", b.getArrayAttr(conditions)),
      b.getNamedAttr("sentinel", b.getI64IntegerAttr(std::numeric_limits<int64_t>::max())),
      b.getNamedAttr("min_len", map->hasAttr("min_len") ? map->getAttr("min_len") : b.getUnitAttr()),
      b.getNamedAttr("max_len", map->hasAttr("max_len") ? map->getAttr("max_len") : b.getUnitAttr()),
      b.getNamedAttr("kernels", b.getArrayAttr(kernels))});
}

struct ToGPUPass : PassWrapper<ToGPUPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ToGPUPass)
  StringRef getArgument() const final { return "ragged-to-gpu"; }
  StringRef getDescription() const final { return "Generate sm_89 GPU kernels with uniform CTA reductions"; }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect, gpu::GPUDialect,
                    math::MathDialect, memref::MemRefDialect, scf::SCFDialect>();
  }
  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<DispatchOp> dispatches;
    module.walk([&](DispatchOp op) { dispatches.push_back(op); });
    if (dispatches.empty()) {
      module.emitError("ragged-to-gpu requires planned ragged.dispatch operations");
      return signalPassFailure();
    }
    bool classified = false;
    for (DispatchOp dispatch : dispatches) {
      auto classes = dispatch->getAttrOfType<ArrayAttr>("classes");
      auto map = *dispatch.getBody().front().getOps<MapSegmentsOp>().begin();
      auto debug = map->getAttrOfType<BoolAttr>("debug_visits");
      classified |= classes.size() > 1 || cast<StringAttr>(classes[0]).getValue() == "split" || (debug && debug.getValue());
    }
    if (classified && dispatches.size() != 1) {
      module.emitError("classified workspace ABI requires exactly one dispatch per module");
      return signalPassFailure();
    }
    if (dispatches.size() != 1) {
      module.emitError("offsets validation ABI requires exactly one dispatch per module");
      return signalPassFailure();
    }
    OpBuilder b(module.getContext());
    b.setInsertionPointToEnd(module.getBody());
    auto gpuModule = b.create<gpu::GPUModuleOp>(module.getLoc(), "ragged_kernels");
    module->setAttr(gpu::GPUDialect::getContainerModuleAttrName(), b.getUnitAttr());
    SmallVector<Attribute> manifest;
    unsigned dispatchId = 0;
    for (DispatchOp dispatch : dispatches) {
      auto map = *dispatch.getBody().front().getOps<MapSegmentsOp>().begin();
      module->setAttr("ragged.mask", b.getBoolAttr(bool(map.getMask())));
      auto budget = [&](StringRef name, int64_t fallback) -> Attribute {
        if (auto attr = map->getAttr(name)) return attr;
        return b.getI64IntegerAttr(fallback);
      };
      module->setAttr("ragged.resources", b.getDictionaryAttr({
          b.getNamedAttr("status", b.getStringAttr("unmeasured")),
          b.getNamedAttr("source", b.getUnitAttr()),
          b.getNamedAttr("requested", b.getDictionaryAttr({
              b.getNamedAttr("regs_budget", budget("regs_budget", 65536)),
              b.getNamedAttr("smem_per_cta", budget("smem_per_cta", 49152))})),
          b.getNamedAttr("kernels", b.getArrayAttr({}))}));
      auto classes = dispatch->getAttrOfType<ArrayAttr>("classes");
      bool hasWarp = llvm::any_of(classes, [](Attribute item) { return cast<StringAttr>(item).getValue() == "warp"; });
      bool hasSplit = llvm::any_of(classes, [](Attribute item) { return cast<StringAttr>(item).getValue() == "split"; });
      bool materialized = materializesPointwise(map);
      SmallVector<NamedAttribute> requested;
      for (StringRef name : {"classification", "online_fusion", "warp_mapping", "split_mapping", "shuffle", "pointwise_fusion",
                              "row_pipelining"})
        requested.push_back(b.getNamedAttr(name, b.getBoolAttr(enabled(map, name))));
      requested.push_back(b.getNamedAttr("feature_tile", map->hasAttr("feature_tile") ? map->getAttr("feature_tile") : b.getUnitAttr()));
      auto effective = b.getDictionaryAttr({
          b.getNamedAttr("classification", b.getBoolAttr(classified)),
          b.getNamedAttr("online_fusion", b.getBoolAttr(str(map, "fusion") == "online" || str(map, "fusion") == "mixed")),
          b.getNamedAttr("warp_mapping", b.getBoolAttr(hasWarp)),
          b.getNamedAttr("split_mapping", b.getBoolAttr(hasSplit)),
          b.getNamedAttr("shuffle", b.getBoolAttr(hasWarp && enabled(map, "shuffle"))),
          b.getNamedAttr("pointwise_fusion", b.getBoolAttr(hasPointwise(map) && !materialized)),
          b.getNamedAttr("row_pipelining", b.getBoolAttr(enabled(map, "row_pipelining")))});
      module->setAttr("ragged.optimizations", b.getDictionaryAttr({
          b.getNamedAttr("requested", b.getDictionaryAttr(requested)), b.getNamedAttr("effective", effective)}));
      module->setAttr("ragged.d_tile", dispatch->getAttr("d_tile"));
      module->setAttr("ragged.pointwise_materialized", b.getBoolAttr(materialized));
      if (classified) {
        ClassifiedEmitter emitter(b, gpuModule, dispatch, dispatchId, manifest);
        module->setAttr("ragged.workspace", emitter.emit());
        module->setAttr("ragged.dispatch_abi", b.getStringAttr("classified-v2"));
        module->setAttr("ragged.split_parts", b.getI64IntegerAttr(8));
        auto debug = map->getAttrOfType<BoolAttr>("debug_visits");
        module->setAttr("ragged.debug_visits", b.getBoolAttr(debug && debug.getValue()));
        module->setAttr("ragged.validation", emitOffsetsValidation(b, gpuModule, map, dispatchId));
        ++dispatchId;
        dispatch.erase();
        continue;
      }
      int64_t tile = cast<IntegerAttr>(dispatch->getAttr("d_tile")).getInt();
      int64_t threads = cast<IntegerAttr>(dispatch->getAttr("threads")).getInt();
      int64_t d = cast<MemRefType>(map.getValues().getType()).getDimSize(1);
      StringRef operation = str(map, "operation");
      bool softmax = operation == "softmax";
      bool online = softmax && str(map, "fusion") == "online";
      bool separateStats = softmax && !online;
      SmallVector<StringRef> phases = online ? SmallVector<StringRef>{"online"} : softmax
          ? SmallVector<StringRef>{"max_stats", "sum_stats", "normalize"}
          : SmallVector<StringRef>{operation};
      SmallVector<Type> types(map->getOperandTypes());
      SmallVector<Attribute> names{b.getStringAttr("values"), b.getStringAttr("offsets"), b.getStringAttr("out")};
      if (map.getMask()) names.push_back(b.getStringAttr("mask"));
      SmallVector<Attribute> workspace;
      auto addWorkspace = [&](StringRef name, StringRef dimension) {
        types.push_back(MemRefType::get({ShapedType::kDynamic, d}, b.getF32Type()));
        names.push_back(b.getStringAttr(name));
        workspace.push_back(b.getDictionaryAttr({
            b.getNamedAttr("name", b.getStringAttr(name)), b.getNamedAttr("dtype", b.getStringAttr("f32")),
            b.getNamedAttr("shape", b.getArrayAttr({b.getStringAttr(dimension), b.getI64IntegerAttr(d)}))}));
      };
      if (separateStats) { addWorkspace("max_stats", "B"); addWorkspace("sum_stats", "B"); }
      unsigned logitsIndex = types.size();
      if (materialized) addWorkspace("logits", "N");
      module->setAttr("ragged.workspace", b.getArrayAttr(workspace));
      if (materialized)
        manifest.push_back(emitMaterialize(b, gpuModule, map, dispatchId, types, b.getArrayAttr(names), logitsIndex));
      bool useShuffle = hasWarp && enabled(map, "shuffle");
      for (StringRef phase : phases) {
        b.setInsertionPointToStart(gpuModule.getBody());
        std::string name = "ragged_" + std::to_string(dispatchId) + "_" + phase.str();
        auto function = b.create<gpu::GPUFuncOp>(map.getLoc(), name, b.getFunctionType(types, {}));
        function.setKernelAttr(b.getUnitAttr());
        auto space = gpu::AddressSpaceAttr::get(b.getContext(), gpu::AddressSpace::Workgroup);
        int64_t sharedElements = phase == "normalize" || useShuffle ? 0 : online ? threads * 2 : threads;
        Value shared;
        if (sharedElements) {
          auto sharedType = MemRefType::get({sharedElements}, b.getF32Type(), MemRefLayoutAttrInterface{}, space);
          shared = function.addWorkgroupAttribution(sharedType, map.getLoc());
        }
        b.setInsertionPointToStart(&function.getBody().front());
        unsigned sourceArguments = map->getNumOperands();
        Value maxStats = separateStats ? Value(function.getArgument(sourceArguments)) : Value();
        Value sumStats = separateStats ? Value(function.getArgument(sourceArguments + 1)) : Value();
        Value mask = map.getMask() ? Value(function.getArgument(3)) : Value();
        Value logits = materialized ? Value(function.getArgument(logitsIndex)) : Value();
        CTAEmitter emitter(b, map, threads, tile, function.getArguments().take_front(3), shared,
                           {}, maxStats, sumStats, {}, {}, useShuffle, mask, logits);
        emitter.emit(phase);
        b.create<gpu::ReturnOp>(map.getLoc());
        SmallVector<NamedAttribute> fields{
          b.getNamedAttr("name", b.getStringAttr(name)),
          b.getNamedAttr("grid", b.getArrayAttr({b.getStringAttr("B"), b.getI64IntegerAttr((d + tile - 1) / tile), b.getI64IntegerAttr(1)})),
          b.getNamedAttr("block", b.getDenseI64ArrayAttr({threads, 1, 1})),
          b.getNamedAttr("args", b.getArrayAttr(names)),
          b.getNamedAttr("shared_bytes", b.getI64IntegerAttr(0)),
          b.getNamedAttr("static_shared_bytes", b.getI64IntegerAttr(sharedElements * 4))};
        manifest.push_back(b.getDictionaryAttr(fields));
      }
      module->setAttr("ragged.validation", emitOffsetsValidation(b, gpuModule, map, dispatchId));
      ++dispatchId;
      dispatch.erase();
    }
    module->setAttr("ragged.kernels", b.getArrayAttr(manifest));
    module->setAttr("ragged.abi", b.getStringAttr("mlir-memref64-v1"));
    module->setAttr("ragged.backend", b.getStringAttr(classified ? "classified-persistent" : "cta-or-warp"));
  }
};
}
std::unique_ptr<Pass> createToGPUPass() { return std::make_unique<ToGPUPass>(); }
}
