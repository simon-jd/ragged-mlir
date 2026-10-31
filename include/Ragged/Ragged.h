#ifndef RAGGED_RAGGED_H
#define RAGGED_RAGGED_H
#include "mlir/IR/Dialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "Ragged/RaggedDialect.h.inc"
#define GET_OP_CLASSES
#include "Ragged/RaggedOps.h.inc"
#endif
