"""Dependency-free, immutable source specification and execution evidence."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import re
from typing import Any

from .errors import ValidationError
from .optimizations import BOOL_OPTIONS, Optimizations

DIALECT_VERSION = "1"


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _nonnegative_int(name: str, value: int | None) -> None:
    if value is not None and (type(value) is not int or value < 0 or value > 2**63 - 1):
        raise ValidationError(f"{name} must be an integer in [0, 2^63-1]")


@dataclass(frozen=True)
class Hints:
    max_len: int | None = None
    min_len: int | None = None
    offsets_checked: bool = False

    def __post_init__(self) -> None:
        _nonnegative_int("max_len", self.max_len)
        _nonnegative_int("min_len", self.min_len)
        if type(self.offsets_checked) is not bool:
            raise ValidationError("offsets_checked must be bool")
        if self.max_len is not None and self.min_len is not None and self.min_len > self.max_len:
            raise ValidationError("min_len must not exceed max_len")


@dataclass(frozen=True)
class ProgramSpec:
    op: str
    dtype: str = "f32"
    feature_dim: int = 128
    offsets_dtype: str = "i64"
    hints: Hints = field(default_factory=Hints)
    strategy: str = "auto"
    alias: str = "none"
    chip: str = "sm_89"
    force_fusion: bool = False
    scale: float = 1.0
    debug_visits: bool = False
    mask: bool = False
    optimizations: Optimizations = field(default_factory=Optimizations)
    smem_per_cta: int = 49152
    regs_budget: int = 65536

    def __post_init__(self) -> None:
        if self.op not in ("sum", "max", "softmax"):
            raise ValidationError("op must be sum, max or softmax")
        if self.dtype not in ("f32", "bf16"):
            raise ValidationError("dtype must be f32 or bf16; accumulation is always f32")
        if type(self.feature_dim) is not int or not 1 <= self.feature_dim <= 1024:
            raise ValidationError("feature_dim must be a static integer in [1, 1024]")
        if self.offsets_dtype not in ("i32", "i64"):
            raise ValidationError("offsets_dtype must be i32 or i64")
        if not isinstance(self.hints, Hints):
            raise ValidationError("hints must be Hints")
        if self.strategy not in ("auto", "naive"):
            raise ValidationError("strategy must be auto or naive")
        if self.alias not in ("none", "inplace"):
            raise ValidationError("alias must be none or inplace")
        if self.alias == "inplace" and self.op != "softmax":
            raise ValidationError("in-place output is only defined for softmax")
        if self.chip != "sm_89":
            raise ValidationError("only sm_89 is supported")
        if type(self.force_fusion) is not bool:
            raise ValidationError("force_fusion must be bool")
        if type(self.debug_visits) is not bool:
            raise ValidationError("debug_visits must be bool")
        if type(self.mask) is not bool:
            raise ValidationError("mask must be bool")
        if self.mask and self.op != "softmax":
            raise ValidationError("additive mask is only defined for softmax")
        if not isinstance(self.optimizations, Optimizations):
            raise ValidationError("optimizations must be Optimizations")
        if self.debug_visits and not self.optimizations.classification:
            raise ValidationError("debug_visits requires classification")
        if self.force_fusion and not self.optimizations.online_fusion:
            raise ValidationError("force_fusion conflicts with online_fusion disabled")
        if self.force_fusion and self.op != "softmax":
            raise ValidationError("force_fusion is only defined for softmax")
        if type(self.scale) not in (int, float) or abs(self.scale) > 3.402823466e38 or not math.isfinite(self.scale):
            raise ValidationError("scale must be a finite f32 value")
        for name in ("smem_per_cta", "regs_budget"):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= 2**63 - 1:
                raise ValidationError(f"{name} must be a positive signed i64 integer")

    @property
    def signature(self) -> str:
        """Hints and mapping policy specialize this semantic tensor signature."""
        return f"{self.op}:{self.dtype}:D{self.feature_dim}:{self.offsets_dtype}:{self.chip}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProgramSpec:
        data = dict(data)
        data["hints"] = Hints(**data.get("hints", {}))
        data["optimizations"] = Optimizations(**data.get("optimizations", {}))
        return cls(**data)

    def to_mlir(self, function_name: str = "main") -> str:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", function_name):
            raise ValidationError("function_name must be an MLIR identifier")
        tensor = f"memref<?x{self.feature_dim}x{self.dtype}>"
        offsets = f"memref<?x{self.offsets_dtype}>"
        names, types = ["values", "offsets", "out"], [tensor, offsets, tensor]
        if self.mask:
            names.append("mask")
            types.append(f"memref<?x{self.feature_dim}xf32>")
        arguments = ", ".join(f"%{name}: {type_}" for name, type_ in zip(names, types))
        operands = ", ".join(f"%{name}" for name in names)
        attrs: list[str] = []
        op_name = "segment_softmax" if self.op == "softmax" else "segment_reduce"
        if self.op != "softmax":
            attrs.append(f'kind = "{self.op}"')
        for key in ("max_len", "min_len"):
            value = getattr(self.hints, key)
            if value is not None:
                attrs.append(f"{key} = {value} : i64")
        attrs.extend([
            f"offsets_checked = {str(self.hints.offsets_checked).lower()}",
            f'alias = "{self.alias}"',
            f'strategy = "{self.strategy}"',
            f"smem_per_cta = {self.smem_per_cta} : i64",
            f"regs_budget = {self.regs_budget} : i64",
        ])
        if self.force_fusion:
            attrs.append("force_fusion = true")
        if self.debug_visits:
            attrs.append("debug_visits = true")
        if self.scale != 1.0:
            attrs.append(f"scale = {float(self.scale):.9e} : f32")
        for name in BOOL_OPTIONS:
            if not getattr(self.optimizations, name):
                attrs.append(f"{name} = false")
        if self.optimizations.feature_tile is not None:
            attrs.append(f"feature_tile = {self.optimizations.feature_tile} : i64")
        return (
            "module {\n"
            f"  func.func @{function_name}({arguments}) {{\n"
            f'    "ragged.{op_name}"({operands}) '
            "{" + ", ".join(attrs) + "}"
            f" : ({', '.join(types)}) -> ()\n"
            "    return\n  }\n}\n"
        )

    @property
    def ir_hash(self) -> str:
        # Generated text has stable names, ordering, whitespace and a final newline.
        return hashlib.sha256(self.to_mlir().encode()).hexdigest()


@dataclass(frozen=True)
class ExecutionReport:
    classes: tuple[str, ...]
    thresholds: tuple[int, ...]
    grids: tuple[tuple[int, int, int], ...]
    workspace_bytes: int
    cache_key: str
    fallback_reason: str | None = None
    mode: str = "validated"
    kernel_names: tuple[str, ...] = ()
    host_enqueue_ms: float | None = None
    # Device timings need CUDA events and are not fabricated from host timings.
    device_elapsed_ms: float | None = None
    cache_hit: bool = False
    validation_workspace_bytes: int = 0
    # Host wall time through the checked path's stream synchronization, not a
    # CUDA-event measurement. It can include earlier work on that stream.
    validation_host_ms: float | None = None
    optimizations: dict[str, Any] | None = None
    d_tile: int | None = None
    d_vector: int = 1
    pointwise_materialized: bool = False
    # Actual ptxas evidence for the compiled artifact, including entries that
    # may not launch in this invocation. None means legacy/unmeasured metadata.
    resources: dict[str, Any] | None = None
    # Streams the worker kernels ran on: the caller's plus one per additional
    # concurrently launched segment class.
    streams: int = 1
    # The schedule that ran ("classified" or "naive" for an adaptive program)
    # and the statistics and predictions that chose it (ragged.selection).
    selected: str | None = None
    selection: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=2, allow_nan=False)
