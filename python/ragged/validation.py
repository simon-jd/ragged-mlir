"""Kernel manifest validation, and the checked mode's offsets metadata and status decoding.

A program is launched only from a manifest that validate_manifest accepts and
that validate_program binds to its source specialization and cache key.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from .errors import CacheMismatch, ValidationError
from .layout import CLASSIFIED_GRID_SYMBOLS, GRID_LIMITS, I64_MAX, SHAPE_SYMBOLS, WORKSPACE_DTYPES, valid_dimension
from .model import ProgramSpec
from .optimizations import (class_fusion, computation_metadata, validate_optimization_metadata,
                            validate_optimization_spec)
from .resources import validate_resources_metadata, validate_resources_spec

if TYPE_CHECKING:
    from .cache import Artifact

CONDITIONS = ("first_nonzero", "final_mismatch", "negative", "above_n",
              "nonmonotone", "below_min_len", "above_max_len")
STATUS_WORKSPACE = [{"name": "validation_status", "dtype": "i64", "shape": [7]}]
STATUS_ARGUMENTS = ["values", "offsets", "validation_status"]
ABI = "mlir-memref64-v1"


def validate_validation_metadata(validation: Any, *, main_names=()) -> None:
    fields = {"abi", "workspace", "conditions", "sentinel", "min_len", "max_len", "kernels"}
    if not isinstance(validation, dict) or set(validation) != fields:
        raise CacheMismatch("offsets validation metadata fields differ from offsets-check-v1")
    if validation["abi"] != "offsets-check-v1":
        raise CacheMismatch("unsupported offsets validation ABI")
    if (validation["workspace"] != STATUS_WORKSPACE or validation["conditions"] != list(CONDITIONS)
            or type(validation["sentinel"]) is not int or validation["sentinel"] != I64_MAX):
        raise CacheMismatch("offsets validation status layout, conditions or sentinel differs")
    if type(validation["workspace"][0]["shape"][0]) is not int:
        raise CacheMismatch("offsets validation status shape must use exact integers")
    for field in ("min_len", "max_len"):
        value = validation[field]
        if value is not None and (type(value) is not int or not 0 <= value <= I64_MAX):
            raise CacheMismatch("offsets validation bounds must be null or nonnegative i64")
    if (validation["min_len"] is not None and validation["max_len"] is not None
            and validation["min_len"] > validation["max_len"]):
        raise CacheMismatch("offsets validation minimum exceeds maximum")
    kernels = validation["kernels"]
    if not isinstance(kernels, list) or len(kernels) != 2:
        raise CacheMismatch("offsets validation requires init and check kernels")
    names = set(main_names)
    for kernel, role, grid in zip(kernels, ("init", "check"), ([1, 1, 1], ["ceil_B1_128", 1, 1])):
        if not isinstance(kernel, dict) or set(kernel) - {
                "name", "role", "grid", "block", "args", "shared_bytes", "static_shared_bytes"}:
            raise CacheMismatch("invalid offsets validation kernel fields")
        name = kernel.get("name")
        if not isinstance(name, str) or not name or "\0" in name or name in names:
            raise CacheMismatch("invalid or duplicate offsets validation kernel symbol")
        names.add(name)
        if (kernel.get("role") != role or kernel.get("grid") != grid or kernel.get("block") != [128, 1, 1]
                or kernel.get("args") != STATUS_ARGUMENTS):
            raise CacheMismatch("offsets validation kernel order, launch geometry or arguments differs")
        # JSON equality alone treats True == 1. Require exact integral geometry.
        for field in ("grid", "block"):
            for dimension in kernel[field]:
                if not isinstance(dimension, str) and type(dimension) is not int:
                    raise CacheMismatch("offsets validation grid/block dimensions must be exact integers")
        for field in ("shared_bytes", "static_shared_bytes"):
            if type(kernel.get(field, 0)) is not int or kernel.get(field, 0) != 0:
                raise CacheMismatch("offsets validation kernels use no shared memory")


def validate_validation_hints(spec: ProgramSpec, validation: dict[str, Any]) -> None:
    if any(validation[name] != getattr(spec.hints, name) for name in ("min_len", "max_len")):
        raise CacheMismatch("offsets validation bounds differ from source hints")


def raise_for_status(status: list[int], *, b: int) -> None:
    """Decode only the seven host status slots, after stream synchronization."""
    if not isinstance(status, list) or len(status) != len(CONDITIONS):
        raise ValidationError("invalid offsets validation status layout")
    errors = {}
    for slot, (condition, index) in enumerate(zip(CONDITIONS, status)):
        if type(index) is not int or not 0 <= index <= I64_MAX:
            raise ValidationError("invalid offsets validation status index")
        if index == I64_MAX:
            continue
        limit = b if slot < 4 else b - 1
        if index > limit or (slot == 0 and index != 0) or (slot == 1 and index != b):
            raise ValidationError("offsets validation status index exceeds its condition domain")
        errors[condition] = index
    if errors:
        first = next(iter(errors))
        detail = "; ".join(f"{condition} at index {index}" for condition, index in errors.items())
        raise ValidationError("offsets validation failed: " + detail,
                              condition=first, index=errors[first], errors=errors)


def validate_program(spec: ProgramSpec, artifact: Artifact) -> None:
    """Check that a cached artifact is the compiled form of spec: its key, its
    manifest, and the toolchain that built it."""
    if artifact.key.ir_hash != spec.ir_hash:
        raise CacheMismatch("compiled program source does not match cached IR hash")
    for field in ("op", "dtype", "feature_dim", "offsets_dtype", "strategy", "chip"):
        if getattr(artifact.key, field) != getattr(spec, field):
            raise CacheMismatch(f"compiled program {field} differs from artifact key")
    if not artifact.cubin.startswith(b"\x7fELF"):
        raise CacheMismatch("runtime requires an ELF cubin")
    validate_manifest(artifact.metadata)
    validate_optimization_spec(spec, artifact.metadata)
    validate_resources_spec(spec, artifact.metadata)
    if artifact.metadata.get("mask", False) != spec.mask:
        raise CacheMismatch("kernel manifest mask presence differs from the source specialization")
    if "validation" in artifact.metadata:
        validate_validation_hints(spec, artifact.metadata["validation"])
    if artifact.metadata.get("chip", spec.chip) != spec.chip:
        raise CacheMismatch("kernel manifest chip differs from the compiled target")
    if (artifact.key.compiler_sha256 == "unbound"
            or artifact.metadata.get("compiler_sha256") != artifact.key.compiler_sha256):
        raise CacheMismatch("kernel manifest is not bound to the compiler executable hash")
    workspace = computation_metadata(artifact.metadata).get("workspace", [])
    if artifact.metadata.get("dispatch_abi") == "classified-v3":
        validate_classified_spec(spec, artifact.metadata)
    elif spec.debug_visits or artifact.metadata.get("debug_visits", False):
        raise CacheMismatch("debug visits require the classified dispatch ABI")
    elif workspace:
        expected_workspace = [{"name": name, "shape": ["B", spec.feature_dim], "dtype": "f32"}
                              for name in ("max_stats", "sum_stats")]
        if spec.op != "softmax" or workspace != expected_workspace:
            raise CacheMismatch("softmax workspace must be max_stats and sum_stats with shape [B,D]")
    provenance = artifact.metadata.get("toolchain")
    if not isinstance(provenance, dict):
        raise CacheMismatch("kernel manifest lacks compiler toolchain provenance")
    for field in ("llvm_commit", "ptxas_version", "dialect_version", "chip", "features"):
        if provenance.get(field) != getattr(artifact.key, field):
            raise CacheMismatch(f"kernel manifest provenance differs from cache key: {field}")


def validate_classified_spec(spec: ProgramSpec, metadata: dict[str, Any], *, check_tiles: bool = True) -> None:
    metadata = computation_metadata(metadata)
    classes, fusion = metadata["classes"], metadata["fusion"]
    if metadata["debug_visits"] != spec.debug_visits:
        raise CacheMismatch("debug visits differ from the source specialization")
    if spec.op == "softmax":
        if fusion == "not_applicable":
            raise CacheMismatch("softmax requires a statistics schedule")
    elif fusion != "not_applicable":
        raise CacheMismatch("reductions must declare fusion not_applicable")
    schedule = class_fusion(metadata)
    if metadata.get("alias_safe", False) != (spec.op == "softmax" and fusion == "three_pass"):
        raise CacheMismatch("alias safety differs from the classified kernel schedule")
    workspace = [{"name": "class_counts", "shape": [3], "dtype": "i64"},
                 {"name": "segment_ids", "shape": [3, "B"], "dtype": "i64"}]
    split_online = schedule is not None and "split" in classes and schedule[-1] == "online"
    if "split" in classes:
        workspace.append({"name": "partials", "shape": ["B_split", 8 * spec.feature_dim], "dtype": "f32"})
    if split_online:
        workspace.append({"name": "partial_sums", "shape": ["B_split", 8 * spec.feature_dim], "dtype": "f32"})
    if schedule is not None and ("three_pass" in schedule or "split" in classes):
        workspace.extend({"name": name, "shape": ["B", spec.feature_dim], "dtype": "f32"}
                         for name in ("max_stats", "sum_stats"))
    if spec.debug_visits:
        workspace.append({"name": "visits", "shape": ["B"], "dtype": "i32"})
    if metadata["workspace"] != workspace:
        raise CacheMismatch("classified workspace names, order, dtypes or shapes differ from the source specialization")
    expected_roles = ["init", "classify"]
    for index, segment_class in enumerate(classes):
        if segment_class == "split":
            if split_online:
                expected_roles.extend(("split_partial_online", "split_combine_online", "split_normalize"))
            elif spec.op == "softmax":
                expected_roles.extend(("split_partial_max", "split_combine_max_stats", "split_partial_exp_sum",
                                       "split_combine_sum_stats", "split_normalize"))
            else:
                expected_roles.extend((f"split_partial_{spec.op}", f"split_combine_{spec.op}"))
        elif spec.op == "softmax":
            if schedule[index] == "online":
                expected_roles.append(f"{segment_class}_online")
            else:
                expected_roles.extend(f"{segment_class}_{phase}" for phase in ("max_stats", "sum_stats", "normalize"))
        else:
            expected_roles.append(f"{segment_class}_{spec.op}")
    if [kernel["role"] for kernel in metadata["kernels"]] != expected_roles:
        raise CacheMismatch("classified kernel role ordering does not implement the source operation")
    span = metadata.get("d_tile",32) * metadata.get("d_vector",1)
    if check_tiles and any(kernel["grid"][1] != (spec.feature_dim + span - 1) // span
                           for kernel in metadata["kernels"][2:]):
        raise CacheMismatch("classified worker grid must cover exactly ceil(D/(d_tile*d_vector)) feature tiles")

def validate_manifest(metadata: dict[str, Any]) -> None:
    if not isinstance(metadata, dict) or type(metadata.get("schema")) is not int or metadata["schema"] != 1:
        raise CacheMismatch("unsupported kernel manifest schema")
    if metadata.get("abi") != ABI:
        raise CacheMismatch("unsupported kernel ABI; expected mlir-memref64-v1")
    if type(metadata.get("mask", False)) is not bool:
        raise CacheMismatch("mask presence must be boolean")
    if metadata.get("mask", False) and metadata.get("fusion") == "not_applicable":
        raise CacheMismatch("additive mask is only supported by softmax schedules")
    dispatch_abi = metadata.get("dispatch_abi")
    if dispatch_abi not in (None, "classified-v3"):
        raise CacheMismatch("unsupported dispatch ABI")
    classified = dispatch_abi == "classified-v3"
    if type(metadata.get("debug_visits", False)) is not bool:
        raise CacheMismatch("debug_visits must be boolean")
    declared_bytes = metadata.get("workspace_bytes", 0)
    if type(declared_bytes) is not int or declared_bytes < 0:
        raise CacheMismatch("workspace_bytes must be a nonnegative integer")
    if declared_bytes:
        raise CacheMismatch("workspace bytes are derived from workspace shapes; omit static workspace_bytes")
    if "workspace_bytes_expr" in metadata and not isinstance(metadata["workspace_bytes_expr"], str):
        raise CacheMismatch("workspace_bytes_expr is a descriptive string, never executable code")
    classes = metadata.get("classes", [])
    if not isinstance(classes, list) or any(value not in ("warp", "cta", "split") for value in classes):
        raise CacheMismatch("classes must list supported mapping classes")
    if classified and (not classes or classes != [name for name in ("warp", "cta", "split") if name in classes]):
        raise CacheMismatch("classified classes must be unique and ordered warp, cta, split")
    thresholds = metadata.get("thresholds", [])
    if (not isinstance(thresholds, list)
            or any(type(value) is not int or value < 0 or value > 2**63-1 for value in thresholds)
            or thresholds != sorted(set(thresholds))):
        raise CacheMismatch("thresholds must be ascending nonnegative i64 values")
    if "class_fusion" in metadata or metadata.get("fusion") == "mixed":
        class_fusion(metadata)
    if classified:
        if metadata.get("backend") != "classified-persistent":
            raise CacheMismatch("classified ABI requires the classified-persistent backend")
        if (type(metadata.get("split_parts")) is not int or metadata["split_parts"] != 8
                or thresholds != [32, 4096] or "debug_visits" not in metadata):
            raise CacheMismatch("classified ABI requires split_parts=8, thresholds=[32,4096] and explicit debug_visits")
        if metadata.get("fusion") not in ("not_applicable", "online", "three_pass", "mixed"):
            raise CacheMismatch("classified ABI requires an explicit fusion schedule")
    if metadata.get("fallback_reason") is not None and not isinstance(metadata["fallback_reason"], str):
        raise CacheMismatch("fallback_reason must be a string or null")
    if classified and "workspace" not in metadata:
        raise CacheMismatch("classified ABI requires an explicit workspace list")
    workspaces = metadata.get("workspace", [])
    if not isinstance(workspaces, list):
        raise CacheMismatch("workspace must be an array")
    names = ["values", "offsets", "out"] + (["mask"] if metadata.get("mask", False) else [])
    for workspace in workspaces:
        if not isinstance(workspace, dict) or not isinstance(workspace.get("name"), str):
            raise CacheMismatch("workspace must be named")
        if workspace["name"] in names or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", workspace["name"]):
            raise CacheMismatch("duplicate or empty workspace name")
        names.append(workspace["name"])
        if not isinstance(workspace.get("dtype"), str) or workspace["dtype"] not in WORKSPACE_DTYPES:
            raise CacheMismatch("workspace dtype must be i32, i64 or f32")
        shape = workspace.get("shape")
        if not isinstance(shape, list) or len(shape) not in (1, 2):
            raise CacheMismatch("workspace must have rank one or two")
        for value in shape:
            if valid_dimension(value, SHAPE_SYMBOLS):
                continue
            raise CacheMismatch("invalid workspace shape")
    kernels = metadata.get("kernels")
    if not isinstance(kernels, list) or not kernels:
        raise CacheMismatch("kernel manifest has no launch sequence")
    validate_optimization_metadata(metadata)
    materialized = metadata.get("pointwise_materialized",False)
    if type(metadata.get("alias_safe", False)) is not bool:
        raise CacheMismatch("alias_safe must be boolean")
    symbols = SHAPE_SYMBOLS | (CLASSIFIED_GRID_SYMBOLS if classified else frozenset())
    kernel_names = set()
    for index, kernel in enumerate(kernels):
        if (not isinstance(kernel, dict) or not isinstance(kernel.get("name"), str)
                or not kernel["name"] or "\0" in kernel["name"]):
            raise CacheMismatch("invalid kernel symbol")
        if kernel["name"] in kernel_names:
            raise CacheMismatch("kernel symbols must be unique in the launch sequence")
        kernel_names.add(kernel["name"])
        for field in ("grid", "block"):
            shape = kernel.get(field)
            if not isinstance(shape, list) or len(shape) != 3:
                raise CacheMismatch(f"kernel {field} must have three dimensions")
            for axis, value in enumerate(shape):
                limit = GRID_LIMITS[axis] if field == "grid" else (1024, 1024, 64)[axis]
                if type(value) is int and 1 <= value <= limit:
                    continue
                if field == "grid" and isinstance(value, str) and value in symbols:
                    continue
                raise CacheMismatch(f"unsupported kernel {field} dimension {value!r}")
        if kernel.get("args") != names:
            raise CacheMismatch("kernel argument order must be values, offsets, out then declared workspaces")
        if type(kernel.get("shared_bytes", 0)) is not int or not 0 <= kernel.get("shared_bytes", 0) <= 49152:
            raise CacheMismatch("invalid shared memory allocation")
        if (type(kernel.get("static_shared_bytes", 0)) is not int
                or not 0 <= kernel.get("static_shared_bytes", 0) <= 49152):
            raise CacheMismatch("invalid static shared memory allocation")
        product = 1
        for size in kernel["block"]:
            product *= size
        if product > 1024:
            raise CacheMismatch("block exceeds 1024 threads")
        if classified and not (materialized and index == 0):
            _validate_classified_kernel(kernel, index - int(materialized), classes)
    if classified:
        _validate_classified_structure(metadata)
    if "validation" in metadata:
        validate_validation_metadata(metadata["validation"], main_names=kernel_names)
    validate_resources_metadata(metadata)

def _validate_classified_kernel(kernel: dict[str, Any], index: int, classes: list[str]) -> None:
    if kernel.get("shared_bytes", 0) != 0:
        raise CacheMismatch("classified kernels use static workgroup memory")
    if index < 2:
        role, grid = ("init", [1, 1, 1]) if index == 0 else ("classify", ["ceil_B_128", 1, 1])
        if (kernel.get("role") != role or kernel["grid"] != grid or kernel["block"] != [128, 1, 1]
                or "segment_class" in kernel):
            raise CacheMismatch("classified launches must begin with init then classify using the fixed ABI grids")
        return
    segment_class, role = kernel.get("segment_class"), kernel.get("role")
    if segment_class not in classes or not isinstance(role, str):
        raise CacheMismatch("classified worker requires an active segment_class and role")
    known = {"warp": {"warp_" + phase for phase in ("sum", "max", "online", "max_stats", "sum_stats", "normalize")},
             "cta": {"cta_" + phase for phase in ("sum", "max", "online", "max_stats", "sum_stats", "normalize")},
             "split": {"split_partial_sum", "split_combine_sum", "split_partial_max", "split_combine_max",
                       "split_combine_max_stats", "split_partial_exp_sum", "split_combine_sum_stats", "split_normalize",
                       "split_partial_online", "split_combine_online"}}
    if role not in known[segment_class]:
        raise CacheMismatch("worker role does not match its segment class")
    split_z = 8 if role.startswith("split_partial_") or role == "split_normalize" else 1
    grid = kernel["grid"]
    if (grid[0] != "min_B_resident" or type(grid[1]) is not int or not 1 <= grid[1] <= 1024 or grid[2] != split_z
            or kernel["block"] != ([32, 1, 1] if segment_class == "warp" else [128, 1, 1])):
        raise CacheMismatch("classified worker grid/block differs from the persistent scheduling ABI")

def _validate_classified_structure(metadata: dict[str, Any]) -> None:
    """Validate complete schedules even without a source ProgramSpec."""
    core = computation_metadata(metadata)
    workspace = core["workspace"]
    if len(workspace) < 2 or workspace[:2] != [
            {"name": "class_counts", "shape": [3], "dtype": "i64"},
            {"name": "segment_ids", "shape": [3, "B"], "dtype": "i64"}]:
        raise CacheMismatch("classified workspace must begin with i64 counters and segment worklists")
    widths = [metadata["workspace"][-1]["shape"][1]] if metadata.get("pointwise_materialized",False) else []
    for buffer in workspace[2:]:
        if buffer["name"] in ("partials", "partial_sums"):
            if (buffer["dtype"] != "f32" or len(buffer["shape"]) != 2 or buffer["shape"][0] != "B_split"
                    or type(buffer["shape"][1]) is not int or buffer["shape"][1] % 8):
                raise CacheMismatch("split partials must be f32[B_split,8*D]")
            widths.append(buffer["shape"][1] // 8)
        elif buffer["name"] in ("max_stats", "sum_stats"):
            if buffer["dtype"] != "f32" or len(buffer["shape"]) != 2 or buffer["shape"][0] != "B":
                raise CacheMismatch("softmax statistics must be f32[B,D]")
            widths.append(buffer["shape"][1])
        elif buffer != {"name": "visits", "shape": ["B"], "dtype": "i32"}:
            raise CacheMismatch("unknown classified workspace")
    if any(type(width) is not int or not 1 <= width <= 1024 for width in widths) or len(set(widths)) > 1:
        raise CacheMismatch("classified partial/statistic feature widths disagree")
    fusion = metadata["fusion"]
    roles = [kernel.get("role") for kernel in core["kernels"]][2:]
    operation = "softmax" if fusion in ("online", "three_pass", "mixed") else (
        "sum" if roles and roles[0].endswith("_sum") else "max")
    inferred = ProgramSpec(operation, feature_dim=widths[0] if widths else 1,
                           debug_visits=metadata["debug_visits"])
    validate_classified_spec(inferred, metadata, check_tiles=bool(widths))
