"""Strict metadata and fixed-size host status decoding for checked execution."""
from __future__ import annotations

from typing import Any

from .errors import CacheMismatch, ValidationError
from .layout import I64_MAX
from .model import ProgramSpec

CONDITIONS = ("first_nonzero", "final_mismatch", "negative", "above_n",
              "nonmonotone", "below_min_len", "above_max_len")
STATUS_WORKSPACE = [{"name": "validation_status", "dtype": "i64", "shape": [7]}]
STATUS_ARGUMENTS = ["values", "offsets", "validation_status"]


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
