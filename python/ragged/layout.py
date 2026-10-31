"""Finite manifest dimension vocabulary and overflow-checked host layouts.

All inputs here are host tensor shapes or compiler metadata. Device counters
never determine allocation sizes or launch geometry, and expressions are not
evaluated as Python or parsed as an arithmetic language.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any, Mapping

from .errors import CacheMismatch, ValidationError

I64_MAX = 2**63 - 1
GRID_LIMITS = (2**31 - 1, 65535, 65535)
WORKSPACE_DTYPES = {"i32": 4, "i64": 8, "f32": 4}
SHAPE_SYMBOLS = frozenset(("N", "B", "D"))
CLASSIFIED_GRID_SYMBOLS = frozenset(("ceil_B_128", "min_B_resident"))
VALIDATION_GRID_SYMBOLS = frozenset(("ceil_B1_128",))


@dataclass(frozen=True)
class WorkspaceLayout:
    name: str
    dtype: str
    shape: tuple[int, ...]
    nbytes: int


def valid_dimension(value: Any, symbols: frozenset[str]) -> bool:
    return ((type(value) is int and 1 <= value <= I64_MAX)
            or (isinstance(value, str) and value in symbols))


def resolve_layout(metadata: dict[str, Any], *, n: int, b: int, d: int,
                   workers: Mapping[str, int] | None = None
                   ) -> tuple[tuple[WorkspaceLayout, ...], tuple[tuple[int, int, int], ...], int]:
    """Preflight every layout before allocation or kernel launch.

    ``workers`` maps each kernel whose grid x is ``min_B_resident`` to the
    number of its (y, z) block groups the device holds at once; the runtime
    measures it with the occupancy API after loading the module.
    """
    for name, value in (("N", n), ("B", b), ("D", d)):
        if type(value) is not int or not 0 <= value <= I64_MAX:
            raise ValidationError(f"{name} must fit a nonnegative signed i64")
    if not 1 <= d <= 1024:
        raise ValidationError("D must lie in [1, 1024]")
    dimensions = {"N": n, "B": b, "D": d,
                  "ceil_B_128": b // 128 + (b % 128 != 0)}
    classified = metadata.get("dispatch_abi") == "classified-v2"
    validation = metadata.get("abi") == "offsets-check-v1"
    if validation:
        if b == I64_MAX:
            raise ValidationError("B+1 offsets dimension overflows signed i64")
        dimensions["ceil_B1_128"] = (b + 1) // 128 + ((b + 1) % 128 != 0)
    workspaces, grids, total = [], [], 0
    for workspace in metadata.get("workspace", []):
        shape = workspace["shape"]
        if any(not valid_dimension(value, SHAPE_SYMBOLS) for value in shape):
            raise CacheMismatch("workspace dimensions must be static integers or N/B/D symbols")
        concrete = tuple(dimensions[value] if isinstance(value, str) else value for value in shape)
        # Checking suffix products also establishes representable contiguous
        # strides when an earlier dynamic dimension is zero.
        for index in range(len(concrete)):
            if prod(concrete[index:]) > I64_MAX:
                raise ValidationError("workspace shape/stride product overflows signed i64")
        nbytes = prod(concrete) * WORKSPACE_DTYPES[workspace["dtype"]]
        if nbytes > I64_MAX or total > I64_MAX - nbytes:
            raise ValidationError("workspace byte size or total allocation overflows signed i64")
        total += nbytes
        workspaces.append(WorkspaceLayout(workspace["name"], workspace["dtype"], concrete, nbytes))
    allowed_symbols = SHAPE_SYMBOLS | (CLASSIFIED_GRID_SYMBOLS if classified else frozenset())
    if validation:
        allowed_symbols |= VALIDATION_GRID_SYMBOLS
    for kernel in metadata["kernels"]:
        grid = []
        for axis, value in enumerate(kernel["grid"]):
            if not valid_dimension(value, allowed_symbols):
                raise CacheMismatch("unsupported launch dimension")
            if value == "min_B_resident":
                count = (workers or {}).get(kernel["name"])
                if type(count) is not int or count < 1:
                    raise CacheMismatch(f"no device worker count for {kernel['name']}")
                dimension = min(b, count)
            else:
                dimension = dimensions[value] if isinstance(value, str) else value
            if dimension > GRID_LIMITS[axis]:
                raise ValidationError(f"grid axis {axis} exceeds the CUDA launch limit {GRID_LIMITS[axis]}")
            grid.append(dimension)
        grids.append(tuple(grid))
    return tuple(workspaces), tuple(grids), total
