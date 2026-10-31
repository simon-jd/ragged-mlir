"""Validate measured ptxas resource evidence separately from IR estimates."""
from __future__ import annotations

from math import prod
from typing import TYPE_CHECKING, Any

from .errors import CacheMismatch

if TYPE_CHECKING:
    from .model import ProgramSpec

DEFAULT_SMEM_PER_CTA = 49152
DEFAULT_REGS_BUDGET = 65536
I64_MAX = 2**63 - 1
RESOURCE_FIELDS = frozenset({
    "registers_per_thread", "threads_per_cta", "registers_per_cta",
    "static_shared_bytes", "dynamic_shared_bytes", "shared_bytes_per_cta",
    "stack_frame_bytes", "spill_store_bytes", "spill_load_bytes",
})


def validate_resources_metadata(metadata: dict[str, Any]) -> None:
    """Require a complete, internally consistent record when resources exist.

    The compiler obtains these numbers from its actual cubin assembly. Runtime
    checks their structure and source/launch binding; it does not relabel the
    kernel manifest's IR allocation estimates as measurements. Legacy objects
    without this field are handled separately by validate_resources_spec.
    """
    if "resources" not in metadata:
        return
    resources = metadata["resources"]
    if (not isinstance(resources, dict)
            or set(resources) != {"status", "source", "requested", "kernels"}
            or resources["status"] != "measured" or resources["source"] != "ptxas"):
        raise CacheMismatch("cubin resources must contain measured ptxas evidence")
    requested = resources["requested"]
    if (not isinstance(requested, dict) or set(requested) != {"smem_per_cta", "regs_budget"}
            or any(type(value) is not int or not 1 <= value <= I64_MAX for value in requested.values())):
        raise CacheMismatch("resource budgets must be positive signed i64 integers")
    entries = resources["kernels"]
    launches = metadata["kernels"] + metadata.get("validation", {}).get("kernels", [])
    if not isinstance(entries, list) or len(entries) != len(launches):
        raise CacheMismatch("resource evidence must cover every main and validation kernel")
    names = set()
    for entry, kernel in zip(entries, launches):
        if not isinstance(entry, dict) or set(entry) != RESOURCE_FIELDS | {"name"}:
            raise CacheMismatch("ptxas kernel resource fields are incomplete or unsupported")
        name = entry["name"]
        if not isinstance(name, str) or name != kernel["name"] or name in names:
            raise CacheMismatch("resource kernel names must uniquely follow main then validation launch order")
        names.add(name)
        if any(type(entry[field]) is not int or not 0 <= entry[field] <= I64_MAX for field in RESOURCE_FIELDS):
            raise CacheMismatch("ptxas resource measurements must be nonnegative signed i64 integers")
        threads = prod(kernel["block"])
        registers = entry["registers_per_thread"] * threads
        dynamic = kernel.get("shared_bytes", 0)
        shared = entry["static_shared_bytes"] + dynamic
        if entry["threads_per_cta"] != threads or not 1 <= threads <= 1024:
            raise CacheMismatch("resource threads_per_cta differs from the kernel launch block")
        if registers > I64_MAX or entry["registers_per_cta"] != registers:
            raise CacheMismatch("resource registers_per_cta must equal measured registers_per_thread times threads")
        if (shared > I64_MAX or entry["dynamic_shared_bytes"] != dynamic
                or entry["shared_bytes_per_cta"] != shared):
            raise CacheMismatch("resource shared_bytes_per_cta must equal measured static plus launch dynamic bytes")
        if registers > requested["regs_budget"]:
            raise CacheMismatch(f"kernel {name} measured registers exceed the requested regs_budget")
        if shared > requested["smem_per_cta"]:
            raise CacheMismatch(f"kernel {name} measured shared bytes exceed the requested smem_per_cta")


def validate_resources_spec(spec: ProgramSpec, metadata: dict[str, Any]) -> None:
    requested = {"smem_per_cta": spec.smem_per_cta, "regs_budget": spec.regs_budget}
    if "resources" not in metadata:
        if requested != {"smem_per_cta": DEFAULT_SMEM_PER_CTA, "regs_budget": DEFAULT_REGS_BUDGET}:
            raise CacheMismatch("nondefault resource budgets require measured ptxas evidence")
        return
    if metadata["resources"]["requested"] != requested:
        raise CacheMismatch("resource budgets differ from the source specialization")


def resource_report(metadata: dict[str, Any]) -> dict[str, Any] | None:
    """Detached compiled-resource evidence; absence never implies measurement."""
    resources = metadata.get("resources")
    if resources is None:
        return None
    return resources | {"requested": dict(resources["requested"]),
                        "kernels": [dict(entry) for entry in resources["kernels"]]}
