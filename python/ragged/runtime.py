"""PyTorch current-stream launch adapter; imports torch/native code only on use."""
from __future__ import annotations

import atexit
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass, replace
import importlib
import json
import re
import threading
import time
from typing import Any

from .cache import Artifact
from .errors import CacheMismatch, RuntimeUnavailable, ValidationError
from .layout import (CLASSIFIED_GRID_SYMBOLS, GRID_LIMITS, I64_MAX, SHAPE_SYMBOLS,
                     WORKSPACE_DTYPES, resolve_layout, valid_dimension)
from .model import ExecutionReport, ProgramSpec, canonical_json
from .optimizations import (Optimizations, class_fusion, computation_metadata, validate_optimization_metadata,
                            validate_optimization_spec)
from .resources import resource_report, validate_resources_metadata, validate_resources_spec
from .selection import default_model
from .validation import raise_for_status, validate_validation_hints, validate_validation_metadata

_module_lock = threading.RLock()
# Loaded modules by (device ordinal, artifact digest), shared by every program
# compiled to the same artifact. A module must not be unloaded while kernels
# launched from it may still run, and cuModuleUnload is not stream-ordered. So
# when the last program holding a module is closed, an event is recorded on
# every stream the module launched on, and the module is unloaded once all of
# them have completed (checked whenever modules are acquired or released, and
# waited for at exit).
_resident_modules: dict[tuple[int, str], Any] = {}
_holders: Counter = Counter()                       # live programs per module
_streams: dict[tuple[int, str], dict[int, Any]] = {}  # streams each module launched on
_retiring: list[tuple[Any, list[Any]]] = []         # (module, events) awaiting completion
_sm89_devices: set[int] = set()                     # device ordinals already checked
# Device-sized worker counts per module: for each kernel whose grid x is
# min_B_resident, how many of its (y, z) block groups the device holds at once.
_workers: dict[tuple[int, str], dict[str, int]] = {}
# Native launch plans per module, by (N, B, concurrent).
_plans: dict[tuple[int, str], dict[tuple[int, int, bool], Any]] = {}
# Submission order of concurrent class chains; the last runs on the caller's stream.
# Concurrent submission order. The CTA chain goes first: when the split
# chain's first kernel, a full resident wave, was submitted first and
# dispatched beside the CTA workers from another hardware queue, uniform
# D=128 calls took 20 to 40 us longer (with CUDA_DEVICE_MAX_CONNECTIONS=1,
# which serializes dispatch, they did not). The last chain runs on the
# caller's stream.
_CLASS_ORDER = ("cta", "split", "warp")


def _reap(wait: bool = False) -> None:
    """Drop retired modules whose launches have completed; the native
    destructor unloads them in the context that loaded them."""
    with _module_lock:
        pending = []
        for module, events in _retiring:
            if wait:
                for event in events:
                    event.synchronize()
            elif not all(event.query() for event in events):
                pending.append((module, events))
        _retiring[:] = pending


def _release(key: tuple[int, str]) -> None:
    with _module_lock:
        _holders[key] -= 1
        if _holders[key] > 0:
            return
        del _holders[key]
        module = _resident_modules.pop(key, None)
        streams = _streams.pop(key, {})
        _workers.pop(key, None)
        _plans.pop(key, None)
        if module is None:
            return
        torch = _torch()
        events = []
        try:
            for stream in streams.values():
                event = torch.cuda.Event()
                event.record(stream)
                events.append(event)
        except Exception:  # noqa: BLE001 - a stream that cannot record: wait for the device instead
            torch.cuda.synchronize()
            events = []
        _retiring.append((module, events))
        _reap()


@atexit.register
def _unload_all() -> None:
    # Unload before interpreter teardown, after the work of every module ends.
    try:
        with _module_lock:
            for key in list(_resident_modules):
                _holders[key] = 1
                _release(key)
            _reap(wait=True)
    except Exception:  # noqa: BLE001 - best effort during shutdown
        pass


def _worker_counts(module: Any, metadata: dict[str, Any]) -> dict[str, int]:
    counts = {}
    for kernel in metadata["kernels"]:
        if kernel["grid"][0] == "min_B_resident":
            threads = kernel["block"][0] * kernel["block"][1] * kernel["block"][2]
            resident = module.resident_blocks(kernel["name"], threads, kernel.get("shared_bytes", 0))
            counts[kernel["name"]] = max(1, resident // (kernel["grid"][1] * kernel["grid"][2]))
    return counts


def _launch_schedule(metadata: dict[str, Any]) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]:
    """Kernel indices of the shared prologue and of each segment class's worker
    chain, in manifest order. Workers of different classes read and write
    disjoint queue rows, segments and workspace rows (classified-abi.md), so
    their chains may run concurrently once classification has finished."""
    kernels = metadata["kernels"]
    if metadata.get("dispatch_abi") != "classified-v3":
        return tuple(range(len(kernels))), ()
    prologue, chains = [], {}
    for index, kernel in enumerate(kernels):
        if "segment_class" in kernel:
            chains.setdefault(kernel["segment_class"], []).append(index)
        else:
            prologue.append(index)
    return tuple(prologue), tuple(tuple(chain) for chain in chains.values())


@dataclass(frozen=True)
class _Plan:
    id: int                         # native launch plan
    nbytes: int                     # scratch bytes
    offsets: tuple[int, ...]        # scratch offset of each workspace buffer
    names: tuple[str, ...]          # kernels in submission order
    grids: tuple[tuple[int, int, int], ...]
    streams: int


SCRATCH_ALIGNMENT = 256


def _scratch_layout(layouts: Any) -> tuple[list[int], list[int], list[int], int]:
    """Memref descriptors of the workspace buffers inside one scratch
    allocation, with their pointer fields as offsets from its base, the
    indices of those fields, each buffer's offset and the total bytes."""
    descriptors, relocations, offsets, end = [], [], [], 0
    for layout in layouts:
        offset = -(-end // SCRATCH_ALIGNMENT) * SCRATCH_ALIGNMENT
        strides, stride = [], 1
        for size in reversed(layout.shape):
            strides.append(stride)
            stride *= max(size, 1)
        relocations += [len(descriptors), len(descriptors) + 1]
        descriptors += [offset, offset, 0, *layout.shape, *reversed(strides)]
        offsets.append(offset)
        end = offset + layout.nbytes
    if end > I64_MAX:
        raise ValidationError("aligned workspace bytes overflow signed i64")
    return descriptors, relocations, offsets, end


def _torch():
    try:
        return importlib.import_module("torch")
    except ImportError as exc:
        raise RuntimeUnavailable("PyTorch is not installed; install the pinned GPU environment to execute") from exc


def memref_arguments(tensor: Any) -> list[int]:
    """Flatten ranked contiguous memref with data_ptr as aligned base, offset zero.

    data_ptr already includes a tensor view's storage offset. This is intentional:
    the descriptor must not count the storage offset a second time.
    """
    pointer = int(tensor.data_ptr())
    shape, strides = tuple(tensor.shape), tuple(tensor.stride())
    if not 0 <= pointer < 2**64:
        raise ValidationError("memref pointer does not fit a 64-bit address")
    if len(shape) not in (1, 2) or len(strides) != len(shape):
        raise ValidationError("memref must have rank one or two with matching strides")
    if min(shape + strides) < 0 or max(shape + strides) > I64_MAX:
        raise ValidationError("memref dimensions and strides must fit nonnegative signed i64")
    if len(shape) == 2:
        if shape[0] and shape[1] and (shape[0] - 1) * strides[0] + (shape[1] - 1) * strides[1] > I64_MAX:
            raise ValidationError("memref address calculation overflows signed i64")
    elif shape[0] and (shape[0] - 1) * strides[0] > I64_MAX:
        raise ValidationError("memref address calculation overflows signed i64")
    return [pointer, pointer, 0, *shape, *strides]


def _overlaps(left: Any, right: Any) -> bool:
    # Validated row-major tensors have contiguous byte intervals; no private API.
    lstart, rstart = left.data_ptr(), right.data_ptr()
    lend = lstart + left.numel() * left.element_size()
    rend = rstart + right.numel() * right.element_size()
    return lstart < rend and rstart < lend


class CompiledProgram:
    ABI = "mlir-memref64-v1"

    def __init__(self, spec: ProgramSpec, artifact: Artifact, *, cache_hit: bool = False,
                 naive: "CompiledProgram | None" = None):
        if artifact.key.ir_hash != spec.ir_hash:
            raise CacheMismatch("compiled program source does not match cached IR hash")
        for field in ("op", "dtype", "feature_dim", "offsets_dtype", "strategy", "chip"):
            if getattr(artifact.key, field) != getattr(spec, field):
                raise CacheMismatch(f"compiled program {field} differs from artifact key")
        if not artifact.cubin.startswith(b"\x7fELF"):
            raise CacheMismatch("runtime requires an ELF cubin")
        self.validate_metadata(artifact.metadata)
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
            self._validate_classified_spec(spec, artifact.metadata)
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
        self.spec, self.artifact, self.cache_hit = spec, artifact, cache_hit
        self._metadata = json.loads(canonical_json(artifact.metadata))
        self._module_key: tuple[int, str] | None = None
        # Per-call work that depends only on the artifact, computed once: the
        # key digest (a SHA-256 of the canonical key), the static report fields,
        # and launch layouts per (main/validation, N, B).
        self._digest = artifact.key.digest
        self._report_fields = self._optimization_report()
        self._schedule = _launch_schedule(self._metadata)
        self._concurrent_chains = tuple(sorted(self._schedule[1], key=lambda chain: _CLASS_ORDER.index(
            self._metadata["kernels"][chain[0]]["segment_class"])))
        self._layouts: dict[tuple[bool, int, int, int], Any] = {}
        self._selects = (spec.strategy == "auto" and not spec.debug_visits
                         and computation_metadata(self._metadata).get("dispatch_abi") == "classified-v3")
        self._schedule_name = "classified" if self._selects else spec.strategy
        if naive is not None and (not self.selects or naive.spec != replace(spec, strategy="naive")):
            raise CacheMismatch("a block-per-segment companion must be the naive strategy of a classifying program")
        # The block-per-segment schedule of the same source, chosen per call
        # by the cost model (ragged.selection) when it predicts it faster.
        self.naive = naive
        self._choices: dict[tuple[int, int, int | None], tuple[str, dict[str, Any]]] = {}

    @property
    def selects(self) -> bool:
        """Whether this program classifies on the device, so that one block
        per segment is an alternative schedule worth choosing per call."""
        return self._selects

    def choose(self, n: int, b: int, max_len: int | None) -> tuple[str, dict[str, Any]]:
        """The schedule run(select="auto") uses for this input and why."""
        if self.naive is None:
            return ("classified" if self.selects else self.spec.strategy), {}
        key = (n, b, max_len)
        choice = self._choices.get(key)
        if choice is None:
            if len(self._choices) >= 256:
                self._choices.clear()
            span = self._metadata.get("d_tile", 32) * self._metadata.get("d_vector", 1)
            element = 2 if self.spec.dtype == "bf16" else 4
            choice = self._choices[key] = default_model().choose(self.spec.op, n, b, self.spec.feature_dim,
                                                                 max_len, span, element)
        return choice

    def _layout(self, metadata: dict[str, Any], validation: bool, n: int, b: int,
                device: int, workers: dict[str, int] | None = None):
        key = (validation, n, b, device)
        layout = self._layouts.get(key)
        if layout is None:
            if len(self._layouts) >= 256:
                self._layouts.clear()
            layout = self._layouts[key] = resolve_layout(metadata, n=n, b=b, d=self.spec.feature_dim,
                                                         workers=workers)
        return layout

    def close(self) -> None:
        """Release this program's loaded module. The module is unloaded once no
        program holds it and the work launched from it has completed; running
        the program again loads it again."""
        key, self._module_key = getattr(self, "_module_key", None), None
        if key is not None:
            _release(key)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001 - finalizers must not raise
            pass

    @staticmethod
    def _validate_classified_spec(spec: ProgramSpec, metadata: dict[str, Any], *, check_tiles: bool = True) -> None:
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

    @classmethod
    def validate_metadata(cls, metadata: dict[str, Any]) -> None:
        if not isinstance(metadata, dict) or type(metadata.get("schema")) is not int or metadata["schema"] != 1:
            raise CacheMismatch("unsupported kernel manifest schema")
        if metadata.get("abi") != cls.ABI:
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
                cls._validate_classified_kernel(kernel, index - int(materialized), classes)
        if classified:
            cls._validate_classified_structure(metadata)
        if "validation" in metadata:
            validate_validation_metadata(metadata["validation"], main_names=kernel_names)
        validate_resources_metadata(metadata)

    @staticmethod
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

    @classmethod
    def _validate_classified_structure(cls, metadata: dict[str, Any]) -> None:
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
        cls._validate_classified_spec(inferred, metadata, check_tiles=bool(widths))

    def _optimization_report(self) -> dict[str, Any]:
        effective = self._metadata.get("optimizations",{}).get("effective")
        return {"optimizations":{"requested":self.spec.optimizations.to_dict(),
                                 "effective":dict(effective) if effective is not None else None},
                "d_tile":self._metadata.get("d_tile"),
                "d_vector":self._metadata.get("d_vector",1),
                "pointwise_materialized":self._metadata.get("pointwise_materialized",False),
                "resources":resource_report(self._metadata)}

    def _validate_tensors(self, values, offsets, out, torch, *, checked: bool = False, mask=None) -> tuple[int, int]:
        if self.spec.mask != (mask is not None):
            raise ValidationError("mask presence must match the compiled specialization")
        tensors = [("values", values), ("offsets", offsets), ("out", out)]
        if mask is not None:
            tensors.append(("mask", mask))
        for name, tensor in tensors:
            if not isinstance(tensor, torch.Tensor):
                raise ValidationError(f"{name} must be a torch.Tensor")
            if not tensor.is_cuda:
                raise RuntimeUnavailable(f"{name} must be on an sm_89 CUDA device; no CPU execution fallback exists")
            if not tensor.is_contiguous():
                raise ValidationError(f"{name} must be row-major contiguous")
            if tensor.requires_grad:
                raise ValidationError("autograd is outside the supported forward-only contract")
        if values.device != offsets.device or values.device != out.device:
            raise ValidationError("values, offsets and out must use the same CUDA device")
        if values.ndim != 2 or values.shape[1] != self.spec.feature_dim:
            raise ValidationError("values must have shape [N, static D]")
        if offsets.ndim != 1 or offsets.numel() < 1:
            raise ValidationError("offsets must have shape [B+1], including [0] when B=0")
        dtype = {"f32": torch.float32, "bf16": torch.bfloat16}[self.spec.dtype]
        offset_dtype = {"i32": torch.int32, "i64": torch.int64}[self.spec.offsets_dtype]
        if values.dtype != dtype or out.dtype != dtype or offsets.dtype != offset_dtype:
            raise ValidationError("tensor dtype differs from compiled dtype/offsets specialization")
        vector = self._metadata.get("d_vector", 1)
        if vector > 1:
            # Kernels read and write d_vector adjacent features per access;
            # D is a multiple of it, so only the base addresses can misalign.
            for name, tensor in (("values", values), ("out", out), ("mask", mask)):
                if tensor is not None and tensor.numel():
                    need = min(16, vector * tensor.element_size())
                    if tensor.data_ptr() % need:
                        raise ValidationError(f"{name} must be {need}-byte aligned: this specialization accesses "
                                              f"{vector} features at a time (compile with vectorize=False for "
                                              "unaligned views)")
        n, b = values.shape[0], offsets.numel() - 1
        if b == 0 and n != 0 and not checked:
            raise ValidationError("B=0 requires N=0; values cannot exist without segments")
        if self.spec.offsets_dtype == "i32" and n > 2**31 - 1:
            raise ValidationError("N exceeds the i32 offsets range")
        expected = (n if self.spec.op == "softmax" else b, self.spec.feature_dim)
        if tuple(out.shape) != expected:
            raise ValidationError(f"out shape must be {expected}")
        if mask is not None:
            if mask.dtype != torch.float32 or tuple(mask.shape) != tuple(values.shape):
                raise ValidationError("mask must have f32 dtype and exactly values shape [N,D]; broadcasting is unsupported")
            if mask.device != values.device:
                raise ValidationError("mask must use the values CUDA device")
            if _overlaps(mask, out) or _overlaps(mask, offsets):
                raise ValidationError("mask must not overlap out or offsets")
        if _overlaps(offsets, values) or _overlaps(offsets, out):
            raise ValidationError("offsets must not alias values or out")
        overlap = _overlaps(values, out)
        exact = values.data_ptr() == out.data_ptr() and tuple(values.shape) == tuple(out.shape)
        if overlap:
            if not exact or self.spec.alias != "inplace" or not self._metadata.get("alias_safe", False):
                raise ValidationError("out aliases values without a compiler-proven safe in-place schedule")
        elif self.spec.alias == "inplace" and not (exact and values.numel() == 0):
            raise ValidationError("in-place specialization requires out and values to be the same buffer")
        return n, b

    def _plan(self, module_key: tuple[int, str], module: Any, layouts: Any, grids: Any, n: int, b: int,
              concurrent: bool) -> "_Plan":
        """The module's launch plan for this problem size, built on first use:
        kernels in submission order with their lanes, and the scratch layout."""
        key = (n, b, concurrent)
        with _module_lock:
            plans = _plans.setdefault(module_key, {})
            plan = plans.get(key)
            if plan is not None:
                return plan
            kernels = self._metadata["kernels"]
            prologue, chains = self._schedule
            lanes = {}
            if concurrent and len(chains) > 1:
                # One side lane per class chain but the last, which runs on
                # the caller's stream, in _CLASS_ORDER.
                chains = self._concurrent_chains
                for lane, chain in enumerate(chains[:-1], start=1):
                    lanes.update(dict.fromkeys(chain, lane))
            order = [index for index in (*prologue, *(index for chain in chains for index in chain))
                     if 0 not in grids[index]]
            steps = [(kernels[index]["name"], grids[index], tuple(kernels[index]["block"]),
                      kernels[index].get("shared_bytes", 0), lanes.get(index, 0)) for index in order]
            descriptors, relocations, offsets, nbytes = _scratch_layout(layouts)
            if len(plans) >= 256:
                module.clear_plans()
                plans.clear()
            plan = plans[key] = _Plan(module.add_plan(steps, descriptors, relocations), nbytes, tuple(offsets),
                                      tuple(kernels[index]["name"] for index in order),
                                      tuple(grids[index] for index in order), 1 + max(lanes.values(), default=0))
            return plan

    def run(self, values, offsets, out, stream=None, mode: str = "validated", *,
            debug_buffers: dict[str, Any] | None = None, mask=None,
            concurrent: bool = True, select: str = "auto", max_len: int | None = None) -> ExecutionReport:
        """Launch on the requested stream, optionally checking device offsets.

        Validated mode performs no device readback or host synchronization.
        Checked mode launches the artifact's offsets validator and copies seven
        i64 slots to pinned host memory, then synchronizes this stream before
        submitting the main computation. Invalid offsets never reach main.
        For debug specializations, an empty ``debug_buffers`` dictionary may
        retain the named workspace tensors. The caller must synchronize the
        execution stream before inspecting their contents on the host.
        With ``concurrent`` (the default), the worker chains of different
        segment classes run on separate streams between a fork after
        classification and a join on the execution stream.
        A program compiled with a block-per-segment companion (``naive``)
        chooses between the two schedules per call: ``select="auto"`` asks
        the cost model, given the longest segment length ``max_len`` when the
        caller knows it (a wrong value costs time, never correctness);
        ``"classified"`` and ``"naive"`` force a schedule.
        """
        if select not in ("auto", "classified", "naive"):
            raise ValidationError("select must be auto, classified or naive")
        if max_len is not None and (type(max_len) is not int or max_len < 0):
            raise ValidationError("max_len must be None or a nonnegative int")
        if select == "naive" and self.naive is None:
            raise ValidationError("select=naive requires a program compiled with a block-per-segment companion")
        if self.naive is not None and select != "classified":
            if select == "naive":
                chosen, selection = "naive", {"forced": True}
            else:
                try:
                    n, b = int(values.shape[0]), int(offsets.numel()) - 1
                except (AttributeError, IndexError, TypeError):
                    n = b = None   # the classified path's validation reports it
                chosen, selection = self.choose(n, b, max_len) if n is not None else ("classified", {})
            if chosen == "naive":
                return self.naive._run(values, offsets, out, stream, mode, debug_buffers=debug_buffers,
                                       mask=mask, concurrent=concurrent, selected="naive", selection=selection)
        else:
            selection = {"forced": True} if self.naive is not None else {}
        return self._run(values, offsets, out, stream, mode, debug_buffers=debug_buffers, mask=mask,
                         concurrent=concurrent, selected=self._schedule_name, selection=selection)

    def _run(self, values, offsets, out, stream, mode, *, debug_buffers, mask, concurrent,
             selected: str, selection: dict[str, Any]) -> ExecutionReport:
        if mode not in ("validated", "checked"):
            raise ValidationError("mode must be validated or checked")
        if self.spec.mask != (mask is not None):
            raise ValidationError("mask presence must match the compiled specialization")
        if mode == "checked" and "validation" not in self._metadata:
            raise RuntimeUnavailable("checked execution requires an artifact with the offsets-check-v1 device validation ABI")
        if debug_buffers is not None and (type(debug_buffers) is not dict or debug_buffers or not self.spec.debug_visits):
            raise ValidationError("debug_buffers requires an empty dict and a debug_visits specialization")
        torch = _torch()
        n, b = self._validate_tensors(values, offsets, out, torch, checked=mode == "checked", mask=mask)
        started = time.perf_counter()
        metadata = self._metadata
        input_buffers = [values, offsets, out] + ([mask] if mask is not None else [])
        arguments = [argument for tensor in input_buffers for argument in memref_arguments(tensor)]
        validation = metadata.get("validation") if mode == "checked" else None
        if b == 0 and validation is None:
            return ExecutionReport((), tuple(metadata.get("thresholds", ())),
                                   (), 0, self._digest, metadata.get("fallback_reason"), mode, (),
                                   (time.perf_counter() - started) * 1000, cache_hit=self.cache_hit, selected=selected, selection=selection,
                                   **self._report_fields)
        launched_names, launched_grids = [], []
        with torch.cuda.device(values.device):
            if values.device.index not in _sm89_devices:
                if tuple(torch.cuda.get_device_capability(values.device)) != (8, 9):
                    raise RuntimeUnavailable("compiled target sm_89 does not match the current device")
                _sm89_devices.add(values.device.index)
            active_stream = stream if stream is not None else torch.cuda.current_stream(values.device)
            if not isinstance(active_stream, torch.cuda.Stream) or active_stream.device != values.device:
                raise ValidationError("stream must be a torch.cuda.Stream on the values device")
            ordinal = values.device.index
            module_key = (ordinal, self._digest)
            with _module_lock:
                # Querying events is not permitted while a stream is captured
                # into a CUDA graph; retired modules wait for the next call.
                if not torch.cuda.is_current_stream_capturing():
                    _reap()
                if module_key not in _resident_modules:
                    try:
                        native = importlib.import_module("ragged._native")
                    except ImportError as exc:
                        raise RuntimeUnavailable("native Driver API extension is missing; build runtime/ with nanobind") from exc
                    _resident_modules[module_key] = native.DriverModule(self.artifact.cubin)
                if self._module_key != module_key:
                    if self._module_key is not None:
                        _release(self._module_key)
                    self._module_key = module_key
                    _holders[module_key] += 1
                module = _resident_modules[module_key]
                workers = _workers.get(module_key)
                if workers is None:
                    workers = _workers[module_key] = _worker_counts(module, metadata)
                _streams.setdefault(module_key, {})[active_stream.cuda_stream] = active_stream
            # Preflight all internal allocation sizes and launch dimensions
            # before allocating scratch or submitting any GPU work.
            layouts, grids, workspace_bytes = self._layout(metadata, False, n, b, ordinal, workers)
            validation_bytes, validation_host_ms = 0, None
            if validation is not None:
                _, validation_grids, validation_bytes = self._layout(validation, True, n, b, ordinal)
                if b and workspace_bytes > I64_MAX - validation_bytes:
                    raise ValidationError("combined validation and main workspace bytes overflow signed i64")
            if validation is not None:
                validation_started = time.perf_counter()
                with torch.cuda.stream(active_stream):
                    status = torch.empty((7,), dtype=torch.int64, device=values.device)
                    host_status = torch.empty((7,), dtype=torch.int64, device="cpu", pin_memory=True)
                    validation_arguments = arguments[:12] + memref_arguments(status)
                    for tensor in (values, offsets, status):
                        tensor.record_stream(active_stream)
                    for kernel, grid in zip(validation["kernels"], validation_grids):
                        module.launch(kernel["name"], grid, kernel["block"], 0,
                                      active_stream.cuda_stream, validation_arguments)
                        launched_names.append(kernel["name"])
                        launched_grids.append(grid)
                    host_status.copy_(status, non_blocking=True)
                # Only this stream is waited on; neither default/current stream
                # nor the whole device is implicitly synchronized.
                active_stream.synchronize()
                raise_for_status(host_status.tolist(), b=b)
                validation_host_ms = (time.perf_counter() - validation_started) * 1000
                if b == 0:
                    return ExecutionReport((), tuple(metadata.get("thresholds", ())), tuple(launched_grids),
                                           validation_bytes, self._digest, metadata.get("fallback_reason"),
                                           mode, tuple(launched_names), (time.perf_counter() - started) * 1000,
                                           cache_hit=self.cache_hit, selected=selected, selection=selection, validation_workspace_bytes=validation_bytes,
                                           validation_host_ms=validation_host_ms,**self._report_fields)
            plan = self._plan(module_key, module, layouts, grids, n, b, concurrent)
            # The caching allocator's reuse safety is tied to the allocation
            # stream. record_stream alone tracks lifetime, not pending writes on
            # another allocation stream. Allocate scratch on its consumer stream.
            # One allocation holds every workspace buffer at 256-byte offsets.
            scratch, base = None, 0
            if plan.nbytes:
                with torch.cuda.stream(active_stream) if stream is not None else nullcontext():
                    scratch = torch.empty((plan.nbytes,), dtype=torch.uint8, device=values.device)
                base = scratch.data_ptr()
            # Record use for the caching allocator when callers supply another stream.
            for tensor in input_buffers:
                tensor.record_stream(active_stream)
            module.run_plan(plan.id, active_stream.cuda_stream, arguments, base)
            launched_names.extend(plan.names)
            launched_grids.extend(plan.grids)
            if debug_buffers is not None:
                torch_dtypes = {"f32": torch.float32, "i32": torch.int32, "i64": torch.int64}
                debug_buffers.update({layout.name: scratch[offset:offset + layout.nbytes].view(
                    torch_dtypes[layout.dtype]).view(layout.shape) for layout, offset in zip(layouts, plan.offsets)})
        return ExecutionReport(tuple(metadata.get("classes", ())), tuple(metadata.get("thresholds", ())),
                               tuple(launched_grids), workspace_bytes + validation_bytes, self._digest,
                               metadata.get("fallback_reason"), mode, tuple(launched_names),
                               (time.perf_counter() - started) * 1000, cache_hit=self.cache_hit, selected=selected, selection=selection,
                               validation_workspace_bytes=validation_bytes, validation_host_ms=validation_host_ms,
                               streams=plan.streams, **self._report_fields)
