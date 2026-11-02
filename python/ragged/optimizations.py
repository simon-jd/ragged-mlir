"""Immutable optimization requests and their compiled metadata contract."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import struct
from typing import TYPE_CHECKING, Any

from .errors import CacheMismatch, ValidationError

if TYPE_CHECKING:
    from .model import ProgramSpec

BOOL_OPTIONS = ("classification", "online_fusion", "warp_mapping", "split_mapping", "shuffle", "pointwise_fusion",
                "row_pipelining", "vectorize")
VECTOR_WIDTHS = (1, 2, 4, 8)
FEATURE_TILES = (1, 2, 4, 8, 16, 32)


@dataclass(frozen=True)
class Optimizations:
    classification: bool = True
    online_fusion: bool = True
    warp_mapping: bool = True
    split_mapping: bool = True
    shuffle: bool = True
    pointwise_fusion: bool = True
    row_pipelining: bool = True
    vectorize: bool = True
    feature_tile: int | None = None

    def __post_init__(self):
        for name in BOOL_OPTIONS:
            if type(getattr(self, name)) is not bool:
                raise ValidationError(f"optimization {name} must be bool")
        if self.feature_tile is not None and (type(self.feature_tile) is not int or self.feature_tile not in FEATURE_TILES):
            raise ValidationError("feature_tile must be None or a power of two in [1,32]")

    def to_dict(self):
        return asdict(self)

    def geometry(self, d: int, dtype: str = "f32", smem_per_cta: int = 49152) -> tuple[int, int]:
        """(d_tile, d_vector) as ragged-plan resolves them: each thread owns
        d_vector adjacent features (a power of two dividing D, at most 16 bytes,
        whose 128-thread tree fits smem_per_cta) and d_tile threads span a
        feature tile of min(32, next power of two of D) features. An explicit
        feature_tile keeps one feature per thread."""
        element = 2 if dtype == "bf16" else 4
        vec = 1
        if self.vectorize and self.feature_tile is None:
            while 2 * vec * element <= 16 and d % (2 * vec) == 0 and 128 * 2 * vec * 4 <= smem_per_cta:
                vec *= 2
        if self.feature_tile is not None:
            return self.feature_tile, vec
        features = min(32, 1 << (d - 1).bit_length())
        vec = min(vec, features)
        return features // vec, vec

    def resolved_tile(self, d: int, dtype: str = "f32", smem_per_cta: int = 49152) -> int:
        return self.geometry(d, dtype, smem_per_cta)[0]


def computation_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """View excluding the optional materialization prefix and logits buffer."""
    if not metadata.get("pointwise_materialized", False):
        return metadata
    return metadata | {"kernels": metadata["kernels"][1:], "workspace": metadata["workspace"][:-1]}


def class_fusion(metadata: dict[str, Any]) -> list[str] | None:
    """Softmax statistics schedule per class, aligned with ``classes``; None for reductions.

    ``fusion`` summarizes it: online or three_pass when every class agrees,
    otherwise mixed. A manifest without ``class_fusion`` is uniform.
    """
    fusion, classes = metadata.get("fusion"), metadata.get("classes")
    if fusion == "not_applicable":
        if "class_fusion" in metadata:
            raise CacheMismatch("reductions have no per-class fusion schedule")
        return None
    if fusion not in ("online", "three_pass", "mixed") or not isinstance(classes, list) or not classes:
        raise CacheMismatch("softmax requires an online, three_pass or mixed schedule over its classes")
    schedule = metadata.get("class_fusion")
    if schedule is None:
        if fusion == "mixed":
            raise CacheMismatch("mixed fusion requires class_fusion")
        return [fusion] * len(classes)
    if (not isinstance(schedule, list) or len(schedule) != len(classes)
            or any(entry not in ("online", "three_pass") for entry in schedule)):
        raise CacheMismatch("class_fusion must name online or three_pass for each class")
    if fusion != (schedule[0] if len(set(schedule)) == 1 else "mixed"):
        raise CacheMismatch("fusion does not summarize class_fusion")
    return list(schedule)


def validate_optimization_metadata(metadata: dict[str, Any]) -> Optimizations | None:
    if "optimizations" not in metadata:
        if "pointwise_materialized" in metadata or "d_tile" in metadata:
            raise CacheMismatch("optimization results require explicit requested/effective metadata")
        return None
    options = metadata["optimizations"]
    if not isinstance(options, dict) or set(options) != {"requested", "effective"}:
        raise CacheMismatch("optimizations must contain requested and effective objects")
    requested, effective = options["requested"], options["effective"]
    if not isinstance(requested, dict) or set(requested) != set(BOOL_OPTIONS) | {"feature_tile"}:
        raise CacheMismatch("requested optimization fields differ")
    try:
        config = Optimizations(**requested)
    except (TypeError, ValidationError) as exc:
        raise CacheMismatch(f"invalid requested optimizations: {exc}") from exc
    if (not isinstance(effective, dict) or set(effective) != set(BOOL_OPTIONS)
            or any(type(value) is not bool for value in effective.values())):
        raise CacheMismatch("effective optimizations must contain one boolean field per control")
    tile, materialized = metadata.get("d_tile"), metadata.get("pointwise_materialized")
    vector = metadata.get("d_vector", 1)
    if type(tile) is not int or tile not in FEATURE_TILES or type(materialized) is not bool:
        raise CacheMismatch("optimization results require a supported d_tile and boolean pointwise_materialized")
    if type(vector) is not int or vector not in VECTOR_WIDTHS:
        raise CacheMismatch("d_vector must be a power of two in [1,8]")
    if config.feature_tile is not None and tile != config.feature_tile:
        raise CacheMismatch("actual feature tile differs from the requested explicit tile")
    classes = metadata.get("classes", [])
    expected = {"classification": metadata.get("dispatch_abi") == "classified-v3",
                "online_fusion": metadata.get("fusion") in ("online", "mixed"), "warp_mapping": "warp" in classes,
                "split_mapping": "split" in classes, "shuffle": "warp" in classes and config.shuffle,
                "row_pipelining": config.row_pipelining, "vectorize": vector > 1}
    if any(effective[name] != value for name, value in expected.items()):
        raise CacheMismatch("effective optimizations disagree with actual dispatch/classes/fusion")
    if any(effective[name] and not requested[name] for name in BOOL_OPTIONS):
        raise CacheMismatch("a disabled optimization cannot be reported as effective")
    if not config.classification and (expected["classification"] or classes != ["cta"]):
        raise CacheMismatch("classification disabled requires direct CTA execution")
    if materialized and (config.pointwise_fusion or effective["pointwise_fusion"]):
        raise CacheMismatch("materialized pointwise work cannot also be fused")
    if metadata.get("mask", False) and not (materialized or effective["pointwise_fusion"]):
        raise CacheMismatch("mask arithmetic must be fused or materialized")
    if materialized:
        kernels, workspace = metadata.get("kernels"), metadata.get("workspace")
        if not isinstance(kernels, list) or len(kernels) < 2 or not isinstance(workspace, list) or not workspace:
            raise CacheMismatch("materialized pointwise work requires a prefix kernel and logits workspace")
        first, logits = kernels[0], workspace[-1]
        if (not isinstance(first, dict) or first.get("role") != "materialize" or first.get("grid") != [256,1,1]
                or first.get("block") != [128,1,1] or "segment_class" in first
                or first.get("shared_bytes",0) != 0 or first.get("static_shared_bytes",0) != 0):
            raise CacheMismatch("pointwise materialization requires the fixed first-kernel launch ABI")
        if (not isinstance(logits, dict) or set(logits) != {"name","dtype","shape"}
                or logits.get("name") != "logits" or logits.get("dtype") != "f32"
                or not isinstance(logits.get("shape"),list) or len(logits["shape"]) != 2
                or logits["shape"][0] != "N" or type(logits["shape"][1]) is not int
                or not 1 <= logits["shape"][1] <= 1024):
            raise CacheMismatch("materialized logits must be the final f32[N,D] workspace")
    elif any(entry.get("name") == "logits" for entry in metadata.get("workspace",[]) if isinstance(entry,dict)):
        raise CacheMismatch("logits workspace is only defined for pointwise materialization")
    if any(kernel.get("role") == "materialize" for kernel in computation_metadata(metadata).get("kernels",[]) if isinstance(kernel,dict)):
        raise CacheMismatch("materialize may only appear as the declared first kernel")
    return config


def validate_optimization_spec(spec: ProgramSpec, metadata: dict[str, Any]) -> None:
    if "optimizations" not in metadata:
        if spec.optimizations != Optimizations():
            raise CacheMismatch("nondefault optimization requests require explicit compiler metadata")
        return
    if metadata["optimizations"]["requested"] != spec.optimizations.to_dict():
        raise CacheMismatch("requested optimizations differ from the source specialization")
    tile, vector = spec.optimizations.geometry(spec.feature_dim, spec.dtype, spec.smem_per_cta)
    if metadata["d_tile"] != tile:
        raise CacheMismatch("actual feature tile differs from source request/heuristic")
    if metadata.get("d_vector", 1) != vector:
        raise CacheMismatch("actual features per thread differ from source request/heuristic")
    pointwise = struct.unpack("f",struct.pack("f",spec.scale))[0] != 1.0 or spec.mask
    materialized = pointwise and not spec.optimizations.pointwise_fusion
    if (metadata["pointwise_materialized"] != materialized
            or metadata["optimizations"]["effective"]["pointwise_fusion"] != (pointwise and not materialized)):
        raise CacheMismatch("pointwise fusion/materialization differs from source arithmetic and requests")
    if materialized and metadata["workspace"][-1]["shape"] != ["N",spec.feature_dim]:
        raise CacheMismatch("materialized logits feature width differs from source")
    core = computation_metadata(metadata)
    workers = core["kernels"][2:] if core.get("dispatch_abi") == "classified-v3" else core["kernels"]
    span = tile * vector
    if any(kernel["grid"][1] != (spec.feature_dim + span - 1) // span for kernel in workers):
        raise CacheMismatch("worker grids must cover exactly ceil(D/(d_tile*d_vector)) feature tiles")
    fusion = core.get("fusion")
    if ((spec.op == "softmax" and fusion not in ("online","three_pass","mixed"))
            or (spec.op != "softmax" and fusion != "not_applicable")):
        raise CacheMismatch("fusion schedule does not implement the source operation")
    class_fusion(core)
    if spec.force_fusion and fusion != "online":
        raise CacheMismatch("forced online fusion is absent from the compiled schedule")
    if core.get("dispatch_abi") != "classified-v3":
        classes = core.get("classes")
        if classes not in (["warp"],["cta"]):
            raise CacheMismatch("direct execution requires exactly one warp or CTA class")
        block = [32 if classes == ["warp"] else 128,1,1]
        if any(kernel["grid"][0] != "B" or kernel["grid"][2] != 1 or kernel["block"] != block
               or kernel.get("shared_bytes",0) != 0 for kernel in workers):
            raise CacheMismatch("direct worker launch dimensions disagree with the selected mapping")
        three_pass = spec.op == "softmax" and fusion == "three_pass"
        expected_workspace = [{"name":name,"shape":["B",spec.feature_dim],"dtype":"f32"}
                              for name in ("max_stats","sum_stats")] if three_pass else []
        if len(workers) != (3 if three_pass else 1) or core.get("workspace",[]) != expected_workspace:
            raise CacheMismatch("direct kernel count/workspace disagrees with the computation schedule")
    if spec.strategy == "naive" and (core.get("classes") != ["cta"]
            or (core.get("dispatch_abi") == "classified-v3") != spec.debug_visits):
        raise CacheMismatch("naive strategy requires CTA execution; only debug visits use classified dispatch")
