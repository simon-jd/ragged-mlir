"""Deterministic, bounded test workloads; no Torch, compiler, or GPU imports."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from bench.distributions import DISTRIBUTIONS, describe, make_lengths

ROOT = Path(__file__).resolve().parents[2]
SOURCES = DISTRIBUTIONS
LENGTH_SEED = 711
VALUE_SEED = 981
PERMUTATION_SEED = 17701
BASE_ELEMENTS = 4096
ANCHORS = (7, 65, 4097)
SPLIT_FLOOR = 4097
MAX_ELEMENTS = 4 * SPLIT_FLOOR + BASE_ELEMENTS


def classes_for(lengths: np.ndarray) -> np.ndarray:
    return np.where(lengths <= 32, 0, np.where(lengths <= 4096, 1, 2))


@dataclass
class Workload:
    lengths: np.ndarray
    source_lengths: np.ndarray
    base_lengths: np.ndarray
    permutation: np.ndarray
    metadata: dict

    @property
    def hints(self) -> dict:
        # Only the explicitly shifted all-long workload may prune to split.
        if self.metadata["layout"] == "split":
            return {"min_len": int(self.lengths.min()), "max_len": int(self.lengths.max())}
        return {}


def make_workload(source: str, layout: str) -> Workload:
    if source not in SOURCES or layout not in ("mixed", "split"):
        raise ValueError("unsupported workload source/layout")
    segments = 4 if layout == "split" else 64
    raw = make_lengths(source, segments, seed=LENGTH_SEED)
    base = make_lengths(source, segments, seed=LENGTH_SEED, total_elements=BASE_ELEMENTS)
    provenance = {"generator": "bench.distributions.make_lengths", "mean_length": 64}
    if layout == "mixed":
        unshuffled = np.concatenate((base, np.asarray(ANCHORS, dtype=np.int64)))
        transformation = "fixed-N sample plus explicit warp/CTA/split anchors, then segment permutation"
    else:
        unshuffled = base + SPLIT_FLOOR
        transformation = "fixed-N sample plus 4097 rows per segment, then segment permutation"
    permutation = np.random.default_rng(PERMUTATION_SEED).permutation(len(unshuffled))
    lengths = np.ascontiguousarray(unshuffled[permutation], dtype=np.int64)
    counts = np.bincount(classes_for(lengths), minlength=3)
    metadata = describe(lengths) | {
        "source": source, "layout": layout, "source_provenance": provenance,
        "length_seed": LENGTH_SEED, "value_seed": VALUE_SEED,
        "permutation_seed": PERMUTATION_SEED,
        "normalization": "fixed_elements_before_mapping_transform",
        "base_elements": int(base.sum()), "transformation": transformation,
        "class_counts": counts.tolist(), "source_lengths": raw.tolist(),
        "base_lengths": base.tolist(), "permutation": permutation.tolist(),
    }
    if int(lengths.sum()) > MAX_ELEMENTS:
        raise AssertionError("test workload exceeded its explicit memory bound")
    return Workload(lengths, raw, base, permutation, metadata)


def save_failure(destination: Path, workload: Workload, values: np.ndarray,
                 offsets: np.ndarray, programs: list[dict], error: Exception) -> Path:
    """Keep full numeric input, source IR/hash, immutable spec, and all seeds."""
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination / "case.npz", values=values, offsets=offsets,
                        lengths=workload.lengths, source_lengths=workload.source_lengths,
                        base_lengths=workload.base_lengths, permutation=workload.permutation)
    sources = []
    for ordinal, program in enumerate(programs):
        text = program["source"]
        path = destination / f"program-{ordinal}.mlir"
        path.write_text(text)
        sources.append({"spec": program["spec"], "source_file": path.name,
                        "source_sha256": hashlib.sha256(text.encode()).hexdigest()})
    metadata = workload.metadata | {
        "error": repr(error), "programs": sources,
        "values_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
        "offsets_sha256": hashlib.sha256(offsets.tobytes()).hexdigest(),
        "values_shape": list(values.shape), "values_dtype": str(values.dtype),
        "offsets_dtype": str(offsets.dtype),
    }
    (destination / "case.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return destination
