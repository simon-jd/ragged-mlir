"""Deterministic segment lengths for both workload-normalization conventions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


DISTRIBUTIONS = ("uniform", "long_tail", "bimodal", "burst")


def make_lengths(distribution: str, segments: int, *, seed: int = 0,
                 total_elements: int | None = None, mean_length: int = 64) -> np.ndarray:
    if distribution not in DISTRIBUTIONS:
        raise ValueError(f"distribution must be one of {DISTRIBUTIONS}")
    for name, value in (("segments", segments), ("seed", seed), ("mean_length", mean_length)):
        if type(value) is not int or not 0 <= value <= 2**63 - 1:
            raise ValueError(f"{name} must be a nonnegative signed-64 integer")
    if total_elements is not None and (type(total_elements) is not int or not 0 <= total_elements <= 2**63 - 1):
        raise ValueError("total_elements must be a nonnegative signed-64 integer")
    if segments == 0:
        if total_elements not in (None, 0):
            raise ValueError("nonzero elements cannot belong to zero segments")
        return np.zeros(0, dtype=np.int64)
    rng = np.random.default_rng(seed)
    if distribution == "uniform":
        weights = np.ones(segments)
    elif distribution == "long_tail":
        weights = np.minimum(rng.pareto(1.3, segments) + 1, 100000.0)
    elif distribution == "bimodal":
        weights = np.where(rng.random(segments) < 0.9, 4.0, 1024.0)
    else:
        weights = np.ones(segments)
        width = max(1, segments // 100)
        start = int(rng.integers(segments - width + 1))
        weights[start:start + width] = 16384
    if total_elements is None:
        # Fixed segment count: same typical scale; realized N is recorded and
        # intentionally not forced equal across the different distributions.
        scaled = weights * mean_length
        if np.any(scaled >= float(2**63)):
            raise ValueError("generated length exceeds i64")
        result = np.floor(scaled).astype(np.int64)
        if sum(map(int, result)) > 2**63 - 1:
            raise ValueError("total generated elements exceed i64")
        return result
    # Convert sampled ratios to integer weights, then allocate with integer
    # arithmetic so N remains exact even beyond float64's exact integer range.
    units = np.maximum(1, np.rint(weights * 1024).astype(np.int64))
    denominator = sum(map(int, units))
    numerators = [int(weight) * total_elements for weight in units]
    lengths = [numerator // denominator for numerator in numerators]
    remainder = total_elements - sum(lengths)
    order = sorted(range(segments), key=lambda i: (-(numerators[i] % denominator), i))
    for i in order[:remainder]:
        lengths[i] += 1
    return np.asarray(lengths, dtype=np.int64)


def offsets_from_lengths(lengths: np.ndarray, dtype: str = "i64") -> np.ndarray:
    if dtype not in ("i32", "i64"):
        raise ValueError("offsets dtype must be i32 or i64")
    if lengths.ndim != 1 or lengths.dtype.kind not in "iu" or np.any(lengths < 0):
        raise ValueError("lengths must be a nonnegative integer vector")
    total = sum(map(int, lengths))
    target = np.int32 if dtype == "i32" else np.int64
    if total > np.iinfo(target).max:
        raise ValueError(f"total elements do not fit {dtype}")
    return np.concatenate([np.zeros(1, dtype=target), np.cumsum(lengths, dtype=target)])


def describe(lengths: np.ndarray) -> dict:
    # Histogram uses explicit right-exclusive edges, with the last bucket
    # including all longer segments. Preserve exact lengths in the NPZ too.
    edges = [0, 1, 2, 8, 32, 33, 128, 1024, 4096, 4097, 16384]
    counts = [int(np.count_nonzero((lengths >= lo) & (lengths < hi)))
              for lo, hi in zip(edges, edges[1:])]
    counts.append(int(np.count_nonzero(lengths >= edges[-1])))
    return {"segments": int(lengths.size), "elements": sum(map(int, lengths)),
            "min_length": int(lengths.min()) if lengths.size else None,
            "max_length": int(lengths.max()) if lengths.size else None,
            "histogram_lower_bounds": edges, "histogram_counts": counts,
            "lengths_sha256": hashlib.sha256(lengths.astype("<i8").tobytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distribution", choices=DISTRIBUTIONS, required=True)
    parser.add_argument("--segments", type=int, required=True)
    parser.add_argument("--total-elements", type=int)
    parser.add_argument("--mean-length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lengths = make_lengths(args.distribution, args.segments, seed=args.seed,
                           total_elements=args.total_elements, mean_length=args.mean_length)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, lengths=lengths, offsets=offsets_from_lengths(lengths))
    metadata = describe(lengths) | {"seed": args.seed, "distribution": args.distribution,
                                  "normalization": "fixed_elements" if args.total_elements is not None else "fixed_segments",
                                  "mean_length_parameter": args.mean_length}
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
