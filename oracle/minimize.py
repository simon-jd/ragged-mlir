"""Reduce a valid packed numeric failure, preserving an external failing predicate.

CLI: python -m oracle.minimize input.npz output.npz -- checker --case {case}
The checker must exit 1 for the same numerical failure, 0 for no failure, and
another status for infrastructure errors. A timeout/error aborts minimization.
An output is only written after rechecking that it still reproduces the failure.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import tempfile
from typing import Callable

import numpy as np

from .reference import validate_offsets


Predicate = Callable[[np.ndarray, np.ndarray], bool]


def _pack(segments: list[np.ndarray], d: int, values_dtype, offsets_dtype):
    values = np.concatenate(segments, axis=0) if segments else np.empty((0, d), dtype=values_dtype)
    offsets = np.array([0, *np.cumsum([len(part) for part in segments], dtype=np.int64)], dtype=offsets_dtype)
    return np.ascontiguousarray(values), offsets


def minimize_case(values: np.ndarray, offsets: np.ndarray, fails: Predicate) -> tuple[np.ndarray, np.ndarray]:
    """Deterministically reach a deletion-minimal segment/row/feature case.

    This is a structural reducer, not proof of global minimality. It checks all
    remaining individual deletions after chunk reduction, then repeats until
    stable because removing a feature may enable further row deletions.
    """
    values = np.ascontiguousarray(values)
    offsets = validate_offsets(offsets, len(values)).copy()
    if values.ndim != 2 or not 1 <= values.shape[1] <= 1024 or values.dtype.kind != "f":
        raise ValueError("values must be floating rank 2 with D in [1, 1024]")
    if not fails(values, offsets):
        raise ValueError("input does not reproduce the requested failure")
    while True:
        before = (len(offsets), *values.shape)
        # Remove whole groups of segments first, then rows within each segment.
        parts = [values[int(a):int(b)] for a, b in zip(offsets[:-1], offsets[1:])]
        granularity = max(1, len(parts) // 2)
        while granularity:
            index = 0
            while index < len(parts):
                candidate_parts = parts[:index] + parts[index+granularity:]
                candidate = _pack(candidate_parts, values.shape[1], values.dtype, offsets.dtype)
                if fails(*candidate):
                    parts, (values, offsets) = candidate_parts, candidate
                else:
                    index += granularity
            granularity //= 2
        for segment in range(len(parts)):
            granularity = max(1, len(parts[segment]) // 2)
            while granularity:
                index = 0
                while index < len(parts[segment]):
                    part = parts[segment]
                    candidate_parts = parts.copy()
                    candidate_parts[segment] = np.concatenate((part[:index], part[index+granularity:]))
                    candidate = _pack(candidate_parts, values.shape[1], values.dtype, offsets.dtype)
                    if fails(*candidate):
                        parts, (values, offsets) = candidate_parts, candidate
                    else:
                        index += granularity
                granularity //= 2
        feature = 0
        while values.shape[1] > 1 and feature < values.shape[1]:
            candidate_values = np.ascontiguousarray(np.delete(values, feature, axis=1))
            if fails(candidate_values, offsets):
                values = candidate_values
            else:
                feature += 1
        if before == (len(offsets), *values.shape):
            break
    if not fails(values, offsets):
        raise RuntimeError("failure predicate became nondeterministic; refusing output")
    return values, offsets


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or not any("{case}" in part for part in command):
        parser.error("checker command must contain a {case} path placeholder")
    with np.load(args.input, allow_pickle=False) as archive:
        values, offsets = archive["values"], archive["offsets"]
    with tempfile.TemporaryDirectory(prefix="ragged-minimize-") as directory:
        path = Path(directory) / "candidate.npz"

        def fails(candidate_values, candidate_offsets):
            np.savez(path, values=candidate_values, offsets=candidate_offsets)
            result = subprocess.run([part.replace("{case}", str(path)) for part in command],
                                    timeout=args.timeout, check=False)
            if result.returncode not in (0, 1):
                raise RuntimeError(f"checker infrastructure error: exit {result.returncode}")
            return result.returncode == 1

        values, offsets = minimize_case(values, offsets, fails)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as output:
        np.savez(output, values=values, offsets=offsets)
    print(f"reproducer: B={len(offsets)-1}, N={len(values)}, D={values.shape[1]} -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
