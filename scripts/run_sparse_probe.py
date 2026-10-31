#!/usr/bin/env python3
"""Run the sparse_tensor CSR semantic probe on CPU and retain commands/results.

This is a six-segment D=1 feasibility experiment, not GPU correctness or a
performance comparison. Printed finite values have six significant digits;
the numeric comparison accommodates that output precision only.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from oracle import segment_max, segment_softmax  # noqa: E402


def parse_numbers(text: str) -> np.ndarray:
    return np.array([float(part.strip()) for part in text.split(",")], dtype=np.float64)


def check_output(stdout: str) -> dict:
    offsets = np.array([0, 2, 2, 4, 6, 8, 10], dtype=np.int64)
    values = np.array([-2, -4, np.nan, 1, -np.inf, -np.inf, 0, 2, np.inf, 1], dtype=np.float64).reshape(-1, 1)
    lines = stdout.splitlines()
    maxima = parse_numbers(lines[0].strip().removeprefix("(").removesuffix(")"))
    stored_values = re.findall(r"^values : \( (.*) \)$", stdout, flags=re.MULTILINE)
    positions = re.findall(r"^pos\[1\] : \( (.*) \)$", stdout, flags=re.MULTILINE)
    coordinates = re.findall(r"^crd\[1\] : \( (.*) \)$", stdout, flags=re.MULTILINE)
    if len(stored_values) != 2 or len(positions) != 2 or len(coordinates) != 2:
        raise AssertionError("probe did not print both complete CSR tensors")
    for actual in positions:
        np.testing.assert_array_equal(parse_numbers(actual), offsets)
    for actual in coordinates:
        np.testing.assert_array_equal(parse_numbers(actual), [0, 1] * 5)
    np.testing.assert_array_equal(parse_numbers(stored_values[0]), values[:, 0])
    actual = parse_numbers(stored_values[1])
    expected = segment_softmax(values, offsets)[:, 0]
    np.testing.assert_array_equal(maxima, segment_max(values, offsets)[:, 0])
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=5e-7, equal_nan=True)
    finite = np.isfinite(expected)
    return {
        "passed": True,
        "segments": 6,
        "stored_values": 10,
        "feature_dim": 1,
        "arithmetic": "f64 CPU",
        "max_abs_printed_softmax_error": float(np.max(np.abs(actual[finite] - expected[finite]))),
        "covered": ["absence_is_not_zero", "empty_row_max_negative_infinity",
                    "NaN_propagation", "all_negative_infinity_softmax_NaN",
                    "explicit_stored_zero_preserved", "positive_infinity_softmax_NaN"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llvm-root", type=Path, default=ROOT / ".deps/LLVM-23.1.3-Linux-X64")
    parser.add_argument("--output", type=Path, default=ROOT / "evidence/sparse")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = ROOT / "test/sparse/sparse-segment-semantics.mlir"
    optimizer, runner = args.llvm_root / "bin/mlir-opt", args.llvm_root / "bin/mlir-runner"
    libraries = [args.llvm_root / "lib" / name for name in
                 ("libmlir_c_runner_utils.so.23.1", "libmlir_runner_utils.so.23.1")]
    version = subprocess.run([str(optimizer), "--version"], capture_output=True, text=True, check=True).stdout
    result = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "scope": "CPU sparse CSR feasibility only; D=1; no GPU or timing claim",
        "source": str(source.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "tool_version": version.strip(),
        "variants": [],
    }
    for variant, options in (("runtime", "enable-runtime-library=true"),
                             ("direct", "enable-runtime-library=false enable-buffer-initialization=true")):
        lowered = args.output / f"{variant}.llvm.mlir"
        compile_command = [str(optimizer), str(source), f"--sparsifier={options}", "-o", str(lowered)]
        compiled = subprocess.run(compile_command, capture_output=True, text=True, timeout=60)
        (args.output / f"{variant}.compile.stderr.txt").write_text(compiled.stderr)
        record = {"variant": variant, "compile_command": compile_command, "compile_returncode": compiled.returncode}
        if compiled.returncode == 0:
            run_command = [str(runner), str(lowered), "-e", "main", "-entry-point-result=void",
                           "-shared-libs=" + ",".join(map(str, libraries))]
            executed = subprocess.run(run_command, capture_output=True, text=True, timeout=60)
            (args.output / f"{variant}.stdout.txt").write_text(executed.stdout)
            (args.output / f"{variant}.stderr.txt").write_text(executed.stderr)
            record.update(run_command=run_command, run_returncode=executed.returncode)
            if executed.returncode == 0:
                try:
                    record["oracle"] = check_output(executed.stdout)
                except (ValueError, AssertionError, IndexError) as exc:
                    record["oracle"] = {"passed": False, "error": str(exc)}
        result["variants"].append(record)
    passed = all(record.get("oracle", {}).get("passed", False) for record in result["variants"])
    result["passed"] = passed
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"passed": passed, "variants": len(result["variants"]), "evidence": str(args.output / "results.json")}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
