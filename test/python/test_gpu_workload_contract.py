"""Host validation of device-matrix inputs, without importing GPU fixtures."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from bench.distributions import offsets_from_lengths
from oracle import validate_offsets
from ragged import Hints, ProgramSpec

# Import workload_cases the way test/gpu's own tests do, as a module of that
# directory. Through "test.gpu" it fails wherever Python ships its standard
# test package, which takes the name "test". Appended, so test/gpu's
# test_selection.py cannot shadow this directory's.
sys.path.append(str(Path(__file__).resolve().parents[1] / "gpu"))
from workload_cases import (  # noqa: E402
    ANCHORS, BASE_ELEMENTS, LENGTH_SEED, MAX_ELEMENTS,
    SOURCES, SPLIT_FLOOR, classes_for, make_workload, save_failure,
)

CASES = [(source, layout) for source in SOURCES for layout in ("mixed", "split")]


@pytest.mark.parametrize("source,layout", CASES)
@pytest.mark.parametrize("offset_dtype", ["i32", "i64"])
def test_workload_offsets_determinism_and_memory_bound(source, layout, offset_dtype):
    workload = make_workload(source, layout)
    repeated = make_workload(source, layout)
    np.testing.assert_array_equal(workload.lengths, repeated.lengths)
    assert workload.metadata == repeated.metadata
    assert workload.lengths.dtype == np.int64 and workload.lengths.flags.c_contiguous
    assert np.all(workload.lengths >= 0)
    n = sum(map(int, workload.lengths))
    assert 0 < n <= MAX_ELEMENTS
    # At D=1024, even the largest f32 input is below 81 MiB. This bound
    # applies before allocating any dense value tensor, not after a GPU OOM.
    assert n * 1024 * 4 < 81 * 1024**2
    offsets = offsets_from_lengths(workload.lengths, offset_dtype)
    assert offsets.dtype == (np.int32 if offset_dtype == "i32" else np.int64)
    validate_offsets(offsets, n)
    assert workload.metadata["class_counts"] == np.bincount(classes_for(workload.lengths), minlength=3).tolist()


@pytest.mark.parametrize("source", SOURCES)
def test_mixed_has_every_actual_class_without_falsifying_hints(source):
    workload = make_workload(source, "mixed")
    np.testing.assert_array_equal(workload.lengths,
        np.concatenate([workload.base_lengths, np.array(ANCHORS)])[workload.permutation])
    assert sum(map(int, workload.base_lengths)) == BASE_ELEMENTS
    assert sum(map(int, workload.lengths)) == BASE_ELEMENTS + sum(ANCHORS)
    assert all(count > 0 for count in workload.metadata["class_counts"])
    assert workload.hints == {}  # No source claim that all lengths are long.
    assert "anchors" in workload.metadata["transformation"]


@pytest.mark.parametrize("source", SOURCES)
def test_static_split_hints_are_true_for_every_segment(source):
    workload = make_workload(source, "split")
    np.testing.assert_array_equal(workload.lengths,
        (workload.base_lengths + SPLIT_FLOOR)[workload.permutation])
    assert workload.metadata["class_counts"] == [0, 0, 4]
    hints = Hints(**workload.hints)
    assert hints.min_len == int(workload.lengths.min()) >= 4097
    assert hints.max_len == int(workload.lengths.max())
    assert np.all(workload.lengths >= hints.min_len) and np.all(workload.lengths <= hints.max_len)
    assert int(workload.lengths.sum()) == MAX_ELEMENTS


def test_small_split_sample_keeps_synthetic_distribution_variation():
    uniform = make_workload("uniform", "split").base_lengths
    assert len(np.unique(uniform)) == 1
    for source in ("long_tail", "bimodal", "burst"):
        assert len(np.unique(make_workload(source, "split").base_lengths)) > 1


def test_failure_artifact_preserves_exact_input_source_controls_and_seeds(tmp_path):
    workload = make_workload("long_tail", "mixed")
    offsets = offsets_from_lengths(workload.lengths, "i32")
    values = np.arange(int(offsets[-1]) * 2, dtype=np.float32).reshape(-1, 2)
    specs = [ProgramSpec("sum", "f32", 2, "i32", strategy="naive"),
             ProgramSpec("sum", "f32", 2, "i32", debug_visits=True)]
    programs = [{"spec": spec.to_dict(), "source": spec.to_mlir()} for spec in specs]
    save_failure(tmp_path, workload, values, offsets, programs, AssertionError("oracle mismatch"))
    with np.load(tmp_path / "case.npz", allow_pickle=False) as stored:
        for key, expected in [("values", values), ("offsets", offsets), ("lengths", workload.lengths),
                              ("source_lengths", workload.source_lengths), ("base_lengths", workload.base_lengths),
                              ("permutation", workload.permutation)]:
            np.testing.assert_array_equal(stored[key], expected)
    metadata = json.loads((tmp_path / "case.json").read_text())
    assert metadata["length_seed"] == LENGTH_SEED and metadata["value_seed"] == workload.metadata["value_seed"]
    assert metadata["values_sha256"] == hashlib.sha256(values.tobytes()).hexdigest()
    for saved, original in zip(metadata["programs"], programs):
        assert saved["spec"] == original["spec"]
        assert (tmp_path / saved["source_file"]).read_text() == original["source"]
        assert saved["source_sha256"] == hashlib.sha256(original["source"].encode()).hexdigest()
        assert "optimizations" in saved["spec"] and "regs_budget" in saved["spec"]
