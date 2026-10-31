"""Host validation of device-matrix inputs, without importing GPU fixtures."""
import hashlib
import json

import numpy as np
import pytest

from bench.distributions import offsets_from_lengths
from bench.public_histogram import SOURCE_SHA256, sample_histogram
from oracle import validate_offsets
from ragged import Hints, ProgramSpec
from test.gpu.workload_cases import (
    ANCHORS, BASE_ELEMENTS, LENGTH_SEED, MAX_ELEMENTS, PUBLIC_HISTOGRAM,
    SOURCES, SPLIT_FLOOR, classes_for, make_workload, save_failure,
)

CASES = [(source, layout) for source in SOURCES for layout in ("mixed", "split")]
CASES.append(("uci_online_retail", "native"))


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


def test_public_source_license_counts_and_unmodified_sample():
    provenance = json.loads(PUBLIC_HISTOGRAM.read_text())
    assert provenance["license"] == "CC-BY-4.0"
    assert provenance["source_sha256"] == SOURCE_SHA256
    assert sum(count for length, count in provenance["length_counts"]) == 25900
    assert sum(length * count for length, count in provenance["length_counts"]) == 541909
    assert max(length for length, _ in provenance["length_counts"]) == 1114
    workload = make_workload("uci_online_retail", "native")
    sample = sample_histogram(PUBLIC_HISTOGRAM, 64, seed=LENGTH_SEED, total_elements=None)
    np.testing.assert_array_equal(workload.source_lengths, sample)
    np.testing.assert_array_equal(workload.base_lengths, sample)
    np.testing.assert_array_equal(workload.lengths, sample)
    assert workload.metadata["normalization"] == "native_sample"
    assert workload.metadata["class_counts"] == [52, 12, 0]
    assert workload.metadata["source_provenance"]["histogram_sha256"] == hashlib.sha256(PUBLIC_HISTOGRAM.read_bytes()).hexdigest()


@pytest.mark.parametrize("layout,segments", [("mixed", 64), ("split", 4)])
def test_transformed_public_samples_retain_raw_and_scaled_lengths(layout, segments):
    workload = make_workload("uci_online_retail", layout)
    np.testing.assert_array_equal(workload.source_lengths,
        sample_histogram(PUBLIC_HISTOGRAM, segments, seed=LENGTH_SEED, total_elements=None))
    np.testing.assert_array_equal(workload.base_lengths,
        sample_histogram(PUBLIC_HISTOGRAM, segments, seed=LENGTH_SEED, total_elements=BASE_ELEMENTS))
    assert workload.metadata["normalization"] == "fixed_elements_before_mapping_transform"
    assert workload.metadata["source_lengths"] != workload.metadata["base_lengths"]


def test_small_split_sample_keeps_synthetic_distribution_variation():
    uniform = make_workload("uniform", "split").base_lengths
    assert len(np.unique(uniform)) == 1
    for source in ("long_tail", "bimodal", "burst"):
        assert len(np.unique(make_workload(source, "split").base_lengths)) > 1


def test_failure_artifact_preserves_exact_input_source_controls_and_seeds(tmp_path):
    workload = make_workload("uci_online_retail", "native")
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
