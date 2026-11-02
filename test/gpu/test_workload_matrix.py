"""Length-distribution and mapping cross-product against the same float64 oracle.

384 cells compare naive with mixed classified or legal static-split inputs from
four synthetic length distributions. Injected anchors and shifted split
lengths are explicitly labeled transformations.
No CUDA device means a reported skip, not numerical validation.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from bench.distributions import offsets_from_lengths
from oracle import assert_matches
from ragged import Hints, ProgramSpec
from test_correctness import execute, gpu_environment, prepared
from workload_cases import ROOT, SOURCES, VALUE_SEED, classes_for, make_workload, save_failure


def _execute_mapped(environment, spec, values, offsets, lengths):
    torch, compile_spec, _, _ = environment
    program = compile_spec(spec)
    dtype = torch.float32 if spec.dtype == "f32" else torch.bfloat16
    device_values = torch.from_numpy(values).to(device="cuda", dtype=dtype)
    device_offsets = torch.from_numpy(offsets).cuda()
    rows = len(values) if spec.op == "softmax" else len(lengths)
    guard = torch.full((rows + 2, spec.feature_dim), -64, device="cuda", dtype=dtype)
    output = guard[1:rows + 1]
    workspace = {}
    report = program.run(device_values, device_offsets, output, debug_buffers=workspace)
    torch.cuda.current_stream().synchronize()
    assert output.dtype == dtype
    assert torch.all(guard[0] == -64).item() and torch.all(guard[-1] == -64).item()
    assert_matches(output.float().cpu().numpy(), values, offsets,
                   operation=spec.op, input_dtype=spec.dtype)
    # report.classes lists compiled candidates. Read the actual queues only in
    # this test, after execution, to establish that the intended classes ran.
    expected = classes_for(lengths)
    counts = workspace["class_counts"].cpu().numpy()
    ids = workspace["segment_ids"].cpu().numpy()
    np.testing.assert_array_equal(counts, np.bincount(expected, minlength=3))
    for ordinal, count in enumerate(counts):
        np.testing.assert_array_equal(np.sort(ids[ordinal, :int(count)]), np.flatnonzero(expected == ordinal))
    np.testing.assert_array_equal(workspace["visits"].cpu().numpy(), np.ones(len(lengths), dtype=np.int32))
    assert report.workspace_bytes == sum(t.numel() * t.element_size() for t in workspace.values())
    return report


def _compare(gpu_environment, record_property, op, dtype, d, offset_dtype, source, layout):
    workload = make_workload(source, layout)
    offsets = offsets_from_lengths(workload.lengths, offset_dtype)
    # Generate directly into f32 to avoid a transient double-sized f64 input.
    values = np.random.default_rng(VALUE_SEED).standard_normal((int(offsets[-1]), d), dtype=np.float32)
    values = prepared(values, dtype)
    naive = ProgramSpec(op, dtype, d, offset_dtype, strategy="naive")
    mapped = ProgramSpec(op, dtype, d, offset_dtype, hints=Hints(**workload.hints), debug_visits=True)
    record_property("workload", json.dumps(workload.metadata, sort_keys=True))
    try:
        _, baseline = execute(gpu_environment, naive, values, offsets)
        assert baseline.classes == ("cta",)
        report = _execute_mapped(gpu_environment, mapped, values, offsets, workload.lengths)
        assert report.classes == (("split",) if layout == "split" else ("warp", "cta", "split"))
    except Exception as error:
        root = Path(os.environ.get("RAGGED_GPU_FAILURE_DIR", str(ROOT / "evidence/gpu-failures/workload-matrix")))
        name = f"{source}-{layout}-{op}-{dtype}-d{d}-{offset_dtype}"
        destination = save_failure(root / name, workload, values, offsets,
            [{"spec": spec.to_dict(), "source": spec.to_mlir()} for spec in (naive, mapped)], error)
        error.add_note(f"Full workload, seeds, specifications, and source hashes retained in {destination}")
        raise


@pytest.mark.parametrize("op", ["sum", "max", "softmax"])
@pytest.mark.parametrize("dtype", ["f32", "bf16"])
@pytest.mark.parametrize("d", [1, 16, 128, 1024])
@pytest.mark.parametrize("offset_dtype", ["i32", "i64"])
@pytest.mark.parametrize("source", SOURCES)
@pytest.mark.parametrize("layout", ["mixed", "split"])
def test_distribution_mapping_matrix(gpu_environment, record_property, op, dtype, d, offset_dtype, source, layout):
    _compare(gpu_environment, record_property, op, dtype, d, offset_dtype, source, layout)

