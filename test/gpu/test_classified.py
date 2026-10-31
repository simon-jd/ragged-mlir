"""Independent device tests for classified/persistent and split mappings.

These tests execute generated cubins on sm_89. A missing GPU is an explicit
skip; compile failures on a present GPU must fail. Classification and split
paths use the same float64 oracle and fixed bounds as the naive path.
"""

from __future__ import annotations

import numpy as np
import pytest

from bench.distributions import offsets_from_lengths
from oracle import assert_matches
from ragged import Hints, ProgramSpec
from test_correctness import execute, gpu_environment


@pytest.mark.parametrize("op", ["sum", "max", "softmax"])
@pytest.mark.parametrize("dtype", ["f32", "bf16"])
@pytest.mark.parametrize("offset_dtype", ["i32", "i64"])
@pytest.mark.parametrize("d", [1, 7, 128, 1024])
def test_classified_threshold_boundaries(gpu_environment, op, dtype, offset_dtype, d):
    # Original order intentionally differs from class order. Empty segments
    # occur before, between and after nonempty segments, including a split tail.
    lengths = np.array([4097, 0, 1, 33, 8193, 32, 4096, 31, 4095, 0], dtype=np.int64)
    offsets = offsets_from_lengths(lengths, offset_dtype)
    values = np.random.default_rng(5001).normal(size=(int(offsets[-1]), d)).astype(np.float32)
    _, report = execute(gpu_environment, ProgramSpec(op, dtype, d, offset_dtype), values, offsets)
    assert report.classes == ("warp", "cta", "split")
    assert report.workspace_bytes > 0


@pytest.mark.parametrize("op", ["sum", "max", "softmax"])
@pytest.mark.parametrize("dtype", ["f32", "bf16"])
@pytest.mark.parametrize("offsets", [[0], [0, 0], [0, 0, 0, 0]])
def test_classified_zero_elements(gpu_environment, op, dtype, offsets):
    # B=0 must not read offsets[1]; B>0,N=0 reductions still write identities.
    values = np.empty((0, 7), dtype=np.float32)
    execute(gpu_environment, ProgramSpec(op, dtype, 7), values, offsets)


@pytest.mark.parametrize("op", ["sum", "max", "softmax"])
@pytest.mark.parametrize("dtype", ["f32", "bf16"])
def test_nonfinite_values_in_every_class(gpu_environment, op, dtype):
    lengths = np.array([0, 2, 32, 33, 4096, 4097, 8201], dtype=np.int64)
    offsets = offsets_from_lengths(lengths)
    values = np.zeros((int(offsets[-1]), 7), dtype=np.float32)
    for begin, end in zip(offsets[:-1], offsets[1:]):
        if begin == end:
            continue
        values[begin:end, 2] = -np.inf
        values[begin, :6] = [np.nan, np.inf, -np.inf, -np.inf, np.inf, -np.inf]
        values[begin + 1, :6] = [1, 1, -np.inf, 3, -np.inf, 0]
    execute(gpu_environment, ProgramSpec(op, dtype, 7), values, offsets)


@pytest.mark.parametrize("op", ["sum", "softmax"])
@pytest.mark.parametrize("dtype", ["f32", "bf16"])
@pytest.mark.parametrize("d", [128, 1024])
def test_static_split_accumulation_and_partial_tail(gpu_environment, op, dtype, d):
    rows = 16385
    values = np.full((rows, d), np.float32(0.1 if op == "sum" else -np.log(10)), np.float32)
    if op == "softmax":
        values[0] = 0
    spec = ProgramSpec(op, dtype, d, hints=Hints(min_len=4097, max_len=rows))
    _, report = execute(gpu_environment, spec, values, [0, rows])
    assert report.classes == ("split",)
    assert len(report.kernel_names) >= 2


@pytest.mark.parametrize("dtype", ["f32", "bf16"])
def test_classified_softmax_inplace_global_statistics_boundary(gpu_environment, dtype):
    lengths = np.array([4097, 7, 0, 63, 8201], dtype=np.int64)
    offsets = offsets_from_lengths(lengths)
    values = np.random.default_rng(610).normal(size=(int(offsets[-1]), 7)).astype(np.float32)
    _, report = execute(gpu_environment, ProgramSpec("softmax", dtype, 7, alias="inplace"),
                        values, offsets, inplace=True)
    assert report.classes == ("warp", "cta", "split")
    assert len(report.kernel_names) > 3


@pytest.mark.parametrize("op", ["sum", "max", "softmax"])
def test_classified_preserves_original_segment_order(gpu_environment, op):
    lengths = np.array([7, 4097, 0, 63, 33, 8193, 1], dtype=np.int64)
    offsets = offsets_from_lengths(lengths)
    values = np.random.default_rng(849).integers(-4, 5, size=(int(offsets[-1]), 7)).astype(np.float32)
    spec = ProgramSpec(op, "f32", 7)
    original, _ = execute(gpu_environment, spec, values, offsets)
    pieces = [values[a:b] for a, b in zip(offsets[:-1], offsets[1:])]
    permutation = [5, 3, 2, 6, 1, 4, 0]
    permuted_values = np.concatenate([pieces[i] for i in permutation])
    permuted_offsets = offsets_from_lengths(lengths[permutation])
    permuted, _ = execute(gpu_environment, spec, permuted_values, permuted_offsets)
    expected = (np.concatenate([original[offsets[i]:offsets[i+1]] for i in permutation])
                if op == "softmax" else original[permutation])
    np.testing.assert_array_equal(permuted, expected)


@pytest.mark.parametrize("op", ["sum", "max", "softmax"])
@pytest.mark.parametrize("mapping", ["mixed", "warp", "cta", "split"])
def test_debug_visits_and_worklist_membership_exactly_once(gpu_environment, op, mapping):
    torch, compile_spec, _, _ = gpu_environment
    if mapping == "mixed":
        # Every class holds hundreds of segments, in classification order.
        lengths = np.resize(np.array([0, 7, 33, 4097], dtype=np.int64), 1031)
        hints = Hints()
    elif mapping == "warp":
        lengths, hints = np.array([0, 1, 32]), Hints(max_len=32)
    elif mapping == "cta":
        lengths, hints = np.array([33, 65, 4096]), Hints(min_len=33, max_len=4096)
    else:
        lengths, hints = np.array([4097, 8193, 16385]), Hints(min_len=4097, max_len=16385)
    offsets = offsets_from_lengths(lengths)
    values = np.random.default_rng(2529).integers(-2, 3, size=(int(offsets[-1]), 7)).astype(np.float32)
    spec = ProgramSpec(op, "f32", 7, hints=hints, debug_visits=True)
    program = compile_spec(spec)
    device_values, device_offsets = torch.from_numpy(values).cuda(), torch.from_numpy(offsets).cuda()
    output_rows = len(values) if op == "softmax" else len(lengths)
    output = torch.empty((output_rows, 7), device="cuda")
    workspace = {}
    report = program.run(device_values, device_offsets, output, debug_buffers=workspace)
    # Readback belongs to this test after execution, never to the runtime's
    # validated dispatch path. Holding these tensors also preserves their life.
    torch.cuda.current_stream().synchronize()
    descriptors = program.artifact.metadata["workspace"]
    assert set(workspace) == {entry["name"] for entry in descriptors}
    visits = workspace["visits"].cpu().numpy()
    np.testing.assert_array_equal(visits, np.ones(len(lengths), dtype=np.int32))
    counts = workspace["class_counts"].cpu().numpy()
    expected_class = np.where(lengths <= 32, 0, np.where(lengths <= 4096, 1, 2))
    np.testing.assert_array_equal(counts, np.bincount(expected_class, minlength=3))
    gathered = []
    for ordinal, count in enumerate(counts):
        # Queue entries past the class count are never written (classified ABI):
        # read only the written prefix, which also keeps initcheck clean.
        members = workspace["segment_ids"][ordinal, :int(count)].cpu().numpy()
        np.testing.assert_array_equal(np.sort(members), np.flatnonzero(expected_class == ordinal))
        gathered.extend(map(int, members))
    assert sorted(gathered) == list(range(len(lengths)))
    assert report.workspace_bytes == sum(t.numel() * t.element_size() for t in workspace.values())
    assert_matches(output.cpu().numpy(), values, offsets, operation=op, input_dtype="f32")


@pytest.mark.parametrize("op", ["sum", "max", "softmax"])
def test_zero_batch_debug_dispatch_allocates_and_launches_nothing(gpu_environment, op):
    torch, compile_spec, _, _ = gpu_environment
    program = compile_spec(ProgramSpec(op, "f32", 7, debug_visits=True))
    values = torch.empty((0, 7), device="cuda")
    offsets = torch.zeros(1, dtype=torch.int64, device="cuda")
    out = torch.empty_like(values)
    workspace = {}
    report = program.run(values, offsets, out, debug_buffers=workspace)
    assert workspace == {}
    assert report.workspace_bytes == 0
    assert report.classes == () and report.kernel_names == () and report.grids == ()


@pytest.mark.parametrize("segments", [1, 31, 32, 33, 127, 128, 129, 1000, 4133])
@pytest.mark.parametrize("pattern", ["mixed", "one_class", "rotating"])
def test_each_warp_reserves_one_ordered_run_per_class(gpu_environment, segments, pattern):
    # Classification aggregates per warp: lanes 0-2 reserve a run of positions
    # for each class, and the warp's segments of one class fill that run in
    # segment order. Counts, membership and run layout are read back from the
    # debug workspace after execution.
    torch, compile_spec, _, _ = gpu_environment
    rng = np.random.default_rng(segments)
    if pattern == "mixed":
        lengths = rng.choice([0, 1, 32, 33, 4096, 4097], size=segments)
    elif pattern == "one_class":
        lengths = np.full(segments, 40)
    else:
        lengths = np.array([0, 100, 4100])[np.arange(segments) % 3]
    lengths = lengths.astype(np.int64)
    offsets = offsets_from_lengths(lengths)
    values = rng.integers(-2, 3, size=(int(offsets[-1]), 1)).astype(np.float32)
    program = compile_spec(ProgramSpec("sum", "f32", 1, debug_visits=True))
    output = torch.empty((segments, 1), device="cuda")
    workspace = {}
    program.run(torch.from_numpy(values).cuda(), torch.from_numpy(offsets).cuda(), output, debug_buffers=workspace)
    torch.cuda.current_stream().synchronize()
    assert_matches(output.cpu().numpy(), values, offsets, operation="sum", input_dtype="f32")
    expected = np.where(lengths <= 32, 0, np.where(lengths <= 4096, 1, 2))
    counts = workspace["class_counts"].cpu().numpy()
    np.testing.assert_array_equal(counts, np.bincount(expected, minlength=3))
    position = np.empty(segments, dtype=np.int64)
    for ordinal, count in enumerate(counts):
        members = workspace["segment_ids"][ordinal, :int(count)].cpu().numpy()
        np.testing.assert_array_equal(np.sort(members), np.flatnonzero(expected == ordinal))
        position[members] = np.arange(int(count))
    for warp in range(0, segments, 32):
        for ordinal in range(3):
            run = position[warp:warp + 32][expected[warp:warp + 32] == ordinal]
            np.testing.assert_array_equal(run, np.arange(run[0], run[0] + len(run)) if len(run) else run)
