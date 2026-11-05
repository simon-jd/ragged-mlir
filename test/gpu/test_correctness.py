"""Real sm_86 or sm_89 execution against an independent NumPy float64 oracle.

Run explicitly: pytest test/gpu/test_correctness.py -ra
Programs target the current device's chip, or RAGGED_TEST_CHIP (sm_86 cubins
also run on sm_89). With RAGGED_TOOLCHAIN_RECORD and RAGGED_TEST_CACHE they
are loaded from a cache that bench.precompile or an earlier run with the
compiler filled, and no compiler runs; a program missing there fails.
A missing or unsupported CUDA device is a reported skip, never numerical
evidence. With one available, missing compiler/native modules are failures.
No mocked GPU or CPU fallback is permitted. Device classification, split mapping and checked execution
have their own suites (test_classified.py, test_checked.py); sanitizer runs are
separate (results/sanitizer).
"""

from __future__ import annotations

from dataclasses import replace
import gc
import importlib
import os
from pathlib import Path

import numpy as np
import pytest

from bench.distributions import DISTRIBUTIONS, make_lengths, offsets_from_lengths
from oracle import assert_matches, quantize_bf16
from ragged import ArtifactCache, Compiler, Hints, Optimizations, ProgramSpec
from ragged.model import CHIPS, runs_on
from ragged.torch_adapter import (
    njt_softmax, register_program, register_torch_ops,
)


@pytest.fixture(scope="session")
def gpu_environment(tmp_path_factory):
    torch = pytest.importorskip("torch", reason="GPU correctness unverified: PyTorch is not installed")
    if not torch.cuda.is_available():
        pytest.skip("GPU correctness unverified: no CUDA device is available")
    capability = tuple(torch.cuda.get_device_capability())
    chip = target_chip(capability)
    if chip is None or not runs_on(chip, capability):
        pytest.skip(f"GPU correctness unverified: requires sm_86 or sm_89, current device is "
                    f"sm_{capability[0]}{capability[1]}" + (f" (RAGGED_TEST_CHIP={chip})" if chip else ""))
    compiler_path = os.environ.get("RAGGED_COMPILE", str(Path(__file__).resolve().parents[2] / "build/bin/ragged-compile"))
    record = os.environ.get("RAGGED_TOOLCHAIN_RECORD")
    compiler = Compiler(compiler_path, toolchain_record=record)
    # These must fail on a GPU-equipped machine if the project installation is
    # incomplete. They must not silently turn expected numerical tests into skips.
    compiler.toolchain(chip=chip)
    importlib.import_module("ragged._native")
    # One cache serves the whole session, which compiles far more hint and
    # option variants per signature than the production bound allows.
    root = os.environ.get("RAGGED_TEST_CACHE")
    if record and not root:
        raise RuntimeError("RAGGED_TOOLCHAIN_RECORD needs RAGGED_TEST_CACHE, the precompiled cache")
    cache = ArtifactCache(root or tmp_path_factory.mktemp("gpu-compiled-cache"), max_specializations=4096)
    programs = {}

    def compile_spec(spec):
        spec = retarget(spec, chip)
        if spec.ir_hash not in programs:
            programs[spec.ir_hash] = compiler.compile(spec, cache)
        return programs[spec.ir_hash]

    return torch, compile_spec, compiler, cache


def target_chip(capability: tuple[int, int]) -> str | None:
    """RAGGED_TEST_CHIP, or the chip of a device of ``capability`` if supported."""
    chip = os.environ.get("RAGGED_TEST_CHIP") or f"sm_{capability[0]}{capability[1]}"
    return chip if chip in CHIPS else None


def retarget(spec, chip: str):
    """The tests write specs for the default chip; compile them for ``chip``."""
    return spec if spec.chip == chip else replace(spec, chip=chip)


def precompiled() -> bool:
    """Whether programs come from a precompiled cache, with no compiler or ptxas."""
    return bool(os.environ.get("RAGGED_TOOLCHAIN_RECORD"))


def prepared(values, dtype):
    values = np.ascontiguousarray(values, dtype=np.float32)
    return quantize_bf16(values) if dtype == "bf16" else values


def execute(gpu_environment, spec, values, offsets, *, stream=None, inplace=False):
    torch, compile_spec, _, _ = gpu_environment
    values = prepared(values, spec.dtype)
    offsets = np.asarray(offsets, dtype=np.int32 if spec.offsets_dtype == "i32" else np.int64)
    program = compile_spec(spec)
    torch_dtype = torch.float32 if spec.dtype == "f32" else torch.bfloat16
    with torch.cuda.stream(stream or torch.cuda.current_stream()):
        device_values = torch.from_numpy(values).to(device="cuda", dtype=torch_dtype)
        device_offsets = torch.from_numpy(offsets).to(device="cuda")
        rows = len(values) if spec.op == "softmax" else len(offsets)-1
        if inplace:
            output, guard = device_values, None
        else:
            # A storage-offset view exercises descriptor offset handling and
            # catches writes immediately before/after the caller's output.
            guard = torch.full((rows+2, spec.feature_dim), -64, device="cuda", dtype=torch_dtype)
            output = guard[1:rows+1]
        report = program.run(device_values, device_offsets, output, stream=stream)
        if stream is not None:
            stream.synchronize()
        actual = output.float().cpu().numpy()
        if guard is not None:
            assert torch.all(guard[0] == -64).item()
            assert torch.all(guard[-1] == -64).item()
    assert_matches(actual, values, offsets, operation=spec.op, input_dtype=spec.dtype)
    return actual, report


@pytest.mark.parametrize("op", ["sum", "max", "softmax"])
@pytest.mark.parametrize("dtype", ["f32", "bf16"])
@pytest.mark.parametrize("d", [1, 16, 128, 1024])
@pytest.mark.parametrize("offset_dtype", ["i32", "i64"])
@pytest.mark.parametrize("distribution", DISTRIBUTIONS)
def test_full_naive_matrix(gpu_environment, op, dtype, d, offset_dtype, distribution):
    lengths = make_lengths(distribution, 64, total_elements=4096, seed=711)
    offsets = offsets_from_lengths(lengths, offset_dtype)
    values = np.random.default_rng(981).normal(size=(int(offsets[-1]), d)).astype(np.float32)
    _, report = execute(gpu_environment, ProgramSpec(op, dtype, d, offset_dtype, strategy="naive"), values, offsets)
    assert report.classes == ("cta",)
    assert len(report.kernel_names) == 1  # softmax: online statistics, no length bound needed


@pytest.mark.parametrize("dtype", ["f32", "bf16"])
@pytest.mark.parametrize("d", [1, 16, 128, 1024])
@pytest.mark.parametrize("offset_dtype", ["i32", "i64"])
@pytest.mark.parametrize("distribution", DISTRIBUTIONS)
def test_full_naive_three_pass_softmax_matrix(gpu_environment, dtype, d, offset_dtype, distribution):
    lengths = make_lengths(distribution, 64, total_elements=4096, seed=711)
    offsets = offsets_from_lengths(lengths, offset_dtype)
    values = np.random.default_rng(981).normal(size=(int(offsets[-1]), d)).astype(np.float32)
    spec = ProgramSpec("softmax", dtype, d, offset_dtype, strategy="naive",
                       optimizations=Optimizations(online_fusion=False))
    _, report = execute(gpu_environment, spec, values, offsets)
    assert report.classes == ("cta",) and len(report.kernel_names) == 3


@pytest.mark.parametrize("op", ["sum", "max", "softmax"])
@pytest.mark.parametrize("dtype", ["f32", "bf16"])
@pytest.mark.parametrize("offset_dtype", ["i32", "i64"])
@pytest.mark.parametrize("mapping", ["warp", "cta"])
@pytest.mark.parametrize("d", [1, 7, 16, 128, 1024])
def test_static_mappings_and_online_softmax(gpu_environment, op, dtype, offset_dtype, mapping, d):
    lengths = np.array([0, 1, 7, 31, 32] if mapping == "warp" else [33, 65, 128], dtype=np.int64)
    hints = Hints(max_len=32) if mapping == "warp" else Hints(min_len=33, max_len=128)
    values = np.random.default_rng(94).normal(size=(int(lengths.sum()), d)).astype(np.float32)
    _, report = execute(gpu_environment, ProgramSpec(op, dtype, d, offset_dtype, hints=hints),
                        values, offsets_from_lengths(lengths, offset_dtype))
    assert report.classes == (mapping,)
    assert len(report.kernel_names) == 1
    if op == "softmax":
        assert "online" in report.kernel_names[0]


@pytest.mark.parametrize("op", ["sum", "max", "softmax"])
@pytest.mark.parametrize("dtype", ["f32", "bf16"])
@pytest.mark.parametrize("offsets", [[0], [0, 0, 0], [0, 0, 1, 1, 4]])
def test_empty_all_empty_and_tail_features(gpu_environment, op, dtype, offsets):
    values = np.arange(offsets[-1]*3, dtype=np.float32).reshape(offsets[-1], 3)
    execute(gpu_environment, ProgramSpec(op, dtype, 3, strategy="naive"), values, offsets)


@pytest.mark.parametrize("dtype", ["f32", "bf16"])
@pytest.mark.parametrize("variant", ["naive", "online_warp", "online_cta"])
def test_softmax_special_values_including_single_positive_infinity(gpu_environment, dtype, variant):
    # Columns: NaN; +inf with finite; all -inf; finite with -inf;
    # exactly one +inf with only -inf (online empty-lane regression).
    rows = 64 if variant == "online_cta" else 2
    values = np.full((rows, 5), -np.inf, dtype=np.float32)
    values[0] = [np.nan, np.inf, -np.inf, -np.inf, np.inf]
    values[1] = [1, 1, -np.inf, 3, -np.inf]
    if variant == "naive":
        spec = ProgramSpec("softmax", dtype, 5, strategy="naive")
    else:
        hints = Hints(min_len=33, max_len=128) if variant == "online_cta" else Hints(max_len=32)
        spec = ProgramSpec("softmax", dtype, 5, hints=hints)
    execute(gpu_environment, spec, values, [0, rows])


@pytest.mark.parametrize("op", ["sum", "max"])
@pytest.mark.parametrize("dtype", ["f32", "bf16"])
def test_reduction_nonfinite(gpu_environment, op, dtype):
    values = np.array([[np.nan, np.inf, -np.inf, -2], [1, -np.inf, -np.inf, -4]], dtype=np.float32)
    execute(gpu_environment, ProgramSpec(op, dtype, 4, strategy="naive"), values, [0, 2, 2])


@pytest.mark.parametrize("op", ["sum", "max", "softmax"])
def test_single_very_long_segment(gpu_environment, op):
    values = np.tile(np.array([[1, -1, 0.5]], dtype=np.float32), (131073, 1))
    execute(gpu_environment, ProgramSpec(op, "f32", 3, strategy="naive"), values, [0, len(values)])


@pytest.mark.parametrize("d", [128, 1024])
@pytest.mark.parametrize("op,online", [("sum", True), ("softmax", True), ("softmax", False)],
                         ids=["sum", "softmax-online", "softmax-three-pass"])
def test_long_segment_accumulation_accuracy(gpu_environment, d, op, online):
    # With d_tile=32 and 128 threads there are only four row lanes. A single
    # uncompensated f32 accumulator per lane violates the fixed oracle bound
    # even at 16384 rows. Integer-only long segments cannot expose this error.
    rows = 16384
    if op == "sum":
        values = np.full((rows, d), np.float32(0.1), dtype=np.float32)
    else:
        values = np.full((rows, d), np.float32(-np.log(10)), dtype=np.float32)
        values[0] = 0
    spec = ProgramSpec(op, "f32", d, strategy="naive", optimizations=Optimizations(online_fusion=online))
    _, report = execute(gpu_environment, spec, values, [0, rows])
    assert len(report.kernel_names) == (1 if online else 3)


@pytest.mark.parametrize("dtype", ["f32", "bf16"])
@pytest.mark.parametrize("d", [128, 1024])
def test_compensated_sum_preserves_nonfinite_lane_sequences(gpu_environment, dtype, d):
    # Rows 0, 4 and 8 belong to the same row lane at these feature widths.
    # In particular, +inf followed by a finite value must remain +inf;
    # an unguarded Kahan compensation turns inf-inf into a spurious NaN.
    values = np.zeros((12, d), dtype=np.float32)
    values[0, :5] = [np.inf, -np.inf, np.inf, np.nan, 3]
    values[4, :5] = [2, 2, -np.inf, 2, np.inf]
    values[8, :5] = [3, 3, 3, 3, 2]
    execute(gpu_environment, ProgramSpec("sum", dtype, d, strategy="naive"), values, [0, 12])


@pytest.mark.parametrize("op", ["sum", "max", "softmax"])
@pytest.mark.parametrize("dtype", ["f32", "bf16"])
@pytest.mark.parametrize("variant", ["naive", "warp"])
def test_device_metamorphic_relations(gpu_environment, op, dtype, variant):
    # Integer inputs and shifts are exactly representable in both input dtypes,
    # so bf16 input rounding cannot invalidate shift invariance.
    values = np.random.default_rng(90).integers(-4, 5, size=(39, 7)).astype(np.float32)
    offsets = np.array([0, 3, 3, 30, 39], dtype=np.int64)
    hints = Hints(max_len=32) if variant == "warp" else Hints()
    spec = ProgramSpec(op, dtype, 7, strategy="auto" if variant == "warp" else "naive", hints=hints)
    original, _ = execute(gpu_environment, spec, values, offsets)
    pieces = [values[a:b] for a, b in zip(offsets[:-1], offsets[1:])]
    permutation = [3, 1, 0, 2]
    permuted_values = np.concatenate([pieces[i] for i in permutation])
    permuted_offsets = np.array([0, *np.cumsum([len(pieces[i]) for i in permutation])], dtype=np.int64)
    permuted, _ = execute(gpu_environment, spec, permuted_values, permuted_offsets)
    expected = np.concatenate([original[offsets[i]:offsets[i+1]] for i in permutation]) if op == "softmax" else original[permutation]
    np.testing.assert_array_equal(permuted, expected)
    inserted, _ = execute(gpu_environment, spec, values, [0, 0, 3, 3, 30, 39, 39])
    np.testing.assert_array_equal(inserted if op == "softmax" else inserted[1:5], original)
    if op == "sum":
        split, _ = execute(gpu_environment, spec, values, [0, 3, 3, 15, 30, 39])
        recombined = np.stack([split[0], split[1], split[2]+split[3], split[4]])
        assert_matches(recombined, prepared(values, dtype), offsets, operation="sum", input_dtype=dtype)
    if op == "softmax":
        shifted, _ = execute(gpu_environment, spec, values+5, offsets)
        np.testing.assert_allclose(shifted, original, rtol=2e-5 if dtype == "f32" else 2e-2, atol=2e-6 if dtype == "f32" else 2e-3)


@pytest.mark.parametrize("dtype", ["f32", "bf16"])
@pytest.mark.parametrize("rows", [0, 129])
def test_inplace_softmax_requires_three_kernel_schedule(gpu_environment, dtype, rows):
    spec = ProgramSpec("softmax", dtype, 7, alias="inplace", strategy="naive")
    values = np.random.default_rng(64).normal(size=(rows, 7)).astype(np.float32)
    _, report = execute(gpu_environment, spec, values, [0, 0, rows], inplace=True)
    assert len(report.kernel_names) == 3


def test_nondefault_stream_and_program_lifetime(gpu_environment):
    torch, _, compiler, cache = gpu_environment
    spec = retarget(ProgramSpec("sum", "f32", 7, strategy="naive"),
                    target_chip(tuple(torch.cuda.get_device_capability())))
    # Avoid the test fixture's program memo so deleting it actually drops the
    # last local CompiledProgram reference while its kernels may be in flight.
    program = compiler.compile(spec, cache)
    values = np.random.default_rng(143).normal(size=(4096, 7)).astype(np.float32)
    offsets = np.array([0, 1, 64, 4096], dtype=np.int64)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        device_values = torch.from_numpy(values).cuda()
        device_offsets = torch.from_numpy(offsets).cuda()
        output = torch.empty((3, 7), device="cuda")
        program.run(device_values, device_offsets, output, stream=stream)
        del program
        gc.collect()
        # A dependent torch kernel establishes that native launch uses this stream.
        copied = output + 1
        event = torch.cuda.Event()
        event.record(stream)
    event.synchronize()
    assert_matches((copied-1).cpu().numpy(), values, offsets, operation="sum", input_dtype="f32")


def test_three_pass_workspace_does_not_reuse_pending_other_stream_storage(gpu_environment, monkeypatch):
    torch, compile_spec, _, _ = gpu_environment
    spec = ProgramSpec("softmax", "f32", 128, strategy="naive", optimizations=Optimizations(online_fusion=False))
    program = compile_spec(spec)
    values = np.random.default_rng(184).normal(size=(512, 128)).astype(np.float32)
    offsets = np.arange(0, 513, 8, dtype=np.int64)
    device_values, device_offsets = torch.from_numpy(values).cuda(), torch.from_numpy(offsets).cuda()
    output = torch.empty_like(device_values)
    delay_left = torch.ones((4096, 4096), device="cuda")
    delay_right, delay_output = torch.ones_like(delay_left), torch.empty_like(delay_left)
    producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
    producer.wait_stream(torch.cuda.current_stream())
    consumer.wait_stream(torch.cuda.current_stream())
    # Load the module and establish external tensor dependencies before the race
    # setup, so first-load driver work cannot accidentally hide allocator reuse.
    program.run(device_values, device_offsets, output, stream=consumer)
    consumer.synchronize()
    pending = torch.cuda.Event()
    with torch.cuda.stream(producer):
        # The same size as the call's scratch (two 64x128 f32 statistics
        # buffers), so a producer-stream allocation would reuse this block.
        old_stats = [torch.empty((2 * 64, 128), device="cuda")]
        old_addresses = {tensor.data_ptr() for tensor in old_stats}
        for _ in range(64):
            torch.mm(delay_left, delay_right, out=delay_output)
        for tensor in old_stats:
            tensor.fill_(float("nan"))
        pending.record(producer)
        del tensor, old_stats
    real_empty, allocations = torch.empty, []
    def observe_allocation(*args, **kwargs):
        tensor = real_empty(*args, **kwargs)
        allocations.append((tensor.data_ptr(), torch.cuda.current_stream().cuda_stream, pending.query()))
        return tensor
    with monkeypatch.context() as patch:
        patch.setattr(torch, "empty", observe_allocation)
        with torch.cuda.stream(producer):
            assert torch.cuda.current_stream() == producer
            report = program.run(device_values, device_offsets, output, stream=consumer)
            assert torch.cuda.current_stream() == producer
    consumer.synchronize()
    producer.synchronize()
    assert len(report.kernel_names) == 3
    assert len(allocations) == 1 and report.workspace_bytes == 2 * 64 * 128 * 4
    assert all(stream == consumer.cuda_stream for _, stream, _ in allocations)
    assert all(address not in old_addresses for address, _, was_complete in allocations if not was_complete)
    assert_matches(output.cpu().numpy(), values, offsets, operation="softmax", input_dtype="f32")


def test_real_njt_and_torch_compile_fullgraph(gpu_environment):
    torch, compile_spec, _, _ = gpu_environment
    values = np.random.default_rng(15).normal(size=(13, 7)).astype(np.float32)
    offsets = np.array([0, 3, 3, 13], dtype=np.int64)
    spec = ProgramSpec("softmax", "f32", 7, strategy="naive")
    program = compile_spec(spec)
    device_values, device_offsets = torch.from_numpy(values).cuda(), torch.from_numpy(offsets).cuda()
    nt = torch.nested.nested_tensor_from_jagged(device_values, offsets=device_offsets)
    adapted = njt_softmax(nt, program=program)
    assert_matches(adapted.values().cpu().numpy(), values, offsets, operation="softmax", input_dtype="f32")
    register_torch_ops()
    token = register_program(program)

    def function(x, positions):
        return torch.ops.ragged_mlir.segment_softmax(x, positions, token)

    compiled = torch.compile(function, fullgraph=True)
    assert_matches(compiled(device_values, device_offsets).cpu().numpy(), values, offsets,
                   operation="softmax", input_dtype="f32")
