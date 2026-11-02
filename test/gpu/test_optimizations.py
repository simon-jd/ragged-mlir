"""Independent optimization ablations; a missing CUDA device is not a pass."""
from __future__ import annotations

import gc

import numpy as np
import pytest

from oracle.masked import masked_logits
from oracle.reference import assert_matches
from ragged import Hints, Optimizations, ProgramSpec, ValidationError
from test_correctness import gpu_environment, prepared

FLAGS = ("classification", "online_fusion", "warp_mapping", "split_mapping", "shuffle", "pointwise_fusion",
         "row_pipelining", "vectorize")


def offsets_for(lengths):
    return np.asarray([0, *np.cumsum(lengths)], dtype=np.int64)


def run_case(gpu_environment, spec, values, offsets, mask=None, *, mode="validated"):
    torch, compile_spec, _, _ = gpu_environment
    values = prepared(values, spec.dtype)
    dtype = torch.float32 if spec.dtype == "f32" else torch.bfloat16
    source = torch.from_numpy(values).to(device="cuda", dtype=dtype)
    bounds = torch.tensor(offsets, device="cuda", dtype=torch.int32 if spec.offsets_dtype == "i32" else torch.int64)
    rows = len(values) if spec.op == "softmax" else len(offsets) - 1
    guard = torch.full((rows + 2, spec.feature_dim), -64, dtype=dtype, device="cuda")
    output = source if spec.alias == "inplace" else guard[1:-1]
    device_mask = torch.from_numpy(np.ascontiguousarray(mask, dtype=np.float32)).cuda() if mask is not None else None
    buffers = {} if spec.debug_visits else None
    report = compile_spec(spec).run(source, bounds, output, mask=device_mask, mode=mode, debug_buffers=buffers)
    torch.cuda.current_stream().synchronize()
    with np.errstate(invalid="ignore", over="ignore"):
        logits = masked_logits(values, mask, scale=spec.scale) if mask is not None else np.multiply(values, np.float32(spec.scale), dtype=np.float32)
    assert_matches(output.float().cpu().numpy(), logits, offsets, operation=spec.op, input_dtype=spec.dtype)
    assert torch.all(guard[0] == -64).item() and torch.all(guard[-1] == -64).item()
    if spec.alias != "inplace":
        np.testing.assert_array_equal(source.float().cpu().numpy(), values)
    if device_mask is not None:
        np.testing.assert_array_equal(device_mask.cpu().numpy(), mask)
    assert report.optimizations["requested"] == spec.optimizations.to_dict()
    if buffers and "logits" in buffers:
        assert buffers["logits"].dtype == torch.float32
        np.testing.assert_array_equal(buffers["logits"].cpu().numpy(), logits)
    if buffers and "visits" in buffers:
        np.testing.assert_array_equal(buffers["visits"].cpu().numpy(), np.ones(len(offsets) - 1, dtype=np.int32))
    return report


@pytest.mark.parametrize("disabled", FLAGS)
@pytest.mark.parametrize("op", ["sum", "max", "softmax"])
@pytest.mark.parametrize("dtype", ["f32", "bf16"])
@pytest.mark.parametrize("offset_dtype", ["i32", "i64"])
def test_each_disabled_optimization_obeys_the_same_oracle(gpu_environment, disabled, op, dtype, offset_dtype):
    if disabled in ("online_fusion", "warp_mapping", "shuffle"):
        lengths, hints = [0, 1, 7, 32, 2, 31], Hints(max_len=32)
    elif disabled == "split_mapping":
        lengths, hints = [4097, 8193], Hints(min_len=4097, max_len=8193)
    else:
        lengths, hints = [4097, 0, 1, 33, 8193, 32], Hints()
    offsets = offsets_for(lengths)
    mask_enabled = op == "softmax"
    spec = ProgramSpec(op, dtype, 7, offset_dtype, hints=hints, scale=0.5, mask=mask_enabled,
                       debug_visits=disabled == "pointwise_fusion", optimizations=Optimizations(**{disabled: False}))
    rng = np.random.default_rng(573)
    values = rng.normal(size=(int(offsets[-1]), 7)).astype(np.float32)
    mask = rng.uniform(-2, 2, size=values.shape).astype(np.float32) if mask_enabled else None
    report = run_case(gpu_environment, spec, values, offsets, mask)
    assert report.optimizations["effective"][disabled] is False
    if disabled == "classification": assert report.classes == ("cta",)
    if disabled == "warp_mapping": assert "warp" not in report.classes
    if disabled == "split_mapping": assert "split" not in report.classes
    if disabled == "pointwise_fusion":
        assert report.pointwise_materialized and "materialize" in report.kernel_names[0]
        assert report.workspace_bytes >= values.size * 4


@pytest.mark.parametrize("tile", [1, 2, 4, 8, 16, 32])
@pytest.mark.parametrize("d", [7, 128, 1024])
@pytest.mark.parametrize("op", ["sum", "max", "softmax"])
def test_feature_tile_tail_and_shared_tree_geometry(gpu_environment, tile, d, op):
    # A one-warp shared tree stresses feature-interleaved reductions, including
    # tile>D inactive feature lanes. Every lane must reach each barrier.
    offsets = offsets_for([0, 1, 2, 7, 31, 32])
    spec = ProgramSpec(op, "f32", d, hints=Hints(max_len=32),
                       optimizations=Optimizations(feature_tile=tile, shuffle=False))
    values = np.random.default_rng(839).normal(size=(int(offsets[-1]), d)).astype(np.float32)
    report = run_case(gpu_environment, spec, values, offsets)
    assert report.d_tile == tile and report.classes == ("warp",)
    assert all(grid[1] == (d + tile - 1) // tile for grid in report.grids)


@pytest.mark.parametrize("dtype", ["f32", "bf16"])
@pytest.mark.parametrize("op", ["sum", "max", "softmax"])
def test_all_optimizations_off_and_long_accumulation(gpu_environment, dtype, op):
    options = Optimizations(**{name: False for name in FLAGS}, feature_tile=1)
    offsets = offsets_for([0, 16384, 0, 3])
    spec = ProgramSpec(op, dtype, 128, scale=0.5, mask=op == "softmax", optimizations=options)
    values = np.full((int(offsets[-1]), 128), 0.1, dtype=np.float32)
    mask = np.zeros_like(values) if spec.mask else None
    if mask is not None:
        mask[0] = 1  # Catch a normalization path that forgot the materialized mask.
    report = run_case(gpu_environment, spec, values, offsets, mask, mode="checked")
    assert report.classes == ("cta",) and report.pointwise_materialized
    assert not any(report.optimizations["effective"].values())
    assert "validation" in report.kernel_names[0] or "offsets" in report.kernel_names[0]
    assert "materialize" in report.kernel_names[2]


# A 4096-row segment reaches online statistics either through static hints
# (direct CTA mapping) or through classification (the CTA class of a
# classified program without hints).
ONLINE_ROUTES = pytest.mark.parametrize("hints", [Hints(min_len=33, max_len=4096), Hints()],
                                        ids=["static-bounds", "classified"])


@ONLINE_ROUTES
@pytest.mark.parametrize("d", [1, 128, 1024])
@pytest.mark.parametrize("materialized", [False, True])
def test_online_statistics_small_mass_accumulation_meets_fixed_tolerance(gpu_environment, d, materialized, hints):
    # Four row lanes each add 1023 tiny masses to one. Uncompensated online
    # accumulation shifts normalization by ~5.18e-5 and fails the f32 oracle.
    spec = ProgramSpec("softmax", feature_dim=d, hints=hints,
                       mask=materialized, optimizations=Optimizations(feature_tile=32, pointwise_fusion=not materialized))
    values = np.full((4096, d), np.float32(np.log(np.float32(0.00013))), dtype=np.float32)
    values[:4] = 0
    mask = np.zeros_like(values) if materialized else None
    report = run_case(gpu_environment, spec, values, np.array([0,4096], dtype=np.int64), mask)
    assert any(name.endswith("_online") for name in report.kernel_names)


@ONLINE_ROUTES
@pytest.mark.parametrize("step", [7.45717798622536e-9, 6.613177538383752e-5])
def test_online_statistics_repeated_max_rescaling_meets_fixed_tolerance(gpu_environment, step, hints):
    spec = ProgramSpec("softmax", feature_dim=128, hints=hints)
    values = np.repeat((np.arange(4096,dtype=np.float32) * np.float32(step))[:,None], 128, axis=1)
    report = run_case(gpu_environment, spec, values, np.array([0,4096],dtype=np.int64))
    assert any(name.endswith("_online") for name in report.kernel_names)


@pytest.mark.parametrize("dtype", ["f32", "bf16"])
@pytest.mark.parametrize("inplace", [False, True])
def test_materialized_mask_is_f32_and_never_applied_twice(gpu_environment, dtype, inplace):
    offsets = offsets_for([0, 1, 32, 33, 4097, 8193])
    spec = ProgramSpec("softmax", dtype, 7, scale=0.5, mask=True, debug_visits=True,
                       alias="inplace" if inplace else "none",
                       optimizations=Optimizations(pointwise_fusion=False, feature_tile=2, shuffle=False))
    values = np.full((int(offsets[-1]), 7), 65536, dtype=np.float32)
    mask = np.full_like(values, -32772)
    for begin, end in zip(offsets[:-1], offsets[1:]):
        if begin < end: mask[begin] = -32764
    report = run_case(gpu_environment, spec, values, offsets, mask, mode="checked")
    assert report.pointwise_materialized


@pytest.mark.parametrize("scale", [0.0, 0.5])
@pytest.mark.parametrize("dtype", ["f32", "bf16"])
def test_materialized_nonfinite_values_keep_ordinary_arithmetic(gpu_environment, scale, dtype):
    offsets = offsets_for([0, 32, 33, 4097])
    spec = ProgramSpec("softmax", dtype, 7, mask=True, scale=scale,
                       optimizations=Optimizations(pointwise_fusion=False, shuffle=False, feature_tile=4))
    values = np.zeros((int(offsets[-1]), 7), dtype=np.float32)
    mask = np.zeros_like(values)
    for begin, end in zip(offsets[:-1], offsets[1:]):
        if begin == end: continue
        mask[begin:end, 0] = -np.inf
        mask[begin, 1] = np.inf
        mask[begin, 2] = np.nan
        values[begin, 3:5] = [np.inf, -np.inf]
        mask[begin, 5] = -np.inf
        values[begin, 6] = np.inf
        mask[begin, 6] = -np.inf
    run_case(gpu_environment, spec, values, offsets, mask)


@pytest.mark.parametrize("op", ["sum", "max", "softmax"])
@pytest.mark.parametrize("offsets", [[0], [0, 0, 0]])
@pytest.mark.parametrize("mode", ["validated", "checked"])
def test_materialization_empty_shapes_and_B0_early_return(gpu_environment, op, offsets, mode):
    spec = ProgramSpec(op, feature_dim=7, scale=0.5,
                       optimizations=Optimizations(pointwise_fusion=False, classification=False))
    report = run_case(gpu_environment, spec, np.empty((0, 7), dtype=np.float32), np.array(offsets, dtype=np.int64), mode=mode)
    if len(offsets) == 1:
        assert report.classes == () and report.workspace_bytes == (56 if mode == "checked" else 0)
        assert len(report.kernel_names) == (2 if mode == "checked" else 0)


def test_failed_checked_offsets_leave_inplace_values_and_debug_buffers_untouched(gpu_environment):
    torch, compile_spec, _, _ = gpu_environment
    spec = ProgramSpec("softmax", feature_dim=7, mask=True, scale=0.5, alias="inplace", debug_visits=True,
                       optimizations=Optimizations(pointwise_fusion=False))
    values = torch.arange(56, device="cuda", dtype=torch.float32).reshape(8, 7)
    original = values.clone()
    mask = torch.full_like(values, -2)
    bounds = torch.tensor([0, 7, 6, 8], device="cuda", dtype=torch.int64)
    buffers = {}
    with pytest.raises(ValidationError) as caught:
        compile_spec(spec).run(values, bounds, values, mask=mask, mode="checked", debug_buffers=buffers)
    assert caught.value.errors == {"nonmonotone": 1} and buffers == {}
    torch.cuda.current_stream().synchronize()
    assert torch.equal(values, original) and torch.all(mask == -2).item()


@pytest.mark.parametrize("mode", ["validated", "checked"])
def test_materialize_uses_explicit_noncurrent_stream_and_safe_allocator_order(gpu_environment, mode):
    torch, compile_spec, _, _ = gpu_environment
    spec = ProgramSpec("softmax", feature_dim=128, mask=True, scale=0.5, debug_visits=True,
                       optimizations=Optimizations(pointwise_fusion=False, shuffle=False))
    program = compile_spec(spec)
    offsets = offsets_for([33, 4097, 32])
    shape = (int(offsets[-1]), 128)
    original = np.random.default_rng(776).normal(size=shape).astype(np.float32)
    expected_mask = np.zeros(shape, dtype=np.float32)
    expected_mask[::3] = 0.75
    current, consumer = torch.cuda.Stream(), torch.cuda.Stream()
    with torch.cuda.stream(consumer):
        values = torch.from_numpy(original).cuda()
        mask = torch.full(shape, float("nan"), device="cuda")
        bounds = torch.tensor(offsets, dtype=torch.int64, device="cuda")
        output = torch.empty_like(values)
        delay = torch.randn((1024, 1024), device="cuda")
        product = torch.empty_like(delay)
    consumer.synchronize()
    for _ in range(3):
        with torch.cuda.stream(current):
            # Keep a same-size cached allocation busy on A. The run explicitly
            # consumes B while A remains current, including the logits buffer.
            stale = torch.empty(shape, device="cuda")
            for _ in range(6): torch.mm(delay, delay, out=product)
            stale.fill_(float("nan"))
            del stale
            gc.collect()
            with torch.cuda.stream(consumer):
                for _ in range(3): torch.mm(delay, delay, out=torch.empty_like(product))
                mask.copy_(torch.from_numpy(expected_mask).cuda())
            buffers = {}
            assert torch.cuda.current_stream() == current
            report = program.run(values, bounds, output, mask=mask, stream=consumer, mode=mode, debug_buffers=buffers)
            assert torch.cuda.current_stream() == current
        consumer.synchronize()
        current.synchronize()
        expected = masked_logits(original, expected_mask, scale=0.5)
        assert_matches(output.cpu().numpy(), expected, offsets, operation="softmax", input_dtype="f32")
        np.testing.assert_array_equal(buffers["logits"].cpu().numpy(), expected)
        assert report.pointwise_materialized
