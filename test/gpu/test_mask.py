"""Independent additive-mask execution tests; CUDA absence is an explicit skip."""

from __future__ import annotations

import numpy as np
import pytest

from oracle.masked import assert_masked_softmax_matches
from ragged import Hints, ProgramSpec, RuntimeUnavailable, ValidationError
from test_correctness import gpu_environment, prepared


def mask_case(mapping):
    if mapping == "warp":
        lengths, hints = [0, 1, 7, 32], Hints(max_len=32)
    elif mapping == "cta":
        lengths, hints = [33, 65, 4096], Hints(min_len=33, max_len=4096)
    elif mapping == "split":
        lengths, hints = [4097, 8193], Hints(min_len=4097, max_len=8193)
    else:
        lengths, hints = [4097, 0, 1, 33, 8193, 32], Hints()
    return np.array([0, *np.cumsum(lengths)], dtype=np.int64), hints


def mask_spec(mapping, dtype="f32", d=7, offset_dtype="i64", *, scale=1.0, inplace=False):
    offsets, hints = mask_case(mapping)
    return ProgramSpec("softmax", dtype, d, offset_dtype, hints=hints, mask=True, scale=scale,
                       strategy="naive" if mapping == "naive" else "auto",
                       alias="inplace" if inplace else "none"), offsets


def execute_mask(gpu_environment, spec, values, offsets, mask, *, mode="validated"):
    torch, compile_spec, _, _ = gpu_environment
    values = prepared(values, spec.dtype)
    mask = np.ascontiguousarray(mask, dtype=np.float32)
    dtype = torch.float32 if spec.dtype == "f32" else torch.bfloat16
    tensor = torch.from_numpy(values).to(device="cuda", dtype=dtype)
    mask_guard = torch.full((len(values) + 2, spec.feature_dim), 91, device="cuda", dtype=torch.float32)
    device_mask = mask_guard[1:-1]
    device_mask.copy_(torch.from_numpy(mask).cuda())
    offset_type = torch.int32 if spec.offsets_dtype == "i32" else torch.int64
    bounds = torch.tensor(offsets, device="cuda", dtype=offset_type)
    guard = torch.full((len(values) + 2, spec.feature_dim), -64, device="cuda", dtype=dtype)
    output = tensor if spec.alias == "inplace" else guard[1:-1]
    report = compile_spec(spec).run(tensor, bounds, output, mask=device_mask, mode=mode)
    torch.cuda.current_stream().synchronize()
    assert torch.all(guard[0] == -64).item() and torch.all(guard[-1] == -64).item()
    assert torch.all(mask_guard[0] == 91).item() and torch.all(mask_guard[-1] == 91).item()
    np.testing.assert_array_equal(device_mask.cpu().numpy(), mask)
    if spec.alias != "inplace":
        np.testing.assert_array_equal(tensor.float().cpu().numpy(), values)
    assert_masked_softmax_matches(output.float().cpu().numpy(), values, offsets, mask,
                                 scale=spec.scale, input_dtype=spec.dtype)
    return report


@pytest.mark.parametrize("mapping", ["naive", "warp", "cta", "classified", "split"])
@pytest.mark.parametrize("dtype", ["f32", "bf16"])
@pytest.mark.parametrize("d", [1, 7, 128, 1024])
@pytest.mark.parametrize("offset_dtype", ["i32", "i64"])
def test_mask_all_mappings_use_same_fixed_oracle(gpu_environment, mapping, dtype, d, offset_dtype):
    spec, offsets = mask_spec(mapping, dtype, d, offset_dtype, scale=0.75)
    shape = (int(offsets[-1]), d)
    rng = np.random.default_rng(951)
    values = rng.normal(size=shape).astype(np.float32)
    mask = rng.uniform(-3, 3, size=shape).astype(np.float32)
    execute_mask(gpu_environment, spec, values, offsets, mask)


@pytest.mark.parametrize("mapping", ["naive", "warp", "cta", "classified", "split"])
@pytest.mark.parametrize("dtype", ["f32", "bf16"])
def test_f32_mask_cancellation_is_not_rounded_to_values_dtype(gpu_environment, mapping, dtype):
    spec, offsets = mask_spec(mapping, dtype, scale=0.5)
    values = np.full((int(offsets[-1]), 7), 65536, dtype=np.float32)
    mask = np.full_like(values, -32772)
    for begin, end in zip(offsets[:-1], offsets[1:]):
        if begin != end:
            mask[begin, :] = -32764
    execute_mask(gpu_environment, spec, values, offsets, mask)


@pytest.mark.parametrize("mapping", ["naive", "warp", "cta", "classified", "split"])
def test_scale_and_mask_keep_separate_f32_rounding(gpu_environment, mapping):
    # A contracted FMA leaves a large residual in the first row, while ordinary
    # f32 multiply followed by f32 add produces exactly zero for every row.
    scale = float(np.nextafter(np.float32(1), np.float32(2)))
    spec, offsets = mask_spec(mapping, d=1, scale=scale)
    values = np.zeros((int(offsets[-1]), 1), dtype=np.float32)
    mask = np.zeros_like(values)
    for begin, end in zip(offsets[:-1], offsets[1:]):
        if begin != end:
            values[begin, 0] = np.float32(1e20)
            mask[begin, 0] = -np.multiply(values[begin, 0], np.float32(scale), dtype=np.float32)
    execute_mask(gpu_environment, spec, values, offsets, mask)


@pytest.mark.parametrize("mapping", ["naive", "warp", "cta", "classified", "split"])
@pytest.mark.parametrize("dtype", ["f32", "bf16"])
@pytest.mark.parametrize("scale", [0.0, 0.5])
def test_mask_nonfinite_and_all_masked_columns_follow_arithmetic(gpu_environment, mapping, dtype, scale):
    spec, offsets = mask_spec(mapping, dtype, scale=scale)
    values = np.zeros((int(offsets[-1]), 7), dtype=np.float32)
    mask = np.zeros_like(values)
    for begin, end in zip(offsets[:-1], offsets[1:]):
        if begin == end:
            continue
        mask[begin:end, 0] = -np.inf  # Entire column masked: all NaN output.
        mask[begin, 1] = -np.inf       # Partially masked; finite remainder if present.
        mask[begin, 2] = np.inf
        mask[begin, 3] = np.nan
        values[begin, 4] = np.inf
        mask[begin, 4] = -np.inf
        values[begin, 5] = np.inf     # In particular, 0 * inf remains NaN.
        values[begin, 6] = -np.inf
    execute_mask(gpu_environment, spec, values, offsets, mask)


@pytest.mark.parametrize("mapping", ["naive", "warp", "cta", "classified", "split"])
@pytest.mark.parametrize("dtype", ["f32", "bf16"])
def test_mask_checked_success_and_inplace_global_phases(gpu_environment, mapping, dtype):
    spec, offsets = mask_spec(mapping, dtype, scale=-0.5, inplace=True)
    rng = np.random.default_rng(27)
    shape = (int(offsets[-1]), 7)
    execute_mask(gpu_environment, spec, rng.normal(size=shape), offsets,
                 rng.normal(size=shape).astype(np.float32), mode="checked")


@pytest.mark.parametrize("offsets", [[0], [0, 0], [0, 0, 0]])
@pytest.mark.parametrize("mode", ["validated", "checked"])
def test_mask_empty_shape_is_valid(gpu_environment, offsets, mode):
    spec = ProgramSpec("softmax", feature_dim=7, mask=True)
    execute_mask(gpu_environment, spec, np.empty((0, 7), np.float32),
                 np.asarray(offsets, np.int64), np.empty((0, 7), np.float32), mode=mode)


@pytest.mark.parametrize("inplace", [False, True])
def test_checked_bad_offsets_does_not_modify_mask_or_output(gpu_environment, inplace):
    torch, compile_spec, _, _ = gpu_environment
    spec = ProgramSpec("softmax", feature_dim=7, mask=True, alias="inplace" if inplace else "none")
    values = torch.arange(56, device="cuda", dtype=torch.float32).reshape(8, 7) / 8
    original = values.clone()
    mask = torch.full_like(values, -2)
    output = values if inplace else torch.full_like(values, -64)
    offsets = torch.tensor([0, 7, 6, 8], device="cuda", dtype=torch.int64)
    with pytest.raises(ValidationError) as caught:
        compile_spec(spec).run(values, offsets, output, mask=mask, mode="checked")
    assert caught.value.errors == {"nonmonotone": 1}
    torch.cuda.current_stream().synchronize()
    assert torch.equal(values, original) and torch.all(mask == -2).item()
    if not inplace:
        assert torch.all(output == -64).item()


def test_readonly_mask_may_alias_values(gpu_environment):
    torch, compile_spec, _, _ = gpu_environment
    spec = ProgramSpec("softmax", feature_dim=7, mask=True, scale=0.5)
    values = np.random.default_rng(923).normal(size=(40, 7)).astype(np.float32)
    tensor = torch.from_numpy(values).cuda()
    bounds = torch.tensor([0, 7, 40], device="cuda", dtype=torch.int64)
    output = torch.empty_like(tensor)
    compile_spec(spec).run(tensor, bounds, output, mask=tensor)
    assert_masked_softmax_matches(output.cpu().numpy(), values, np.array([0, 7, 40]), values, scale=0.5)
    np.testing.assert_array_equal(tensor.cpu().numpy(), values)


@pytest.mark.parametrize("overlap", ["out", "partial-out", "inplace-values", "offsets-i32", "offsets-i64"])
def test_disallowed_mask_aliases_are_rejected_before_writes(gpu_environment, overlap):
    torch, compile_spec, _, _ = gpu_environment
    offset_dtype = "i32" if overlap == "offsets-i32" else "i64"
    spec = ProgramSpec("softmax", feature_dim=4, offsets_dtype=offset_dtype, mask=True,
                       alias="inplace" if overlap == "inplace-values" else "none")
    values = torch.ones((8, 4), device="cuda")
    storage = torch.full((9, 4), -64, device="cuda")
    output, mask = storage[:8], torch.zeros_like(values)
    offsets = torch.tensor([0, 4, 8], dtype=torch.int32 if offset_dtype == "i32" else torch.int64, device="cuda")
    if overlap == "out":
        mask = output
    elif overlap == "partial-out":
        mask = storage[1:]
    elif overlap == "inplace-values":
        output = mask = values
    else:
        offsets = mask.view(torch.int32 if offset_dtype == "i32" else torch.int64).flatten()[:3]
        offsets.copy_(torch.tensor([0, 4, 8], dtype=offsets.dtype, device="cuda"))
    original_values, original_storage, original_mask = values.clone(), storage.clone(), mask.clone()
    with pytest.raises(ValidationError):
        compile_spec(spec).run(values, offsets, output, mask=mask)
    torch.cuda.current_stream().synchronize()
    assert torch.equal(values, original_values)
    assert torch.equal(storage, original_storage) and torch.equal(mask, original_mask)


@pytest.mark.parametrize("bad", ["missing", "broadcast", "rank", "bf16", "f64", "noncontiguous", "cpu", "grad", "unexpected"])
def test_invalid_mask_contract_preserves_output(gpu_environment, bad):
    torch, compile_spec, _, _ = gpu_environment
    spec = ProgramSpec("softmax", feature_dim=4, mask=bad != "unexpected")
    values = torch.zeros((8, 4), device="cuda")
    offsets = torch.tensor([0, 4, 8], device="cuda", dtype=torch.int64)
    output = torch.full_like(values, -64)
    mask = torch.zeros_like(values)
    if bad == "missing": mask = None
    elif bad == "broadcast": mask = torch.zeros((8, 1), device="cuda")
    elif bad == "rank": mask = torch.zeros((1, 8, 4), device="cuda")
    elif bad == "bf16": mask = mask.to(torch.bfloat16)
    elif bad == "f64": mask = mask.to(torch.float64)
    elif bad == "noncontiguous": mask = torch.zeros((4, 8), device="cuda").t()
    elif bad == "cpu": mask = torch.zeros((8, 4))
    elif bad == "grad": mask.requires_grad_(True)
    with pytest.raises(RuntimeUnavailable if bad == "cpu" else ValidationError):
        compile_spec(spec).run(values, offsets, output, mask=mask)
    torch.cuda.current_stream().synchronize()
    assert torch.all(output == -64).item()


@pytest.mark.parametrize("mode", ["validated", "checked"])
def test_mask_real_fullgraph_custom_op(gpu_environment, mode):
    from ragged.torch_adapter import bind
    torch, compile_spec, _, _ = gpu_environment
    spec = ProgramSpec("softmax", feature_dim=7, mask=True, scale=0.75)
    compiled = torch.compile(bind(compile_spec(spec), mode=mode), fullgraph=True)
    values = np.random.default_rng(812).normal(size=(40, 7)).astype(np.float32)
    mask = np.random.default_rng(813).normal(size=(40, 7)).astype(np.float32)
    offsets = np.array([0, 0, 7, 40], dtype=np.int64)
    output = compiled(torch.from_numpy(values).cuda(), torch.from_numpy(offsets).cuda(), torch.from_numpy(mask).cuda())
    assert_masked_softmax_matches(output.cpu().numpy(), values, offsets, mask, scale=0.75)
