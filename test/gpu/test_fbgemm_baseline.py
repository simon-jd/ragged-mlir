"""FBGEMM 1.9.0 device qualification against the shared oracle.

Missing CUDA/package yields a documented skip. An installed broken package or
incorrect foreign result fails. The cases FBGEMM 1.9.0 gets wrong are strict
xfails, so a fixed release turns them into failures to revisit.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from baselines import fbgemm_reference as adapter
from oracle import assert_matches, quantize_bf16

SEED = 180821


@pytest.fixture(scope="module")
def fbgemm_environment():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("FBGEMM device qualification pending: no CUDA device")
    if tuple(torch.cuda.get_device_capability()) != (8, 9):
        pytest.skip("FBGEMM comparison qualification requires sm_89")
    try:
        provenance = adapter.toolchain()
    except adapter.OptionalDependencyUnavailable as error:
        pytest.skip(f"FBGEMM device qualification pending: {error}")
    return torch, provenance


def check_case(environment, name, values, offsets, dtype, *, with_hint=False):
    torch, provenance = environment
    if dtype == "bf16":
        values = quantize_bf16(values)
    maximum = int(np.diff(offsets).max(initial=0)) if with_hint else None
    info = adapter.configuration(len(values), values.shape[1], len(offsets) - 1, max_len=maximum)
    try:
        # Input creation and the complete foreign call use the same nondefault
        # stream. Synchronization is test observation, not part of the adapter.
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            tensor = torch.from_numpy(values).to("cuda", dtype=torch.float32 if dtype == "f32" else torch.bfloat16)
            bounds = torch.from_numpy(offsets).to("cuda")
            output = adapter.call(tensor, bounds, max_len=maximum)
        stream.synchronize()
        assert output.shape == tensor.shape and output.dtype == tensor.dtype and output.device == tensor.device
        assert_matches(output.float().cpu().numpy(), values, offsets, operation="softmax", input_dtype=dtype)
    except Exception as error:
        root = Path(os.environ.get("RAGGED_GPU_FAILURE_DIR", "evidence/gpu-failures/fbgemm"))
        destination = root / name
        destination.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination / "input.npz", values=values, offsets=offsets)
        source = Path(adapter.__file__).read_bytes()
        (destination / "failure.json").write_text(json.dumps({
            "schema": 1, "case": name, "seed": SEED, "operation": "softmax", "dtype": dtype,
            "offset_dtype": str(offsets.dtype), "configuration": info, "toolchain": provenance,
            "adapter_sha256": hashlib.sha256(source).hexdigest(), "error": repr(error),
            "device_semantics": "failed", "tolerance": "unchanged shared oracle",
        }, indent=2) + "\n")
        (destination / "adapter.py").write_bytes(source)
        error.add_note(f"Exact input, adapter, and optional package identity saved in {destination}")
        raise


@pytest.mark.parametrize("dtype", ["f32", "bf16"])
@pytest.mark.parametrize("bits", [32, 64])
@pytest.mark.parametrize("d", [1, 7, 16, 128, 1024])
@pytest.mark.parametrize("with_hint", [False, True])
def test_fbgemm_finite_mixed_lengths(fbgemm_environment, dtype, bits, d, with_hint):
    offsets = np.array([0, 0, 1, 33, 4130, 4130, 4161], dtype=f"i{bits // 8}")
    values = np.random.default_rng(SEED).normal(size=(4161, d)).astype(np.float32)
    check_case(fbgemm_environment, f"mixed-{dtype}-i{bits}-d{d}-hints{with_hint}",
               values, offsets, dtype, with_hint=with_hint)


SPECIAL_CASES = {
    "positive_infinity_first": [np.inf, 0, -1, -np.inf],
    "positive_infinity_later": [0, np.inf, -1, -np.inf],
    "all_negative_infinity": [-np.inf] * 4,
    "nan_later": [0, -1, np.nan, -2],
    "finite_and_negative_infinity": [-np.inf, -2.5, -np.inf, -3.75],
    "large_negative_i32": [-1e10] * 4,
    "large_negative_i64": [-1e20] * 4,
    "large_positive": [1e20] * 4,
    "all_empty": [],
    "zero_segments": [],
}


@pytest.mark.parametrize("dtype", ["f32", "bf16"])
@pytest.mark.parametrize("bits", [32, 64])
@pytest.mark.parametrize("d", [1, 7, 128])
@pytest.mark.parametrize("case", list(SPECIAL_CASES))
def test_fbgemm_boundaries_and_known_risks(fbgemm_environment, dtype, bits, d, case, request):
    # Measured on 1.9.0+cu130: with CUDA 13, jagged_softmax takes the block
    # maximum in the offsets' integer type (cuda::maximum<index_t>), so logits
    # outside that type's range, or +inf after a finite logit, normalize to NaN.
    if case == "positive_infinity_later" or case == "large_negative_i64" or (case == "large_negative_i32" and bits == 32):
        request.applymarker(pytest.mark.xfail(strict=True, reason="FBGEMM integer-typed block maximum gives NaN"))
    values = np.repeat(np.array(SPECIAL_CASES[case], dtype=np.float32).reshape(-1, 1), d, axis=1)
    offsets = np.array([0] if case == "zero_segments" else [0, 0, len(values), len(values)], dtype=f"i{bits // 8}")
    check_case(fbgemm_environment, f"{case}-{dtype}-i{bits}-d{d}", values, offsets, dtype, with_hint=True)


@pytest.mark.parametrize("dtype", ["f32", "bf16"])
@pytest.mark.parametrize("bits", [32, 64])
def test_fbgemm_long_small_mass_accumulation(fbgemm_environment, dtype, bits):
    values = np.full((131073, 1), np.float32(np.log(np.float32(0.00013))), np.float32)
    values[0] = 0
    offsets = np.array([0, len(values)], dtype=f"i{bits // 8}")
    check_case(fbgemm_environment, f"long-small-mass-{dtype}-i{bits}", values, offsets, dtype)


@pytest.mark.parametrize("dtype", ["f32", "bf16"])
@pytest.mark.parametrize("bits", [32, 64])
@pytest.mark.parametrize("case", ["mixed", "long", "special"])
def test_fbgemm_segment_sum_at_d1(fbgemm_environment, dtype, bits, case):
    torch, _ = fbgemm_environment
    if case == "mixed":
        offsets = np.array([0, 0, 1, 33, 4130, 4130, 4161], dtype=f"i{bits // 8}")
        values = np.random.default_rng(SEED).normal(size=(4161, 1)).astype(np.float32)
    elif case == "long":
        values = np.full((131073, 1), 0.1, np.float32)
        offsets = np.array([0, len(values)], dtype=f"i{bits // 8}")
    else:
        values = np.array([[np.inf], [1], [np.nan], [2], [-np.inf], [np.inf]], np.float32)
        offsets = np.array([0, 2, 4, 6], dtype=f"i{bits // 8}")
    if dtype == "bf16":
        values = quantize_bf16(values)
    tensor = torch.from_numpy(values).to("cuda", dtype=torch.float32 if dtype == "f32" else torch.bfloat16)
    output = adapter.call(tensor, torch.from_numpy(offsets).to("cuda"), "sum")
    assert output.shape == (len(offsets) - 1, 1) and output.dtype == tensor.dtype
    assert_matches(output.float().cpu().numpy(), values, offsets, operation="sum", input_dtype=dtype)
