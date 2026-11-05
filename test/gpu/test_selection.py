"""The adaptive program: both schedules are correct, the choice follows the
cost model, and vector access rejects tensors it cannot address aligned."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from bench.distributions import make_lengths, offsets_from_lengths
from oracle import assert_matches
from ragged import Optimizations, ProgramSpec, ValidationError
from ragged.selection import default_model
from test_correctness import gpu_environment, prepared
from test_scheduling import run


def case(torch, spec, distribution, seed=3):
    lengths = make_lengths(distribution, 512, seed=seed, total_elements=512 * 64)
    offsets = offsets_from_lengths(lengths, spec.offsets_dtype)
    values = prepared(np.random.default_rng(seed).standard_normal((int(offsets[-1]), spec.feature_dim)), spec.dtype)
    dtype = torch.float32 if spec.dtype == "f32" else torch.bfloat16
    return lengths, values, offsets, torch.from_numpy(values).to("cuda", dtype), torch.from_numpy(offsets).cuda()


@pytest.mark.parametrize("op", ["sum", "max", "softmax"])
@pytest.mark.parametrize("dtype,d", [("f32", 1), ("f32", 128), ("bf16", 128)])
@pytest.mark.parametrize("distribution", ["uniform", "long_tail", "burst"])
def test_both_schedules_match_the_oracle_and_auto_follows_the_model(gpu_environment, op, dtype, d, distribution):
    torch, compile_spec, _, _ = gpu_environment
    program = compile_spec(ProgramSpec(op, dtype, d))
    spec = program.spec   # for the chip under test
    assert program.naive is not None and program.naive.spec == replace(spec, strategy="naive")
    lengths, values, offsets, device_values, device_offsets = case(torch, spec, distribution)
    for select in ("classified", "naive"):
        output, report = run(torch, program, device_values, device_offsets, select=select)
        assert report.selected == select and report.selection == {"forced": True}
        assert ("classify" in " ".join(report.kernel_names)) == (select == "classified")
        assert_matches(output.float().cpu().numpy(), values, offsets, operation=op, input_dtype=dtype)
    longest = int(lengths.max())
    # The runtime counts the bytes beyond this device's L2, not the calibration GPU's.
    l2_mib = torch.cuda.get_device_properties(device_values.device).L2_cache_size / 2**20
    expected, _ = default_model().choose(op, len(values), len(lengths), d, longest,
                                         program._metadata["d_tile"] * program._metadata["d_vector"],
                                         2 if dtype == "bf16" else 4, l2_mib=l2_mib)
    output, report = run(torch, program, device_values, device_offsets, max_len=longest)
    assert report.selected == expected and report.selection["max_len"] == longest
    assert report.selection["l2_mib"] == l2_mib
    assert_matches(output.float().cpu().numpy(), values, offsets, operation=op, input_dtype=dtype)
    _, report = run(torch, program, device_values, device_offsets)
    assert report.selected == "classified"


def test_hinted_single_class_programs_have_no_companion(gpu_environment):
    torch, compile_spec, _, _ = gpu_environment
    from ragged import Hints
    program = compile_spec(ProgramSpec("sum", "f32", 128, hints=Hints(min_len=64, max_len=64)))
    assert program.naive is None and not program.selects


@pytest.mark.parametrize("name", ["values", "out"])
def test_vector_access_rejects_misaligned_tensors_and_scalar_access_accepts_them(gpu_environment, name):
    torch, compile_spec, _, _ = gpu_environment
    spec = ProgramSpec("sum", "f32", 128)
    lengths, values, offsets, device_values, device_offsets = case(torch, spec, "uniform")
    shifted = torch.empty(device_values.numel() + 1, device="cuda")[1:].view_as(device_values)
    shifted.copy_(device_values)
    out = torch.empty(len(lengths) * 128 + 1, device="cuda")[1:].view(len(lengths), 128)
    arguments = {"values": shifted if name == "values" else device_values,
                 "out": out if name == "out" else torch.empty((len(lengths), 128), device="cuda")}
    for select in ("classified", "naive"):
        with pytest.raises(ValidationError, match=f"{name} must be 16-byte aligned"):
            compile_spec(spec).run(arguments["values"], device_offsets, arguments["out"], select=select)
    scalar = compile_spec(replace(spec, optimizations=Optimizations(vectorize=False)))
    scalar.run(arguments["values"], device_offsets, arguments["out"])
    torch.cuda.synchronize()
    assert_matches(arguments["out"].cpu().numpy(), values, offsets, operation="sum", input_dtype="f32")
