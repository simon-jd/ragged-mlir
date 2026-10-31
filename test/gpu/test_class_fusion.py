"""Per-class softmax statistics on the device.

Without length hints every class uses online statistics: one kernel each for
the warp and CTA classes, and for split a per-part kernel, a combine kernel
that merges the eight parts, and normalization. Online statistics need no
length bound, so long segments are tested in both the split and the CTA class.
Every schedule is checked against the float64 oracle; a missing CUDA device is
not a pass.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from bench.distributions import offsets_from_lengths
from ragged import Optimizations, ProgramSpec
from test_correctness import execute, gpu_environment
from test_optimizations import run_case

# Original order differs from class order; both thresholds and the split
# boundary are covered on each side.
LENGTHS = np.array([4097, 0, 1, 33, 8193, 32, 4096, 31, 4095, 0, 12000], dtype=np.int64)
ONLINE_ROLES = ["init", "classify", "warp_online", "cta_online",
                "split_partial_online", "split_combine_online", "split_normalize"]
LONG = 65536


def roles(report):
    return [name.split("_", 2)[2] for name in report.kernel_names]


@pytest.mark.parametrize("dtype", ["f32", "bf16"])
@pytest.mark.parametrize("d", [1, 7, 128])
def test_unbounded_softmax_uses_online_statistics_in_every_class(gpu_environment, dtype, d):
    offsets = offsets_from_lengths(LENGTHS)
    values = np.random.default_rng(6101).normal(scale=3, size=(int(offsets[-1]), d)).astype(np.float32)
    spec = ProgramSpec("softmax", dtype, d)
    _, report = execute(gpu_environment, spec, values, offsets)
    assert report.classes == ("warp", "cta", "split")
    # Submission order puts the longest chain first; compare the sets of roles.
    assert sorted(roles(report)) == sorted(ONLINE_ROLES)
    assert report.optimizations["effective"]["online_fusion"]
    _, baseline = execute(gpu_environment, replace(spec, optimizations=Optimizations(online_fusion=False)),
                          values, offsets)
    assert "split_partial_exp_sum" in roles(baseline) and not baseline.optimizations["effective"]["online_fusion"]


@pytest.mark.parametrize("materialized", [False, True])
def test_unbounded_softmax_fuses_scale_and_mask_per_class(gpu_environment, materialized):
    offsets = offsets_from_lengths(LENGTHS)
    rng = np.random.default_rng(6102)
    values = rng.normal(size=(int(offsets[-1]), 7)).astype(np.float32)
    mask = rng.uniform(-2, 2, size=values.shape).astype(np.float32)
    mask[offsets[3]:offsets[4], 1] = -np.inf  # a fully masked feature in a CTA segment
    mask[offsets[4]:offsets[5], 2] = -np.inf  # and in a split segment: every part is empty
    mask[offsets[0]:offsets[0] + 600, 3] = -np.inf  # one split part without mass
    spec = ProgramSpec("softmax", "f32", 7, scale=0.5, mask=True,
                       optimizations=Optimizations(pointwise_fusion=not materialized))
    report = run_case(gpu_environment, spec, values, offsets, mask)
    assert report.pointwise_materialized is materialized
    assert set(ONLINE_ROLES) <= set(roles(report))


def test_split_parts_with_distant_maxima(gpu_environment):
    # Each part's reference is 20 above the previous one, so the combine
    # rescales early parts' masses by exp(-20 * k).
    lengths = np.array([8 * 1000, 8 * 1000 + 5], dtype=np.int64)
    offsets = offsets_from_lengths(lengths)
    rng = np.random.default_rng(6105)
    values = rng.normal(size=(int(offsets[-1]), 7)).astype(np.float32)
    for begin, length in zip(offsets[:-1], lengths):
        values[begin:begin + length] += (20 * (np.arange(length) * 8 // length))[:, None].astype(np.float32)
    values[offsets[1]:offsets[2]] *= -1  # decreasing references in the second segment
    _, report = execute(gpu_environment, ProgramSpec("softmax", "f32", 7), values, offsets)
    assert "split_combine_online" in roles(report)


@pytest.mark.parametrize("dtype", ["f32", "bf16"])
def test_unbounded_inplace_softmax_keeps_three_passes(gpu_environment, dtype):
    offsets = offsets_from_lengths(LENGTHS)
    values = np.random.default_rng(6103).normal(size=(int(offsets[-1]), 7)).astype(np.float32)
    spec = ProgramSpec("softmax", dtype, 7, alias="inplace")
    _, report = execute(gpu_environment, spec, values, offsets, inplace=True)
    assert not any(role.endswith("_online") for role in roles(report))


def test_cta_class_without_split_takes_long_segments_online(gpu_environment):
    # Without split the CTA class also takes the segments above 4096 rows.
    offsets = offsets_from_lengths(LENGTHS)
    values = np.random.default_rng(6104).normal(size=(int(offsets[-1]), 16)).astype(np.float32)
    spec = ProgramSpec("softmax", "f32", 16, optimizations=Optimizations(split_mapping=False))
    _, report = execute(gpu_environment, spec, values, offsets)
    assert report.classes == ("warp", "cta") and {"warp_online", "cta_online"} <= set(roles(report))


def long_route(route):
    # Feature tile 32 leaves four row lanes per block, so each lane folds
    # 2,048 rows of a split part or 16,384 rows of a CTA segment.
    return Optimizations(split_mapping=route == "split", feature_tile=32)


@pytest.mark.parametrize("route", ["split", "cta"])
@pytest.mark.parametrize("d", [1, 128])
def test_long_segment_small_mass_accumulation_meets_fixed_tolerance(gpu_environment, route, d):
    values = np.full((LONG, d), np.float32(np.log(np.float32(0.00013))), dtype=np.float32)
    values[:4] = 0
    spec = ProgramSpec("softmax", feature_dim=d, optimizations=long_route(route))
    report = run_case(gpu_environment, spec, values, np.array([0, LONG], dtype=np.int64))
    assert f"{route}_{'partial_online' if route == 'split' else 'online'}" in roles(report)


@pytest.mark.parametrize("route", ["split", "cta"])
@pytest.mark.parametrize("step", [7.45717798622536e-9, 6.613177538383752e-5, 1e-3])
def test_long_segment_repeated_max_rescaling_meets_fixed_tolerance(gpu_environment, route, step):
    values = np.repeat((np.arange(LONG, dtype=np.float32) * np.float32(step))[:, None], 128, axis=1)
    spec = ProgramSpec("softmax", feature_dim=128, optimizations=long_route(route))
    report = run_case(gpu_environment, spec, values, np.array([0, LONG], dtype=np.int64))
    assert f"{route}_{'partial_online' if route == 'split' else 'online'}" in roles(report)
