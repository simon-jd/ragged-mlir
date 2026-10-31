"""Independent negative cases derived from docs/classified-abi.md.

Synthetic manifests test validation only. They are never GPU execution or
numeric correctness evidence.
"""
from copy import deepcopy
from pathlib import Path

import pytest

from ragged import Artifact, CacheKey, CompiledProgram, ProgramSpec
from ragged.errors import CacheMismatch, ValidationError
from ragged.layout import resolve_layout


THREE_PASS = ("three_pass",) * 3
MIXED = ("online", "online", "three_pass")
ONLINE = ("online",) * 3


def classified_manifest(feature_dim=7, schedule=THREE_PASS):
    spec = ProgramSpec("softmax", "f32", feature_dim, debug_visits=True)
    key = CacheKey.from_spec(spec, llvm_commit="independent-contract-test",
                             ptxas_version="independent-contract-test", compiler_sha256="2" * 64)
    workspace = [
        {"name": "class_counts", "shape": [3], "dtype": "i64"},
        {"name": "segment_ids", "shape": [3, "B"], "dtype": "i64"},
        {"name": "partials", "shape": ["B", 8 * feature_dim], "dtype": "f32"},
        *([{"name": "partial_sums", "shape": ["B", 8 * feature_dim], "dtype": "f32"}] if schedule[2] == "online" else []),
        {"name": "max_stats", "shape": ["B", feature_dim], "dtype": "f32"},
        {"name": "sum_stats", "shape": ["B", feature_dim], "dtype": "f32"},
        {"name": "visits", "shape": ["B"], "dtype": "i32"},
    ]
    args = ["values", "offsets", "out", *[entry["name"] for entry in workspace]]
    kernels = []

    def kernel(role, grid, threads=128, segment_class=None):
        entry = {"name": "review_" + role, "role": role, "grid": grid,
                 "block": [threads, 1, 1], "args": list(args), "shared_bytes": 0,
                 "static_shared_bytes": 512 if threads == 128 else 0}
        if segment_class is not None:
            entry["segment_class"] = segment_class
        kernels.append(entry)

    kernel("init", [1, 1, 1])
    kernel("classify", ["ceil_B_128", 1, 1])
    feature_tiles = (feature_dim + 31) // 32
    for (name, threads), fusion in zip((("warp", 32), ("cta", 128)), schedule):
        for phase in ("online",) if fusion == "online" else ("max_stats", "sum_stats", "normalize"):
            kernel(name + "_" + phase, ["min_B_resident", feature_tiles, 1], threads, name)
    split_phases = ((("partial_online", 8), ("combine_online", 1), ("normalize", 8)) if schedule[2] == "online" else
                    (("partial_max", 8), ("combine_max_stats", 1), ("partial_exp_sum", 8), ("combine_sum_stats", 1),
                     ("normalize", 8)))
    for phase, parts in split_phases:
        kernel("split_" + phase, ["min_B_resident", feature_tiles, parts], segment_class="split")
    metadata = {"schema": 1, "abi": "mlir-memref64-v1", "dispatch_abi": "classified-v2",
        "backend": "classified-persistent", "classes": ["warp", "cta", "split"],
        "thresholds": [32, 4096], "split_parts": 8, "debug_visits": True,
        "fusion": schedule[0] if len(set(schedule)) == 1 else "mixed",
        "alias_safe": schedule == THREE_PASS, "chip": "sm_89",
        "workspace": workspace, "kernels": kernels, "compiler_sha256": key.compiler_sha256,
        "toolchain": {name: getattr(key, name) for name in
                      ("llvm_commit", "ptxas_version", "dialect_version", "chip", "features")}}
    if schedule != THREE_PASS:
        metadata["class_fusion"] = list(schedule)
    return spec, key, metadata


def program_for(metadata, feature_dim=7):
    spec, key, _ = classified_manifest(feature_dim)
    return CompiledProgram(spec, Artifact(Path("independent-host-contract"), key,
                                          b"\x7fELFnot-launched", deepcopy(metadata)))


def test_protocol_manifest_is_accepted():
    _, _, metadata = classified_manifest()
    program_for(metadata)


def test_mixed_schedule_manifest_is_accepted():
    _, _, metadata = classified_manifest(schedule=MIXED)
    assert [kernel["role"] for kernel in metadata["kernels"]][2:5] == ["warp_online", "cta_online", "split_partial_max"]
    program_for(metadata)


def test_absent_class_fusion_means_a_uniform_schedule():
    _, _, metadata = classified_manifest()
    assert "class_fusion" not in metadata
    metadata["class_fusion"] = list(THREE_PASS)
    program_for(metadata)


def test_online_split_manifest_is_accepted():
    _, _, metadata = classified_manifest(schedule=ONLINE)
    assert [kernel["role"] for kernel in metadata["kernels"]][-3:] == [
        "split_partial_online", "split_combine_online", "split_normalize"]
    assert [kernel["grid"][2] for kernel in metadata["kernels"]][-3:] == [8, 1, 8]
    program_for(metadata)


@pytest.mark.parametrize("mutation", ["missing_partial_sums", "narrow_partial_sums", "three_pass_split_roles",
                                      "combine_split_z", "partial_sums_without_online_split"])
def test_online_split_mismatch_fails(mutation):
    _, _, metadata = classified_manifest(schedule=ONLINE)
    kernels = metadata["kernels"]
    if mutation == "missing_partial_sums":
        metadata["workspace"] = [entry for entry in metadata["workspace"] if entry["name"] != "partial_sums"]
        for kernel in kernels:
            kernel["args"].remove("partial_sums")
    elif mutation == "narrow_partial_sums":
        metadata["workspace"][3]["shape"] = ["B", 7]
    elif mutation == "three_pass_split_roles":
        metadata["class_fusion"], metadata["fusion"] = list(MIXED), "mixed"
    elif mutation == "combine_split_z":
        kernels[-2]["grid"][2] = 8
    elif mutation == "partial_sums_without_online_split":
        _, _, metadata = classified_manifest(schedule=MIXED)
        metadata["workspace"].insert(3, {"name": "partial_sums", "shape": ["B", 56], "dtype": "f32"})
        for kernel in metadata["kernels"]:
            kernel["args"].insert(6, "partial_sums")
    with pytest.raises(CacheMismatch):
        program_for(metadata)


@pytest.mark.parametrize("mutation", ["split_roles_disagree", "summary_disagrees", "missing_schedule", "short_schedule",
                                      "unknown_entry", "roles_disagree", "missing_statistics", "alias_safe"])
def test_mixed_schedule_mismatch_fails(mutation):
    _, _, metadata = classified_manifest(schedule=MIXED)
    if mutation == "split_roles_disagree":
        metadata["class_fusion"], metadata["fusion"] = list(ONLINE), "online"
    elif mutation == "summary_disagrees":
        metadata["fusion"] = "three_pass"
    elif mutation == "missing_schedule":
        del metadata["class_fusion"]
    elif mutation == "short_schedule":
        metadata["class_fusion"] = ["online", "three_pass"]
    elif mutation == "unknown_entry":
        metadata["class_fusion"] = ["online", "fused", "three_pass"]
    elif mutation == "roles_disagree":
        metadata["class_fusion"] = ["three_pass", "online", "three_pass"]
    elif mutation == "missing_statistics":
        metadata["workspace"] = [entry for entry in metadata["workspace"] if not entry["name"].endswith("_stats")]
        for kernel in metadata["kernels"]:
            kernel["args"] = [name for name in kernel["args"] if not name.endswith("_stats")]
    elif mutation == "alias_safe":
        metadata["alias_safe"] = True
    with pytest.raises(CacheMismatch):
        program_for(metadata)


def test_reductions_have_no_class_fusion():
    from ragged.optimizations import class_fusion
    assert class_fusion({"fusion": "not_applicable", "classes": ["cta"]}) is None
    with pytest.raises(CacheMismatch):
        class_fusion({"fusion": "not_applicable", "classes": ["cta"], "class_fusion": ["online"]})


@pytest.mark.parametrize("index,field,value", [
    (0, "dtype", "f32"), (0, "shape", [3, 1]), (0, "shape", [True]),
    (1, "dtype", "i32"), (1, "shape", ["B", 3]),
    (2, "dtype", "bf16"), (2, "shape", ["B", 7]), (2, "shape", ["B", 8, 7]),
    (2, "shape", ["B", 55]), (2, "shape", ["B", 2**63]),
    (3, "shape", ["N", 7]), (3, "shape", ["B", 0]),
    (4, "name", "unexpected_scratch"), (4, "shape", ["B", "D"]),
    (5, "dtype", "i64"), (5, "shape", ["B", 1]),
])
def test_workspace_contract_mismatch_fails(index, field, value):
    _, _, metadata = classified_manifest()
    metadata["workspace"][index][field] = value
    with pytest.raises(CacheMismatch):
        program_for(metadata)


@pytest.mark.parametrize("field,value", [
    ("dispatch_abi", "classified-v1"), ("split_parts", 4), ("split_parts", True),
    ("debug_visits", False), ("debug_visits", 1),
    ("classes", ["warp", "warp", "split"]), ("classes", ["cta", "warp", "split"]),
    ("classes", []), ("thresholds", [32, 4097]),
])
def test_dispatch_identity_mismatch_fails(field, value):
    _, _, metadata = classified_manifest()
    metadata[field] = value
    with pytest.raises(CacheMismatch):
        program_for(metadata)


@pytest.mark.parametrize("dimension", ["B*8", "ceil_B_64", "min_B_128", "__import__('os')", True, -1, 1.5])
def test_grid_has_no_expression_language(dimension):
    _, _, metadata = classified_manifest()
    metadata["kernels"][1]["grid"][0] = dimension
    with pytest.raises(CacheMismatch):
        program_for(metadata)


@pytest.mark.parametrize("mutation", ["swap_init", "drop_classify", "drop_partial",
                                      "early_normalize", "wrong_split_z", "wrong_class",
                                      "wrong_argument_order"])
def test_unsafe_launch_sequences_fail_before_driver_load(mutation):
    _, _, metadata = classified_manifest()
    kernels = metadata["kernels"]
    if mutation == "swap_init":
        kernels[0], kernels[1] = kernels[1], kernels[0]
    elif mutation == "drop_classify":
        kernels.pop(1)
    elif mutation == "drop_partial":
        kernels.pop(8)
    elif mutation == "early_normalize":
        kernels[10], kernels[12] = kernels[12], kernels[10]
    elif mutation == "wrong_split_z":
        kernels[8]["grid"][2] = 1
    elif mutation == "wrong_class":
        kernels[8]["segment_class"] = "warp"
    else:
        kernels[-1]["args"][-1], kernels[-1]["args"][-2] = kernels[-1]["args"][-2], kernels[-1]["args"][-1]
    with pytest.raises(CacheMismatch):
        program_for(metadata)


@pytest.mark.parametrize("d", [128, 1024])
def test_worker_grid_cannot_omit_feature_tiles(d):
    _, _, metadata = classified_manifest(d)
    program_for(metadata, d)
    metadata["kernels"][2]["grid"][1] = 1
    with pytest.raises(CacheMismatch):
        program_for(metadata, d)


@pytest.mark.parametrize("b,expected", [(1, 1), (127, 1), (128, 1), (129, 2), (256, 2), (257, 3)])
def test_integer_ceiling_and_persistent_grid_boundaries(b, expected):
    _, _, metadata = classified_manifest()
    workers = {kernel["name"]: 200 + index for index, kernel in enumerate(metadata["kernels"][2:])}
    workspaces, grids, nbytes = resolve_layout(metadata, n=0, b=b, d=7, workers=workers)
    assert grids[1] == (expected, 1, 1)
    assert [grid[0] for grid in grids[2:]] == [min(b, 200 + index) for index in range(len(grids) - 2)]
    # Typed bytes: counters, three worklists, eight partials, two statistics,
    # and one i32 visit counter per segment.
    assert nbytes == 24 + b * (24 + 8 * 7 * 4 + 2 * 7 * 4 + 4)
    assert sum(item.nbytes for item in workspaces) == nbytes


@pytest.mark.parametrize("shape,dtype", [(["B", 8192], "f32"), (["B"], "i64")])
def test_dynamic_workspace_byte_overflow_is_rejected(shape, dtype):
    metadata = {"workspace": [{"name": "overflow", "shape": shape, "dtype": dtype}],
                "kernels": [{"grid": [1, 1, 1]}]}
    with pytest.raises(ValidationError):
        resolve_layout(metadata, n=0, b=2**62, d=1024)


def test_total_workspace_bytes_cannot_overflow_across_individually_valid_buffers():
    metadata = {"workspace": [{"name": name, "shape": ["B"], "dtype": "f32"}
                               for name in ("first", "second")],
                "kernels": [{"grid": [1, 1, 1]}]}
    with pytest.raises(ValidationError):
        resolve_layout(metadata, n=0, b=2**60, d=1)


@pytest.mark.parametrize("b", [128 * (2**31 - 1) + 1, 2**63 - 1])
def test_classifier_grid_overflow_does_not_wrap(b):
    metadata = {"dispatch_abi": "classified-v2", "workspace": [],
                "kernels": [{"grid": ["ceil_B_128", 1, 1]}]}
    with pytest.raises(ValidationError, match="grid"):
        resolve_layout(metadata, n=0, b=b, d=1)


@pytest.mark.parametrize("workers", [None, {}, {"warp_max_stats": 0}, {"warp_max_stats": True}])
def test_persistent_grid_requires_a_measured_worker_count(workers):
    _, _, metadata = classified_manifest()
    with pytest.raises(CacheMismatch, match="worker count"):
        resolve_layout(metadata, n=0, b=4, d=7, workers=workers)


@pytest.mark.parametrize("field,value", [("n", -1), ("b", True), ("b", 2**63), ("d", 0), ("d", 1025)])
def test_invalid_host_dimensions_fail_preflight(field, value):
    dimensions = {"n": 0, "b": 1, "d": 7}
    dimensions[field] = value
    _, _, metadata = classified_manifest()
    with pytest.raises(ValidationError):
        resolve_layout(metadata, **dimensions)
