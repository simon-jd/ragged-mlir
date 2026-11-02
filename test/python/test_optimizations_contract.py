"""Independent optimization requests, cache binding, and host overflow checks.

Synthetic artifacts are only manifest-validation inputs, never executed code.
"""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from ragged import Artifact, CacheKey, CacheMismatch, CompiledProgram, Optimizations, ProgramSpec, ValidationError
from ragged.layout import resolve_layout

FLAGS = ("classification", "online_fusion", "warp_mapping", "split_mapping", "shuffle", "pointwise_fusion",
         "row_pipelining", "vectorize")


def manifest(spec=None):
    spec = spec or ProgramSpec("softmax", feature_dim=7, strategy="naive", mask=True,
                              optimizations=Optimizations(pointwise_fusion=False, feature_tile=2))
    key = CacheKey.from_spec(spec, llvm_commit="independent-contract", ptxas_version="independent-contract",
                             compiler_sha256="8" * 64)
    tile, vector = spec.optimizations.geometry(spec.feature_dim, spec.dtype, spec.smem_per_cta)
    pointwise = spec.mask or spec.scale != 1
    materialized = pointwise and not spec.optimizations.pointwise_fusion
    workspace = [{"name": name, "dtype": "f32", "shape": ["B", spec.feature_dim]}
                 for name in ("max_stats", "sum_stats")]
    if materialized:
        workspace.append({"name": "logits", "dtype": "f32", "shape": ["N", spec.feature_dim]})
    args = ["values", "offsets", "out"] + (["mask"] if spec.mask else []) + [w["name"] for w in workspace]
    kernels = [{"name": name, "args": list(args), "block": [128, 1, 1],
                "grid": ["B", -(-spec.feature_dim // (tile * vector)), 1]}
               for name in ("review_max", "review_sum", "review_normalize")]
    if materialized:
        kernels.insert(0, {"name": "review_materialize", "role": "materialize", "args": list(args),
                           "block": [128, 1, 1], "grid": [256, 1, 1], "static_shared_bytes": 0})
    metadata = {"schema": 1, "abi": "mlir-memref64-v1", "classes": ["cta"], "mask": spec.mask,
                "fusion": "three_pass", "alias_safe": True, "workspace": workspace, "kernels": kernels,
                "d_tile": tile, "d_vector": vector, "pointwise_materialized": materialized,
                "optimizations": {"requested": spec.optimizations.to_dict(),
                                  "effective": {name: pointwise and not materialized if name == "pointwise_fusion"
                                                else spec.optimizations.row_pipelining if name == "row_pipelining"
                                                else vector > 1 if name == "vectorize"
                                                else False for name in FLAGS}},
                "compiler_sha256": key.compiler_sha256,
                "toolchain": {name: getattr(key, name) for name in
                              ("llvm_commit", "ptxas_version", "dialect_version", "chip", "features")}}
    return spec, key, metadata


def construct(spec, key, metadata):
    return CompiledProgram(spec, Artifact(Path("independent-controls-host-only"), key,
                                          b"\x7fELFnot-executed", deepcopy(metadata)))


@pytest.mark.parametrize("name", FLAGS)
@pytest.mark.parametrize("bad", [0, 1, None, "false", [], {}])
def test_flags_are_strict_booleans(name, bad):
    with pytest.raises(ValidationError):
        Optimizations(**{name: bad})


@pytest.mark.parametrize("bad", [False, True, 0, -1, 3, 17, 33, 64, 2.0, "2"])
def test_feature_tile_is_an_optional_bounded_power_of_two(bad):
    with pytest.raises(ValidationError):
        Optimizations(feature_tile=bad)


def test_requests_are_immutable_and_roundtrip():
    options = Optimizations(shuffle=False, feature_tile=4)
    with pytest.raises(FrozenInstanceError):
        options.shuffle = True
    spec = ProgramSpec("sum", optimizations=options)
    assert ProgramSpec.from_dict(spec.to_dict()) == spec
    assert 'shuffle = false' in spec.to_mlir() and 'feature_tile = 4 : i64' in spec.to_mlir()


@pytest.mark.parametrize("options", [*[Optimizations(**{name: False}) for name in FLAGS], Optimizations(feature_tile=1)])
def test_each_request_changes_cache_identity_even_when_current_mapping_cannot_use_it(options):
    baseline = ProgramSpec("sum", feature_dim=7, strategy="naive")
    changed = replace(baseline, optimizations=options)
    assert baseline.ir_hash != changed.ir_hash
    assert CacheKey.from_spec(baseline, llvm_commit="x", ptxas_version="y") != CacheKey.from_spec(changed, llvm_commit="x", ptxas_version="y")


@pytest.mark.parametrize("kwargs", [dict(debug_visits=True, optimizations=Optimizations(classification=False)),
                                  dict(force_fusion=True, optimizations=Optimizations(online_fusion=False)),
                                  dict(optimizations={"shuffle": False})])
def test_conflicting_requests_rejected_before_compilation(kwargs):
    with pytest.raises(ValidationError):
        ProgramSpec("softmax", **kwargs)


def test_complete_materialization_contract_is_accepted():
    construct(*manifest())


@pytest.mark.parametrize("schedule,fusion,valid", [
    (["three_pass"], "three_pass", True), (["online"], "three_pass", False),
    (["three_pass", "three_pass"], "three_pass", False), (None, "mixed", False), (["three_pass"], "mixed", False),
])
def test_direct_class_fusion_has_one_uniform_entry(schedule, fusion, valid):
    spec, key, metadata = manifest()
    metadata["fusion"] = fusion
    if schedule is not None:
        metadata["class_fusion"] = schedule
    if valid:
        construct(spec, key, metadata)
    else:
        with pytest.raises(CacheMismatch):
            construct(spec, key, metadata)


@pytest.mark.parametrize("scale", [1 + 1e-8, 1 - 1e-8])
@pytest.mark.parametrize("enabled", [False, True])
def test_scale_rounding_to_f32_one_has_no_effective_pointwise_work(scale, enabled):
    source = ProgramSpec("softmax", feature_dim=7, strategy="naive",
                         optimizations=Optimizations(pointwise_fusion=enabled))
    _, _, metadata = manifest(source)
    spec = replace(source, scale=scale)
    key = CacheKey.from_spec(spec, llvm_commit="independent-contract", ptxas_version="independent-contract",
                             compiler_sha256="8" * 64)
    # Both source scalars round to exactly 1 in the public f32 IR type. The
    # emitted no-transform schedule is therefore valid for either request.
    assert metadata["pointwise_materialized"] is False
    assert metadata["optimizations"]["effective"]["pointwise_fusion"] is False
    construct(spec, key, metadata)


@pytest.mark.parametrize("name", FLAGS)
def test_effective_claims_must_match_actual_schedule(name):
    spec, key, metadata = manifest()
    metadata["optimizations"]["effective"][name] = not metadata["optimizations"]["effective"][name]
    with pytest.raises(CacheMismatch):
        construct(spec, key, metadata)


@pytest.mark.parametrize("name", (*FLAGS, "feature_tile"))
def test_source_requests_cannot_be_rebound_by_an_artifact(name):
    spec, key, metadata = manifest()
    metadata["optimizations"]["requested"][name] = 1 if name == "feature_tile" else not metadata["optimizations"]["requested"][name]
    with pytest.raises(CacheMismatch):
        construct(spec, key, metadata)


@pytest.mark.parametrize("mutation", ["missing-effective", "missing-request", "extra-request", "extra-effective", "bool-as-int",
                                      "tile-as-bool", "wrong-tile", "wrong-materialized-type", "no-options"])
def test_malformed_control_objects_are_rejected(mutation):
    spec, key, metadata = manifest()
    options = metadata["optimizations"]
    if mutation == "missing-effective": options.pop("effective")
    elif mutation == "missing-request": options["requested"].pop("shuffle")
    elif mutation == "extra-request": options["requested"]["unknown"] = True
    elif mutation == "extra-effective": options["effective"]["unknown"] = False
    elif mutation == "bool-as-int": options["effective"]["shuffle"] = 0
    elif mutation == "tile-as-bool": metadata["d_tile"] = True
    elif mutation == "wrong-tile": metadata["d_tile"] = 4
    elif mutation == "wrong-materialized-type": metadata["pointwise_materialized"] = 1
    else: metadata.pop("optimizations")
    with pytest.raises(CacheMismatch):
        construct(spec, key, metadata)


@pytest.mark.parametrize("mutation", ["no-prefix", "late-prefix", "duplicate-prefix", "grid", "block", "shared", "class",
                                      "no-logits", "logits-first", "logits-bf16", "logits-B", "logits-D", "wrong-width", "missing-argument"])
def test_materialized_arithmetic_has_a_fixed_global_boundary_and_f32_workspace(mutation):
    spec, key, metadata = manifest()
    kernels, workspace = metadata["kernels"], metadata["workspace"]
    if mutation == "no-prefix": kernels.pop(0)
    elif mutation == "late-prefix": kernels[0], kernels[1] = kernels[1], kernels[0]
    elif mutation == "duplicate-prefix": kernels[1]["role"] = "materialize"
    elif mutation == "grid": kernels[0]["grid"] = [128, 1, 1]
    elif mutation == "block": kernels[0]["block"] = [256, 1, 1]
    elif mutation == "shared": kernels[0]["static_shared_bytes"] = 4
    elif mutation == "class": kernels[0]["segment_class"] = "cta"
    elif mutation == "no-logits": workspace.pop()
    elif mutation == "logits-first": workspace.insert(0, workspace.pop())
    elif mutation == "logits-bf16": workspace[-1]["dtype"] = "bf16"
    elif mutation == "logits-B": workspace[-1]["shape"][0] = "B"
    elif mutation == "logits-D": workspace[-1]["shape"][1] = "D"
    elif mutation == "wrong-width": workspace[-1]["shape"][1] = 8
    else: kernels[-1]["args"].pop()
    with pytest.raises(CacheMismatch):
        construct(spec, key, metadata)


@pytest.mark.parametrize("d,tile", [(7, 1), (7, 2), (7, 32), (128, 1), (1024, 1), (1024, 16)])
def test_worker_geometry_covers_exact_feature_tiles(d, tile):
    spec = ProgramSpec("softmax", feature_dim=d, strategy="naive", mask=True,
                       optimizations=Optimizations(pointwise_fusion=False, feature_tile=tile))
    spec, key, metadata = manifest(spec)
    construct(spec, key, metadata)
    metadata["kernels"][-1]["grid"][1] += 1
    with pytest.raises(CacheMismatch):
        construct(spec, key, metadata)


def test_legacy_manifest_is_allowed_only_for_default_requests():
    for options in (Optimizations(), Optimizations(shuffle=False)):
        spec, key, metadata = manifest(ProgramSpec("softmax", feature_dim=7, strategy="naive", optimizations=options))
        for field in ("optimizations", "d_tile", "d_vector", "pointwise_materialized"):
            metadata.pop(field)
        if options == Optimizations():
            construct(spec, key, metadata)
        else:
            with pytest.raises(CacheMismatch):
                construct(spec, key, metadata)


@pytest.mark.parametrize("n,d", [(2**61, 1), (2**53, 1024), (2**62, 7)])
def test_logits_allocation_overflow_is_rejected_before_any_driver_load(n, d):
    metadata = {"workspace": [{"name": "logits", "shape": ["N", d], "dtype": "f32"}],
                "kernels": [{"grid": [256, 1, 1]}]}
    with pytest.raises(ValidationError, match="overflow"):
        resolve_layout(metadata, n=n, b=1, d=d)


def test_empty_logits_has_zero_bytes_but_retains_valid_strides():
    _, _, metadata = manifest()
    layouts, grids, nbytes = resolve_layout(metadata, n=0, b=3, d=7)
    assert layouts[-1].name == "logits" and layouts[-1].shape == (0, 7) and layouts[-1].nbytes == 0
    assert nbytes == 2 * 3 * 7 * 4 and grids[0] == (256, 1, 1)
