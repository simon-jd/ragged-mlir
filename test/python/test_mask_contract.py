"""Independent additive-mask oracle, cache ABI and public Torch shape contracts."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from oracle.masked import assert_masked_softmax_matches, masked_logits
from oracle.reference import TOLERANCES, quantize_bf16, segment_softmax
from ragged import Artifact, CacheKey, CacheMismatch, CompiledProgram, ProgramSpec, ValidationError


def make_program(spec=None, *, metadata_edit=None):
    spec = spec or ProgramSpec("softmax", feature_dim=7, strategy="naive", mask=True)
    key = CacheKey.from_spec(spec, llvm_commit="independent-host-test", ptxas_version="independent-host-test",
                             compiler_sha256="9" * 64)
    args = ["values", "offsets", "out"] + (["mask"] if spec.mask else []) + ["max_stats", "sum_stats"]
    metadata = {
        "schema": 1, "abi": "mlir-memref64-v1", "compiler_sha256": key.compiler_sha256,
        "mask": spec.mask, "classes": ["cta"], "fusion": "three_pass", "alias_safe": True,
        "toolchain": {name: getattr(key, name) for name in
                      ("llvm_commit", "ptxas_version", "dialect_version", "chip", "features")},
        "workspace": [{"name": name, "shape": ["B", 7], "dtype": "f32"}
                      for name in ("max_stats", "sum_stats")],
        "kernels": [{"name": name, "args": args.copy(), "block": [128, 1, 1], "grid": ["B", 1, 1]}
                    for name in ("max", "sum", "normalize")],
    }
    if metadata_edit:
        metadata_edit(metadata)
    return CompiledProgram(spec, Artifact(Path("independent-mask-host-only"), key, b"\x7fELFhost-contract", metadata))


def test_oracle_mask_is_added_after_scale_with_no_bf16_intermediate_rounding():
    values = quantize_bf16(np.full((2, 1), 65536, dtype=np.float32))
    mask = np.array([[-32764], [-32772]], dtype=np.float32)
    np.testing.assert_array_equal(masked_logits(values, mask, scale=0.5), [[4], [-4]])
    reference = segment_softmax(np.array([[4], [-4]], dtype=np.float32), np.array([0, 2]))
    actual = quantize_bf16(reference.astype(np.float32))
    assert_masked_softmax_matches(actual, values, np.array([0, 2]), mask, scale=0.5, input_dtype="bf16")
    with pytest.raises(AssertionError):
        assert_masked_softmax_matches(np.full((2, 1), 0.5), values, np.array([0, 2]), mask,
                                     scale=0.5, input_dtype="bf16")


def test_oracle_requires_two_f32_roundings_instead_of_fused_multiply_add():
    scale = np.nextafter(np.float32(1), np.float32(2))
    values = np.array([[1e20], [0]], dtype=np.float32)
    mask = -np.multiply(values, scale, dtype=np.float32)
    np.testing.assert_array_equal(masked_logits(values, mask, scale=float(scale)), np.zeros_like(values))
    # The mathematical double-precision product retains a substantial residual;
    # evaluating everything in f64 would be the wrong input transform here.
    assert abs(float(values[0, 0]) * float(scale) + float(mask[0, 0])) > 1e10


def test_oracle_preserves_nan_infinity_and_scale_zero_arithmetic():
    values = np.array([[np.inf, -np.inf, 1, 2]], dtype=np.float32)
    mask = np.array([[0, 0, -np.inf, np.nan]], dtype=np.float32)
    logits = masked_logits(values, mask, scale=0)
    np.testing.assert_array_equal(np.isnan(logits), [[True, True, False, True]])
    assert np.isneginf(logits[0, 2])
    fully_masked = np.full((2, 1), -np.inf, dtype=np.float32)
    assert_masked_softmax_matches(np.full((2, 1), np.nan), np.zeros((2, 1), np.float32),
                                 np.array([0, 2]), fully_masked)
    with pytest.raises(AssertionError, match="NaN"):
        assert_masked_softmax_matches(np.zeros((2, 1)), np.zeros((2, 1), np.float32),
                                     np.array([0, 2]), fully_masked)


def test_oracle_uses_existing_fixed_tolerances():
    assert TOLERANCES == {"f32": (2e-5, 2e-6), "bf16": (2e-2, 2e-3)}
    with pytest.raises(AssertionError):
        assert_masked_softmax_matches(np.array([[0.5001], [0.4999]]),
                                     np.zeros((2, 1), np.float32), np.array([0, 2]),
                                     np.zeros((2, 1), np.float32))


@pytest.mark.parametrize("value", [1, 0, "true", None, [], {}])
def test_mask_presence_is_a_strict_static_boolean(value):
    with pytest.raises(ValidationError):
        ProgramSpec("softmax", mask=value)


@pytest.mark.parametrize("op", ["sum", "max"])
def test_mask_is_not_a_reduction_operand(op):
    with pytest.raises(ValidationError):
        ProgramSpec(op, mask=True)


def test_mask_changes_ir_cache_identity_and_has_fixed_f32_type():
    masked = ProgramSpec("softmax", "bf16", 7, mask=True)
    unmasked = ProgramSpec("softmax", "bf16", 7)
    assert masked.ir_hash != unmasked.ir_hash
    assert ProgramSpec.from_dict(masked.to_dict()) == masked
    assert "%mask: memref<?x7xf32>" in masked.to_mlir()
    assert '"ragged.segment_softmax"(%values, %offsets, %out, %mask)' in masked.to_mlir()
    assert "%mask" not in unmasked.to_mlir()


@pytest.mark.parametrize("flag", [True, False, 1, "true"])
def test_manifest_mask_cannot_be_rebound_to_another_source(flag):
    if flag is True:
        assert make_program().spec.mask
    else:
        with pytest.raises(CacheMismatch):
            make_program(metadata_edit=lambda metadata: metadata.update(mask=flag))


def test_legacy_no_mask_manifest_does_not_authorize_a_mask_program():
    with pytest.raises(CacheMismatch):
        make_program(metadata_edit=lambda metadata: metadata.pop("mask"))
    assert not make_program(ProgramSpec("softmax", feature_dim=7, strategy="naive"),
                            metadata_edit=lambda metadata: metadata.pop("mask")).spec.mask


@pytest.mark.parametrize("mutation", ["missing", "mask-before-out", "workspace-before-mask", "extra", "as-workspace"])
def test_mask_descriptor_cannot_be_missing_or_move_within_main_abi(mutation):
    def change(metadata):
        for kernel in metadata["kernels"]:
            if mutation == "missing": kernel["args"].remove("mask")
            elif mutation == "mask-before-out": kernel["args"][2:4] = ["mask", "out"]
            elif mutation == "workspace-before-mask": kernel["args"][3:5] = ["max_stats", "mask"]
            elif mutation == "extra": kernel["args"].append("mask")
        if mutation == "as-workspace":
            metadata["workspace"].insert(0, {"name": "mask", "dtype": "f32", "shape": ["N", 7]})
    with pytest.raises(CacheMismatch):
        make_program(metadata_edit=change)


@pytest.fixture(scope="module")
def torch_ops():
    torch = pytest.importorskip("torch")
    from ragged.torch_adapter import register_torch_ops
    return torch, register_torch_ops()


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("mode", ["validated", "checked"])
def test_mask_real_meta_fullgraph_preserves_the_mask_operand(torch_ops, dtype, mode):
    torch, ops = torch_ops
    graphs = []
    def backend(graph, _):
        graphs.append(graph)
        return graph.forward
    def function(values, offsets, mask):
        return ops["segment_softmax"](values, offsets, "independent-shape-only", mode, mask)
    compiled = torch.compile(function, backend=backend, fullgraph=True)
    values = torch.empty((17, 7), dtype=getattr(torch, dtype), device="meta")
    offsets = torch.empty((5,), dtype=torch.int64, device="meta")
    mask = torch.empty((17, 7), dtype=torch.float32, device="meta")
    result = compiled(values, offsets, mask)
    assert result.shape == values.shape and result.dtype == values.dtype
    assert len(graphs) == 1
    calls = [node for node in graphs[0].graph.nodes if node.op == "call_function"
             and "ragged_mlir.segment_softmax" in str(node.target)]
    assert len(calls) == 1 and len(calls[0].args) == 5
    assert calls[0].args[3] == mode and calls[0].args[4].op == "placeholder"


@pytest.mark.parametrize("bad", ["broadcast", "rank", "bf16", "f64", "noncontiguous", "device", "grad"])
def test_mask_fake_checks_shape_dtype_device_layout_and_grad(torch_ops, bad):
    torch, ops = torch_ops
    values = torch.empty((17, 7), device="meta")
    offsets = torch.empty((5,), dtype=torch.int64, device="meta")
    mask = torch.empty_like(values)
    if bad == "broadcast": mask = torch.empty((17, 1), device="meta")
    elif bad == "rank": mask = torch.empty((1, 17, 7), device="meta")
    elif bad == "bf16": mask = mask.to(dtype=torch.bfloat16)
    elif bad == "f64": mask = mask.to(dtype=torch.float64)
    elif bad == "noncontiguous": mask = torch.empty((7, 17), device="meta").t()
    elif bad == "device": mask = torch.empty((17, 7), device="cpu")
    elif bad == "grad": mask.requires_grad_(True)
    with pytest.raises((ValidationError, RuntimeError)):
        ops["segment_softmax"](values, offsets, "independent-shape-only", "validated", mask)
