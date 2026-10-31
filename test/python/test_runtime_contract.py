"""Independent host-side ABI validation tests, without CUDA mocks or GPU claims."""

from types import SimpleNamespace

import pytest

from ragged.errors import CacheMismatch, RuntimeUnavailable, ValidationError
from ragged.runtime import CompiledProgram, memref_arguments


def manifest():
    return {
        "schema": 1,
        "abi": "mlir-memref64-v1",
        "workspace_bytes": 0,
        "alias_safe": False,
        "kernels": [{"name": "kernel", "args": ["values", "offsets", "out"],
                     "grid": ["B", 1, 1], "block": [128, 1, 1], "shared_bytes": 0}],
    }


def test_descriptor_does_not_double_count_tensor_storage_offset():
    # data_ptr is the address of the first logical element, not storage base.
    tensor = SimpleNamespace(data_ptr=lambda: 0x2000, shape=(7, 16), stride=lambda: (16, 1))
    assert memref_arguments(tensor) == [0x2000, 0x2000, 0, 7, 16, 16, 1]


@pytest.mark.parametrize("shape,stride", [((2**63, 16), (16, 1)), ((7, 16), (2**63, 1)), ((-1, 16), (16, 1))])
def test_descriptor_rejects_out_of_range_signed_index(shape, stride):
    tensor = SimpleNamespace(data_ptr=lambda: 0x2000, shape=shape, stride=lambda: stride)
    with pytest.raises(ValidationError):
        memref_arguments(tensor)


def test_supported_manifest_validates():
    CompiledProgram.validate_metadata(manifest())


@pytest.mark.parametrize("field,value", [
    ("schema", True), ("schema", 1.0),
    ("alias_safe", 1), ("alias_safe", 0), ("alias_safe", "false"),
    ("workspace_bytes", False), ("workspace_bytes", 0.0), ("workspace_bytes", -1),
    ("abi", "bare-ptr"), ("kernels", []),
])
def test_manifest_scalar_types_are_strict(field, value):
    data = manifest()
    data[field] = value
    with pytest.raises((CacheMismatch, RuntimeUnavailable)):
        CompiledProgram.validate_metadata(data)


@pytest.mark.parametrize("field,value", [
    ("name", "kernel\x00extra"), ("name", ""),
    ("args", ["values", "out", "offsets"]),
    ("block", [1024, 2, 1]), ("block", [0, 1, 1]), ("block", [True, 1, 1]),
    ("grid", ["N+1", 1, 1]), ("grid", [False, 1, 1]), ("grid", [-1, 1, 1]),
    ("shared_bytes", True), ("shared_bytes", 49153),
])
def test_malformed_kernel_abi_is_rejected(field, value):
    data = manifest()
    data["kernels"][0][field] = value
    with pytest.raises(CacheMismatch):
        CompiledProgram.validate_metadata(data)
