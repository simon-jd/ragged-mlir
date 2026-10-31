"""Independent hostile-manifest and integer-boundary checks for offsets ABI."""

from copy import deepcopy

import pytest

from ragged import CacheMismatch, Hints, ProgramSpec, ValidationError
from ragged.layout import I64_MAX, resolve_layout
from ragged.validation import (raise_for_status, validate_validation_hints,
                               validate_validation_metadata)


CONDITIONS = ["first_nonzero", "final_mismatch", "negative", "above_n",
              "nonmonotone", "below_min_len", "above_max_len"]


def validation_manifest():
    return {
        "abi": "offsets-check-v1", "conditions": list(CONDITIONS), "sentinel": I64_MAX,
        "min_len": None, "max_len": None,
        "workspace": [{"name": "validation_status", "dtype": "i64", "shape": [7]}],
        "kernels": [
            {"name": "offsets_init", "role": "init", "grid": [1, 1, 1], "block": [128, 1, 1],
             "args": ["values", "offsets", "validation_status"], "shared_bytes": 0,
             "static_shared_bytes": 0},
            {"name": "offsets_check", "role": "check", "grid": ["ceil_B1_128", 1, 1],
             "block": [128, 1, 1], "args": ["values", "offsets", "validation_status"],
             "shared_bytes": 0, "static_shared_bytes": 0},
        ],
    }


@pytest.mark.parametrize("b,blocks", [(0, 1), (1, 1), (126, 1), (127, 1), (128, 2),
                                      (255, 2), (256, 3), (128 * (2**31 - 1) - 1, 2**31 - 1)])
def test_validation_grid_counts_the_initial_offset(b, blocks):
    metadata = validation_manifest()
    validate_validation_metadata(metadata)
    layouts, grids, total = resolve_layout(metadata, n=0, b=b, d=7)
    assert grids == ((1, 1, 1), (blocks, 1, 1))
    assert total == 56 and layouts[0].shape == (7,) and layouts[0].dtype == "i64"


@pytest.mark.parametrize("b", [128 * (2**31 - 1), I64_MAX - 1, I64_MAX])
def test_validation_grid_rejects_overflow_without_wrapping(b):
    with pytest.raises(ValidationError):
        resolve_layout(validation_manifest(), n=0, b=b, d=1)


@pytest.mark.parametrize("path,value", [
    (("abi",), "offsets-check-v2"), (("sentinel",), -1), (("sentinel",), 2**63),
    (("sentinel",), True), (("conditions",), CONDITIONS[::-1]),
    (("workspace", 0, "dtype"), "i32"), (("workspace", 0, "shape"), [8]),
    (("workspace", 0, "shape"), [7.0]), (("workspace", 0, "shape"), ["B"]),
    (("min_len",), -1), (("min_len",), True), (("max_len",), 2**63),
    (("max_len",), "8193"), (("kernels", 0, "grid"), [0, 1, 1]),
    (("kernels", 0, "grid"), [True, 1, 1]),
    (("kernels", 1, "grid"), ["ceil_B_128", 1, 1]),
    (("kernels", 1, "grid"), ["ceil((B+1)/128)", 1, 1]),
    (("kernels", 1, "block"), [64, 1, 1]),
    (("kernels", 1, "args"), ["values", "offsets", "out"]),
    (("kernels", 1, "role"), "init"), (("kernels", 1, "name"), "offsets_init"),
    (("kernels", 0, "shared_bytes"), 8), (("kernels", 1, "static_shared_bytes"), True),
])
def test_malformed_validation_manifest_is_rejected(path, value):
    metadata = deepcopy(validation_manifest())
    target = metadata
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(CacheMismatch):
        validate_validation_metadata(metadata)


def test_validation_kernel_cannot_alias_main_kernel_name():
    with pytest.raises(CacheMismatch):
        validate_validation_metadata(validation_manifest(), main_names=["offsets_check"])


@pytest.mark.parametrize("field", ["min_len", "max_len"])
def test_validation_bounds_are_bound_to_the_source_even_for_claimed_checked(field):
    spec = ProgramSpec("sum", hints=Hints(min_len=1, max_len=8193, offsets_checked=True))
    metadata = validation_manifest()
    metadata.update(min_len=1, max_len=8193)
    validate_validation_metadata(metadata)
    validate_validation_hints(spec, metadata)
    metadata[field] = None
    with pytest.raises(CacheMismatch):
        validate_validation_hints(spec, metadata)


@pytest.mark.parametrize("slot,index", [(0, 0), (1, 8), (2, 3), (3, 5), (4, 7), (5, 0), (6, 7)])
def test_each_status_slot_retains_its_index_domain(slot, index):
    status = [I64_MAX] * 7
    status[slot] = index
    with pytest.raises(ValidationError) as caught:
        raise_for_status(status, b=8)
    assert caught.value.errors == {CONDITIONS[slot]: index}
    assert (caught.value.condition, caught.value.index) == (CONDITIONS[slot], index)


def test_all_conditions_survive_host_decode_in_protocol_order():
    status = [0, 8, 1, 3, 0, 5, 6]
    with pytest.raises(ValidationError) as caught:
        raise_for_status(status, b=8)
    assert caught.value.errors == dict(zip(CONDITIONS, status))
    assert (caught.value.condition, caught.value.index) == ("first_nonzero", 0)
    assert str(caught.value).count(" at index ") == 7


@pytest.mark.parametrize("slot,index", [(0, 1), (1, 7), (2, 9), (3, 9), (4, 8), (5, 8), (6, 8),
                                       (2, -1), (2, True), (2, 2**63), (2, 1.0)])
def test_impossible_device_status_is_rejected(slot, index):
    status = [I64_MAX] * 7
    status[slot] = index
    with pytest.raises(ValidationError) as caught:
        raise_for_status(status, b=8)
    assert caught.value.errors == {} and caught.value.condition is None


@pytest.mark.parametrize("slot", [4, 5, 6])
def test_zero_batch_has_no_segment_error_index(slot):
    status = [I64_MAX] * 7
    status[slot] = 0
    with pytest.raises(ValidationError):
        raise_for_status(status, b=0)


def test_empty_status_sentinel_means_success_for_zero_batch():
    assert raise_for_status([I64_MAX] * 7, b=0) is None
