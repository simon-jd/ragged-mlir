"""Independent resource evidence binding; fixtures are not device measurements."""
from copy import deepcopy
from dataclasses import replace

import pytest

from ragged import CacheMismatch, ProgramSpec, ValidationError
from ragged.resources import resource_report
from test_optimizations_contract import construct, manifest
from test_checked_contract import validation_manifest


def measured_manifest():
    spec, key, metadata = manifest()
    metadata["resources"] = {
        "status": "measured", "source": "ptxas",
        "requested": {"smem_per_cta": 49152, "regs_budget": 65536},
        "kernels": [{"name": kernel["name"], "registers_per_thread": 17, "threads_per_cta": 128,
                     "registers_per_cta": 2176, "static_shared_bytes": 0, "dynamic_shared_bytes": 0,
                     "shared_bytes_per_cta": 0, "stack_frame_bytes": 0, "spill_store_bytes": 0, "spill_load_bytes": 0}
                    for kernel in metadata["kernels"]]}
    return spec, key, metadata


@pytest.mark.parametrize("field", ["regs_budget", "smem_per_cta"])
@pytest.mark.parametrize("value", [False, True, 0, -1, 2**63, 49152.0, "65536", None])
def test_resource_budgets_are_positive_signed_i64(field, value):
    with pytest.raises(ValidationError):
        ProgramSpec("sum", **{field:value})


@pytest.mark.parametrize("field,value", [("regs_budget",4096),("smem_per_cta",512)])
def test_resource_request_is_in_ir_cache_identity_and_roundtrip(field,value):
    baseline = ProgramSpec("sum")
    changed = replace(baseline, **{field:value})
    assert baseline.ir_hash != changed.ir_hash
    assert f"{field} = {value} : i64" in changed.to_mlir()
    assert ProgramSpec.from_dict(changed.to_dict()) == changed


def test_complete_resource_structure_can_be_validated_without_execution():
    construct(*measured_manifest())


@pytest.mark.parametrize("field,value", [("status","unmeasured"),("status",True),("source","IR estimate"),
                                       ("requested",{}),("kernels",[]),("kernels",{}),("unexpected",0)])
def test_only_complete_ptxas_claims_can_back_a_cubin(field,value):
    spec,key,metadata = measured_manifest()
    metadata["resources"][field] = value
    with pytest.raises(CacheMismatch): construct(spec,key,metadata)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "reordered", "unknown", "extra-field", "missing-field",
                                      "negative", "integer-bool", "too-large", "bad-thread-count", "bad-register-product",
                                      "register-overflow", "dynamic-disagrees", "shared-sum-disagrees", "shared-overflow"])
def test_per_entry_measurements_cannot_be_partial_or_internally_inconsistent(mutation):
    spec,key,metadata = measured_manifest()
    entries = metadata["resources"]["kernels"]
    first = entries[0]
    if mutation == "missing": entries.pop()
    elif mutation == "duplicate": entries[1] = deepcopy(first)
    elif mutation == "reordered": entries[0],entries[1] = entries[1],entries[0]
    elif mutation == "unknown": first["name"] = "not_a_kernel"
    elif mutation == "extra-field": first["invented"] = 0
    elif mutation == "missing-field": first.pop("spill_load_bytes")
    elif mutation == "negative": first["stack_frame_bytes"] = -1
    elif mutation == "integer-bool": first["registers_per_thread"] = True
    elif mutation == "too-large": first["spill_store_bytes"] = 2**63
    elif mutation == "bad-thread-count": first["threads_per_cta"] = 32
    elif mutation == "bad-register-product": first["registers_per_cta"] = 17
    elif mutation == "register-overflow": first.update(registers_per_thread=2**63-1, registers_per_cta=2**63-1)
    elif mutation == "dynamic-disagrees": first["dynamic_shared_bytes"] = 4
    elif mutation == "shared-sum-disagrees": first["shared_bytes_per_cta"] = 4
    else:
        # Mutate a compute entry: materialize must have zero shared allocation.
        metadata["kernels"][1]["shared_bytes"] = 4
        entries[1].update(static_shared_bytes=2**63-1,dynamic_shared_bytes=4,shared_bytes_per_cta=2**63-1)
    with pytest.raises(CacheMismatch): construct(spec,key,metadata)


@pytest.mark.parametrize("field,value", [("regs_budget",65537),("smem_per_cta",49153)])
def test_budget_in_measurement_must_match_source_specialization(field,value):
    spec,key,metadata = measured_manifest()
    metadata["resources"]["requested"][field] = value
    with pytest.raises(CacheMismatch): construct(spec,key,metadata)


@pytest.mark.parametrize("resource", ["registers","shared"])
def test_measured_resources_over_budget_are_rejected(resource):
    spec,key,metadata = measured_manifest()
    entry = metadata["resources"]["kernels"][1]
    if resource == "registers": entry.update(registers_per_thread=513,registers_per_cta=65664)
    else: entry.update(static_shared_bytes=49153,shared_bytes_per_cta=49153)
    with pytest.raises(CacheMismatch): construct(spec,key,metadata)


def test_register_and_shared_budget_equality_is_accepted():
    spec,key,metadata = measured_manifest()
    spec = replace(spec, regs_budget=4096, smem_per_cta=512)
    key = replace(key, ir_hash=spec.ir_hash)
    metadata["resources"]["requested"] = {"regs_budget":4096,"smem_per_cta":512}
    metadata["resources"]["kernels"][1].update(registers_per_thread=32,registers_per_cta=4096,
                                                static_shared_bytes=512,shared_bytes_per_cta=512)
    construct(spec,key,metadata)


def test_actual_static_shared_measurement_is_not_replaced_by_ir_estimate():
    spec,key,metadata = measured_manifest()
    metadata["kernels"][1]["static_shared_bytes"] = 1024
    # Optimizations may eliminate IR allocations: measured cubin bytes remain
    # distinct from manifest estimates and must not be silently relabelled.
    construct(spec,key,metadata)
    assert resource_report(metadata)["kernels"][1]["static_shared_bytes"] == 0


def test_report_copies_do_not_mutate_compiled_measurements():
    _,_,metadata = measured_manifest()
    original = deepcopy(metadata["resources"])
    report = resource_report(metadata)
    report["requested"]["regs_budget"] = 1
    report["kernels"][0]["registers_per_thread"] = 1
    assert metadata["resources"] == original
    assert resource_report({}) is None


@pytest.mark.parametrize("mutation", ["missing-validator", "validator-first", "over-budget-validator", "duplicate-main-as-validator"])
def test_validation_kernels_are_in_the_same_measured_resource_gate(mutation):
    spec,key,metadata = measured_manifest()
    metadata["validation"] = validation_manifest()
    entries = metadata["resources"]["kernels"]
    entries.extend(deepcopy(entries[0]) | {"name":kernel["name"]} for kernel in metadata["validation"]["kernels"])
    construct(spec,key,metadata)
    if mutation == "missing-validator": entries.pop()
    elif mutation == "validator-first": entries.insert(0,entries.pop())
    elif mutation == "over-budget-validator": entries[-1].update(static_shared_bytes=49153,shared_bytes_per_cta=49153)
    else: entries[-1] = deepcopy(entries[0])
    with pytest.raises(CacheMismatch): construct(spec,key,metadata)
