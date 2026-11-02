"""Source/manifest budget contracts; synthetic evidence never executes on a GPU."""
from copy import deepcopy
from dataclasses import replace
from math import prod

import pytest

from ragged import Artifact, ArtifactCache, CacheKey, CompiledProgram, CompilationError, ProgramSpec, ValidationError
from ragged.cli import main
from ragged.errors import CacheMismatch
from ragged.resources import RESOURCE_FIELDS
from ragged.validation import validate_manifest
from test_frontend import compiler_transport
from test_runtime_checked import checked_program, host_cuda
from test_runtime_mask import install
from test_runtime_optimizations import controlled_program


def measured_resources(metadata, *, smem_per_cta=49152, regs_budget=65536):
    return {"status": "measured", "source": "ptxas",
            "requested": {"smem_per_cta": smem_per_cta, "regs_budget": regs_budget},
            "kernels": [{"name": kernel["name"], "registers_per_thread": 24,
                         "threads_per_cta": prod(kernel["block"]), "registers_per_cta": 24 * prod(kernel["block"]),
                         "static_shared_bytes": 512, "dynamic_shared_bytes": kernel.get("shared_bytes", 0),
                         "shared_bytes_per_cta": 512 + kernel.get("shared_bytes", 0),
                         "stack_frame_bytes": 0, "spill_store_bytes": 0, "spill_load_bytes": 0}
                        for kernel in metadata["kernels"] + metadata.get("validation", {}).get("kernels", [])]}


def with_resources(base=None, *, smem_per_cta=49152, regs_budget=65536):
    base = base or checked_program()
    spec = replace(base.spec, smem_per_cta=smem_per_cta, regs_budget=regs_budget)
    key = CacheKey.from_spec(spec, llvm_commit=base.artifact.key.llvm_commit,
                            ptxas_version=base.artifact.key.ptxas_version,
                            compiler_sha256=base.artifact.key.compiler_sha256)
    metadata = deepcopy(base.artifact.metadata)
    metadata["resources"] = measured_resources(metadata, smem_per_cta=smem_per_cta, regs_budget=regs_budget)
    return CompiledProgram(spec, Artifact(base.artifact.path, key, base.artifact.cubin, metadata))


@pytest.mark.parametrize("name", ["smem_per_cta", "regs_budget"])
@pytest.mark.parametrize("value", [0, -1, True, False, 1.0, "4096", None, 2**63])
def test_budgets_require_positive_signed_i64(name, value):
    with pytest.raises(ValidationError, match=name):
        ProgramSpec("sum", **{name: value})


@pytest.mark.parametrize("name", ["smem_per_cta", "regs_budget"])
@pytest.mark.parametrize("value", [1, 1000, 2**63 - 1])
def test_budget_is_source_and_cache_specialization(name, value):
    base = ProgramSpec("sum")
    spec = replace(base, **{name: value})
    assert spec == ProgramSpec.from_dict(spec.to_dict())
    assert f"{name} = {value} : i64" in spec.to_mlir()
    assert spec.signature == base.signature and spec.ir_hash != base.ir_hash
    assert CacheKey.from_spec(spec, llvm_commit="host", ptxas_version="host").digest != CacheKey.from_spec(
        base, llvm_commit="host", ptxas_version="host").digest


def test_cli_emits_budgets(capsys):
    assert main(["sum", "--smem-per-cta", "1", "--regs-budget", "2048"]) == 0
    source = capsys.readouterr().out
    assert "smem_per_cta = 1 : i64" in source and "regs_budget = 2048 : i64" in source
    assert main(["sum", "--regs-budget", "0"]) == 1
    assert "positive signed i64" in capsys.readouterr().err


def test_complete_resource_evidence_covers_materialize_and_validation():
    program = with_resources(controlled_program())
    metadata = program.artifact.metadata
    assert [entry["name"] for entry in metadata["resources"]["kernels"]] == [
        kernel["name"] for kernel in metadata["kernels"] + metadata["validation"]["kernels"]]
    # IR estimates and actual ptxas resource measurements may differ after DCE.
    assert metadata["kernels"][0]["static_shared_bytes"] == 0
    assert metadata["resources"]["kernels"][0]["static_shared_bytes"] == 512


@pytest.mark.parametrize("field", sorted(RESOURCE_FIELDS))
@pytest.mark.parametrize("value", [True, 1.0, -1, 2**63])
def test_measurements_are_strict_nonnegative_i64(field, value):
    metadata = deepcopy(with_resources().artifact.metadata)
    metadata["resources"]["kernels"][0][field] = value
    with pytest.raises(CacheMismatch):
        validate_manifest(metadata)


@pytest.mark.parametrize("mutation", ["absent_entry", "extra_entry", "duplicate_entry", "reorder", "missing_validation",
                                      "entry_field", "extra_field", "status", "source", "unmeasured", "none",
                                      "threads", "regs_product", "dynamic", "shared_sum", "regs_overflow", "shared_overflow",
                                      "register_budget", "shared_budget", "budget_bool", "budget_zero", "budget_field"])
def test_incomplete_inconsistent_or_excess_resource_evidence_is_rejected(mutation):
    metadata = deepcopy(with_resources().artifact.metadata)
    resources = metadata["resources"]
    entries = resources["kernels"]
    first = entries[0]
    if mutation == "absent_entry": entries.pop(0)
    elif mutation == "extra_entry": entries.append(deepcopy(first))
    elif mutation == "duplicate_entry": entries[1] = deepcopy(first)
    elif mutation == "reorder": entries[0], entries[1] = entries[1], entries[0]
    elif mutation == "missing_validation": del entries[-2:]
    elif mutation == "entry_field": del first["spill_load_bytes"]
    elif mutation == "extra_field": first["estimated"] = True
    elif mutation == "status": resources["status"] = "estimated"
    elif mutation == "source": resources["source"] = "ir"
    elif mutation == "unmeasured": resources.update(status="unmeasured", source=None, kernels=[])
    elif mutation == "none": metadata["resources"] = None
    elif mutation == "threads": first["threads_per_cta"] = 32
    elif mutation == "regs_product": first["registers_per_cta"] -= 1
    elif mutation == "dynamic": first["dynamic_shared_bytes"] = 1
    elif mutation == "shared_sum": first["shared_bytes_per_cta"] += 1
    elif mutation == "regs_overflow": first["registers_per_thread"] = 2**63 - 1
    elif mutation == "shared_overflow":
        first.update(static_shared_bytes=2**63-1, dynamic_shared_bytes=1, shared_bytes_per_cta=2**63-1)
        metadata["kernels"][0]["shared_bytes"] = 1
    elif mutation == "register_budget": resources["requested"]["regs_budget"] = first["registers_per_cta"] - 1
    elif mutation == "shared_budget": resources["requested"]["smem_per_cta"] = first["shared_bytes_per_cta"] - 1
    elif mutation == "budget_bool": resources["requested"]["regs_budget"] = True
    elif mutation == "budget_zero": resources["requested"]["regs_budget"] = 0
    else: resources["requested"]["registers"] = 65536
    with pytest.raises(CacheMismatch):
        validate_manifest(metadata)


def test_exact_budget_boundary_accepts_but_source_mismatch_rejects():
    program = with_resources(smem_per_cta=512, regs_budget=24*128)
    metadata = deepcopy(program.artifact.metadata)
    metadata["resources"]["requested"]["smem_per_cta"] += 1
    with pytest.raises(CacheMismatch, match="source specialization"):
        CompiledProgram(program.spec, replace(program.artifact, metadata=metadata))


def test_legacy_is_unmeasured_and_rejects_nondefault_budget():
    program = checked_program()
    assert program._optimization_report()["resources"] is None
    spec = replace(program.spec, smem_per_cta=512)
    key = replace(program.artifact.key, ir_hash=spec.ir_hash)
    with pytest.raises(CacheMismatch, match="nondefault resource budgets"):
        CompiledProgram(spec, replace(program.artifact, key=key))


def test_report_retains_detached_compiled_resources_without_new_sync(host_cuda, monkeypatch):
    env = host_cuda
    program = install(env, with_resources(), monkeypatch)
    report = env.run(program=program, mode="validated")
    assert report.resources == program.artifact.metadata["resources"]
    assert not any(event[0] in ("copy", "synchronize") for event in env.log)
    report.resources["kernels"][0]["registers_per_thread"] = 999
    assert program._optimization_report()["resources"]["kernels"][0]["registers_per_thread"] == 24


def test_new_compiler_artifact_without_measurements_cannot_publish(compiler_transport, tmp_path):
    compiler, state = compiler_transport
    state["omit_resources"] = True
    with pytest.raises(CompilationError, match="no measured ptxas resources"):
        compiler.compile(ProgramSpec("sum"), ArtifactCache(tmp_path))
    assert not list(tmp_path.rglob("manifest.json"))


def test_current_compiler_cache_cannot_bypass_missing_measurements(compiler_transport, tmp_path):
    compiler, _ = compiler_transport
    spec = ProgramSpec("sum")
    first = compiler.compile(spec, ArtifactCache(tmp_path / "original"))
    metadata = deepcopy(first.artifact.metadata)
    del metadata["resources"]
    cache = ArtifactCache(tmp_path / "invalid")
    cache.publish(first.artifact.key, first.artifact.cubin, metadata)
    with pytest.raises(CacheMismatch, match="lacks measured"):
        compiler.compile(spec, cache)


def test_over_budget_metadata_cannot_publish(compiler_transport, tmp_path):
    compiler, state = compiler_transport
    resources = measured_resources({"kernels": [{"name": "sum_kernel", "block": [128, 1, 1]}]}, regs_budget=1)
    state["metadata_override"] = {"resources": resources}
    with pytest.raises(CacheMismatch, match="measured registers exceed"):
        compiler.compile(ProgramSpec("sum", regs_budget=1), ArtifactCache(tmp_path))
    assert not list(tmp_path.rglob("manifest.json"))


def test_duplicate_resource_json_field_cannot_publish(compiler_transport, tmp_path, monkeypatch):
    compiler, _ = compiler_transport
    original = compiler._invoke
    def duplicate(source, output, *args):
        result = original(source, output, *args)
        sidecar = output.with_suffix(".cubin.json")
        sidecar.write_text(sidecar.read_text().replace('"registers_per_thread": 16',
                          '"registers_per_thread": 1, "registers_per_thread": 16', 1))
        return result
    monkeypatch.setattr(compiler, "_invoke", duplicate)
    with pytest.raises(CompilationError, match="duplicate manifest field"):
        compiler.compile(ProgramSpec("sum"), ArtifactCache(tmp_path))
    assert not list(tmp_path.rglob("manifest.json"))
