"""Real Python -> C++ -> NVVM -> ptxas cubin/cache checks; no GPU execution."""
import json
import os
from pathlib import Path
import shutil
import sys
from dataclasses import replace
from math import prod

import pytest

from ragged import ArtifactCache, CompilationError, Compiler, Hints, Optimizations, ProgramSpec


@pytest.fixture(scope="module")
def compiler():
    executable = os.environ.get("RAGGED_COMPILE", str(Path(__file__).resolve().parents[2] / "build/bin/ragged-compile"))
    if not shutil.which(executable):
        pytest.skip("real ragged-compile is not built")
    return Compiler(executable)


@pytest.mark.parametrize("op", ["sum", "max", "softmax"])
@pytest.mark.parametrize("dtype", ["f32", "bf16"])
@pytest.mark.parametrize("offset_type", ["i32", "i64"])
@pytest.mark.parametrize("strategy", ["auto", "naive"])
def test_real_cubin_manifest_and_cache(compiler, tmp_path, op, dtype, offset_type, strategy):
    spec = ProgramSpec(op, dtype, 17, offset_type, hints=Hints(max_len=32, offsets_checked=True), strategy=strategy)
    cache = ArtifactCache(tmp_path)
    program = compiler.compile(spec, cache)
    assert program.artifact.cubin.startswith(b"\x7fELF")
    assert not program.cache_hit
    metadata = program.artifact.metadata
    validation = metadata["validation"]
    assert validation["abi"] == "offsets-check-v1"
    assert validation["max_len"] == 32 and validation["min_len"] is None
    assert [kernel["role"] for kernel in validation["kernels"]] == ["init", "check"]
    assert all(kernel["args"] == ["values", "offsets", "validation_status"] for kernel in validation["kernels"])
    assert metadata["toolchain"] == compiler.toolchain()
    assert metadata["toolchain"]["features"] == "+ptx80"
    assert program.artifact.key.chip == metadata["chip"] == "sm_89"
    cached = compiler.compile(spec, cache)
    assert cached.cache_hit
    assert cached.artifact.cubin == program.artifact.cubin
    assert cached.artifact.key == program.artifact.key


def test_real_three_pass_inplace_workspace_abi(compiler, tmp_path):
    spec = ProgramSpec("softmax", "bf16", 128, "i64", hints=Hints(offsets_checked=True),
                       strategy="naive", alias="inplace")
    program = compiler.compile(spec, ArtifactCache(tmp_path))
    metadata = program.artifact.metadata
    assert metadata["alias_safe"] is True
    assert len(metadata["kernels"]) == 3
    assert metadata["workspace"] == [{"name": name, "shape": ["B", 128], "dtype": "f32"}
                                     for name in ("max_stats", "sum_stats")]
    assert all(kernel["args"] == ["values", "offsets", "out", "max_stats", "sum_stats"]
               for kernel in metadata["kernels"])


def test_real_features_change_binary_key_and_provenance(compiler, tmp_path):
    spec = ProgramSpec("sum", feature_dim=17, hints=Hints(max_len=32, offsets_checked=True))
    cache = ArtifactCache(tmp_path)
    first = compiler.compile(spec, cache, features="+ptx80")
    second = compiler.compile(spec, cache, features="+ptx81")
    assert first.artifact.key.digest != second.artifact.key.digest
    assert second.artifact.metadata["toolchain"]["features"] == "+ptx81"


def test_real_toolchain_rejects_false_caller_version(compiler, tmp_path):
    with pytest.raises(CompilationError, match="ptxas_version"):
        compiler.compile(ProgramSpec("sum"), ArtifactCache(tmp_path), ptxas_version="invented")
    assert not list(tmp_path.rglob("manifest.json"))


def test_ptxas_process_failure_cannot_publish_a_binary(compiler, tmp_path, monkeypatch):
    fake_bin = tmp_path / "fault-bin"
    fake_bin.mkdir()
    executable = fake_bin / "ptxas"
    executable.write_text('#!/bin/sh\nif [ "$1" = "--version" ]; then\n echo "fault-injection ptxas"\n exit 0\nfi\necho "intentional PTX assembler failure" >&2\nexit 42\n')
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(fake_bin) + os.pathsep + os.environ["PATH"])
    spec = ProgramSpec("sum", feature_dim=17, hints=Hints(max_len=32, offsets_checked=True))
    cache = ArtifactCache(tmp_path / "cache")
    with pytest.raises(CompilationError, match="intentional PTX assembler failure"):
        compiler.compile(spec, cache)
    assert not list((tmp_path / "cache").rglob("manifest.json"))


def test_bound_real_program_captures_as_opaque_custom_op(compiler, tmp_path):
    torch = pytest.importorskip("torch")
    from ragged.torch_adapter import bind
    spec = ProgramSpec("softmax", feature_dim=17, hints=Hints(max_len=32, offsets_checked=True))
    program = compiler.compile(spec, ArtifactCache(tmp_path))
    operation = bind(program)
    graphs = []
    def backend(graph, example_inputs):
        graphs.append(graph)
        return graph.forward
    compiled = torch.compile(operation, backend=backend, fullgraph=True)
    values = torch.empty((13, 17), device="meta")
    offsets = torch.empty((5,), dtype=torch.int64, device="meta")
    assert compiled(values, offsets).shape == values.shape
    assert len(graphs) == 1
    assert any("ragged_mlir.segment_softmax" in str(node.target) for node in graphs[0].graph.nodes)


@pytest.mark.parametrize("op", ["sum", "max", "softmax"])
@pytest.mark.parametrize("hints,classes,debug", [
    (Hints(offsets_checked=True), ["warp", "cta", "split"], False),
    (Hints(min_len=4097, offsets_checked=True), ["split"], False),
    (Hints(max_len=4096, offsets_checked=True), ["warp", "cta"], False),
    (Hints(min_len=33, offsets_checked=True), ["cta", "split"], False),
    (Hints(max_len=32, offsets_checked=True), ["warp"], True),
], ids=["unknown-length", "static-split", "warp-cta", "cta-split", "debug-warp"])
def test_real_classified_cubin_typed_workspace_and_cache(compiler, tmp_path, op, hints, classes, debug):
    from ragged.layout import resolve_layout
    spec = ProgramSpec(op, "bf16", 33, "i32", hints=hints, debug_visits=debug)
    cache = ArtifactCache(tmp_path)
    program = compiler.compile(spec, cache)
    metadata = program.artifact.metadata
    assert metadata["dispatch_abi"] == "classified-v3"
    assert metadata["classes"] == classes
    assert metadata["debug_visits"] is debug
    assert metadata["split_parts"] == 8
    assert metadata["compiler_sha256"] == program.artifact.key.compiler_sha256
    assert program.artifact.key.compiler_sha256 != "unbound"
    assert metadata["toolchain"] == compiler.toolchain()
    assert [kernel["role"] for kernel in metadata["kernels"][:2]] == ["init", "classify"]
    workspace = metadata["workspace"]
    assert workspace[:2] == [{"name": "class_counts", "shape": [3], "dtype": "i64"},
                             {"name": "segment_ids", "shape": [3, "B"], "dtype": "i64"}]
    assert ("partials" in [entry["name"] for entry in workspace]) == ("split" in classes)
    assert ("visits" in [entry["name"] for entry in workspace]) is debug
    if debug:
        assert workspace[-1] == {"name": "visits", "shape": ["B"], "dtype": "i32"}
    if op == "softmax":
        # Every class uses online statistics; split merges its parts in a combine kernel.
        assert metadata["class_fusion"] == ["online"] * len(classes) and metadata["fusion"] == "online"
        assert metadata["alias_safe"] is False and metadata["optimizations"]["effective"]["online_fusion"] is True
        names = [entry["name"] for entry in workspace]
        assert ("max_stats" in names) == ("partial_sums" in names) == ("split" in classes)
        if "split" in classes:
            assert [kernel["role"] for kernel in metadata["kernels"] if kernel.get("segment_class") == "split"] == [
                "split_partial_online", "split_combine_online", "split_normalize"]
    else:
        assert metadata["fusion"] == "not_applicable" and "class_fusion" not in metadata
    expected_args = ["values", "offsets", "out", *[entry["name"] for entry in workspace]]
    assert all(kernel["args"] == expected_args for kernel in metadata["kernels"])
    workers = {kernel["name"]: 100 + index for index, kernel in enumerate(metadata["kernels"]) if "segment_class" in kernel}
    layouts, grids, total = resolve_layout(metadata, n=7000, b=257, d=33, workers=workers)
    assert grids[:2] == ((1, 1, 1), (3, 1, 1))
    assert [grid[:2] for grid in grids[2:]] == [(workers[kernel["name"]], 2) for kernel in metadata["kernels"][2:]]
    assert total == sum(layout.nbytes for layout in layouts)
    cached = compiler.compile(spec, cache)
    assert cached.cache_hit and cached.artifact.cubin == program.artifact.cubin


def test_real_classified_debug_full_schedule_has_visits(compiler, tmp_path):
    spec = ProgramSpec("softmax", "f32", 128, "i64", hints=Hints(offsets_checked=True), debug_visits=True)
    program = compiler.compile(spec, ArtifactCache(tmp_path))
    metadata = program.artifact.metadata
    assert metadata["workspace"][-1] == {"name": "visits", "shape": ["B"], "dtype": "i32"}
    assert metadata["fusion"] == "online" and metadata["class_fusion"] == ["online", "online", "online"]
    assert [kernel["role"] for kernel in metadata["kernels"]] == [
        "init", "classify", "warp_online", "cta_online", "split_partial_online", "split_combine_online", "split_normalize"]


@pytest.mark.parametrize("offset_dtype", ["i32", "i64"])
@pytest.mark.parametrize("hints", [Hints(), Hints(0, 0, True), Hints(4096, 1, True)],
                         ids=["unknown-bounds", "empty-bound", "trusted-static-bounds"])
def test_real_validation_artifact_retains_all_source_bounds(compiler, tmp_path, offset_dtype, hints):
    from ragged.layout import resolve_layout
    spec = ProgramSpec("softmax", "bf16", 17, offset_dtype, hints=hints)
    cache = ArtifactCache(tmp_path)
    program = compiler.compile(spec, cache)
    validation = program.artifact.metadata["validation"]
    assert validation["min_len"] == hints.min_len and validation["max_len"] == hints.max_len
    assert validation["sentinel"] == 2**63-1
    assert validation["conditions"] == ["first_nonzero", "final_mismatch", "negative", "above_n",
                                         "nonmonotone", "below_min_len", "above_max_len"]
    assert validation["workspace"] == [{"name":"validation_status","shape":[7],"dtype":"i64"}]
    assert len(set(k["name"] for k in validation["kernels"]).intersection(k["name"] for k in program.artifact.metadata["kernels"])) == 0
    for b,expected in ((0,1),(127,1),(128,2)):
        _,grids,nbytes = resolve_layout(validation,n=0,b=b,d=17)
        assert grids == ((1,1,1),(expected,1,1)) and nbytes == 56
    hit = compiler.compile(spec,cache)
    assert hit.cache_hit and hit.artifact.metadata["validation"] == validation


@pytest.mark.parametrize("dtype", ["f32","bf16"])
@pytest.mark.parametrize("offset_dtype", ["i32","i64"])
@pytest.mark.parametrize("strategy,hints", [
    ("naive",Hints(offsets_checked=True)), ("auto",Hints(max_len=32,offsets_checked=True)),
    ("auto",Hints(max_len=256,min_len=33,offsets_checked=True)), ("auto",Hints(offsets_checked=True)),
    ("auto",Hints(min_len=4097,offsets_checked=True)), ("auto",Hints(max_len=4096,offsets_checked=True)),
], ids=["naive","online-warp","bounded-cta","classified","split","two-class"])
def test_real_masked_cubin_abi_and_cache(compiler,tmp_path,dtype,offset_dtype,strategy,hints):
    spec = ProgramSpec("softmax",dtype,33,offset_dtype,hints=hints,strategy=strategy,scale=0.5,mask=True)
    cache = ArtifactCache(tmp_path)
    program = compiler.compile(spec,cache)
    metadata = program.artifact.metadata
    assert metadata["mask"] is True
    args = ["values","offsets","out","mask",*[entry["name"] for entry in metadata.get("workspace",[])]]
    assert all(kernel["args"] == args for kernel in metadata["kernels"])
    assert all(kernel["args"] == ["values","offsets","validation_status"] for kernel in metadata["validation"]["kernels"])
    assert metadata["toolchain"] == compiler.toolchain()
    hit = compiler.compile(spec,cache)
    assert hit.cache_hit and hit.artifact.cubin == program.artifact.cubin


@pytest.mark.parametrize("dtype", ["f32","bf16"])
def test_masked_inplace_debug_classified_manifest(compiler,tmp_path,dtype):
    spec = ProgramSpec("softmax",dtype,17,mask=True,alias="inplace",debug_visits=True)
    program = compiler.compile(spec,ArtifactCache(tmp_path))
    assert program.artifact.metadata["mask"] is True
    assert program.artifact.metadata["alias_safe"] is True
    assert len(program.artifact.metadata["kernels"]) == 13
    assert all(kernel["args"][:4] == ["values","offsets","out","mask"] for kernel in program.artifact.metadata["kernels"])


def test_mask_changes_real_cache_identity(compiler,tmp_path):
    cache = ArtifactCache(tmp_path)
    common = dict(op="softmax",feature_dim=17,hints=Hints(max_len=32,offsets_checked=True))
    plain = compiler.compile(ProgramSpec(**common),cache)
    masked = compiler.compile(ProgramSpec(**common,mask=True),cache)
    assert plain.artifact.key.digest != masked.artifact.key.digest
    assert plain.artifact.key.ir_hash != masked.artifact.key.ir_hash
    assert plain.artifact.metadata["mask"] is False and masked.artifact.metadata["mask"] is True
    assert not plain.cache_hit and not masked.cache_hit


@pytest.mark.parametrize("name,op,hints", [
    ("classification","sum",Hints()), ("classification","softmax",Hints()),
    ("online_fusion","softmax",Hints(max_len=32)), ("warp_mapping","sum",Hints(max_len=32)),
    ("split_mapping","max",Hints(min_len=4097)), ("shuffle","sum",Hints(max_len=32)),
    ("shuffle","softmax",Hints(max_len=32)),
])
def test_real_individual_optimization_controls(compiler,tmp_path,name,op,hints):
    options = Optimizations(**{name:False})
    spec = ProgramSpec(op,"bf16",17,"i32",hints=hints,optimizations=options)
    cache = ArtifactCache(tmp_path)
    program = compiler.compile(spec,cache)
    metadata = program.artifact.metadata
    assert metadata["optimizations"]["requested"] == options.to_dict()
    assert metadata["optimizations"]["effective"][name] is False
    assert metadata["d_tile"] == 32 and metadata["pointwise_materialized"] is False
    if name == "classification": assert metadata["classes"] == ["cta"] and "dispatch_abi" not in metadata
    if name == "online_fusion": assert metadata["fusion"] == "three_pass"
    if name == "warp_mapping": assert "warp" not in metadata["classes"]
    if name == "split_mapping": assert "split" not in metadata["classes"]
    assert compiler.compile(spec,cache).cache_hit


@pytest.mark.parametrize("tile",[1,2,4,8,16,32])
def test_real_feature_tile_is_bound_to_worker_grids(compiler,tmp_path,tile):
    spec = ProgramSpec("sum",feature_dim=33,optimizations=Optimizations(feature_tile=tile))
    program = compiler.compile(spec,ArtifactCache(tmp_path))
    metadata = program.artifact.metadata
    assert metadata["d_tile"] == tile
    assert all(kernel["grid"][1] == (33+tile-1)//tile for kernel in metadata["kernels"][2:])


@pytest.mark.parametrize("op,mask,hints,strategy,alias,debug",[
    ("sum",False,Hints(),"auto","none",False),
    ("max",False,Hints(max_len=32),"auto","none",False),
    ("softmax",True,Hints(max_len=32),"auto","none",False),
    ("softmax",True,Hints(),"naive","none",False),
    ("softmax",True,Hints(),"auto","none",True),
    ("softmax",True,Hints(min_len=4097),"auto","none",False),
    ("softmax",True,Hints(),"auto","inplace",False),
],ids=["scaled-sum","scaled-max","online-mask","naive-mask","classified-debug-mask","split-mask","inplace-mask"])
def test_real_pointwise_materialization_prefix_and_workspace(compiler,tmp_path,op,mask,hints,strategy,alias,debug):
    spec = ProgramSpec(op,"bf16",17,"i32",hints=hints,strategy=strategy,alias=alias,mask=mask,debug_visits=debug,
                       scale=0.5,optimizations=Optimizations(pointwise_fusion=False,feature_tile=4))
    cache = ArtifactCache(tmp_path)
    program = compiler.compile(spec,cache)
    metadata = program.artifact.metadata
    assert metadata["pointwise_materialized"] is True
    assert metadata["optimizations"]["effective"]["pointwise_fusion"] is False
    assert metadata["kernels"][0]["role"] == "materialize"
    assert metadata["kernels"][0]["grid"] == [256,1,1] and metadata["kernels"][0]["block"] == [128,1,1]
    assert metadata["workspace"][-1] == {"name":"logits","dtype":"f32","shape":["N",17]}
    assert all(kernel["args"][-1] == "logits" for kernel in metadata["kernels"])
    assert all(kernel["args"] == ["values","offsets","validation_status"] for kernel in metadata["validation"]["kernels"])
    assert compiler.compile(spec,cache).cache_hit


@pytest.mark.parametrize("scale",[1.0,1.0+1e-8])
def test_pointwise_disable_has_no_extra_work_for_f32_identity_scale(compiler,tmp_path,scale):
    spec = ProgramSpec("sum",feature_dim=7,hints=Hints(max_len=32),scale=scale,optimizations=Optimizations(pointwise_fusion=False))
    metadata = compiler.compile(spec,ArtifactCache(tmp_path)).artifact.metadata
    assert metadata["pointwise_materialized"] is False
    assert metadata["optimizations"]["effective"]["pointwise_fusion"] is False
    assert "logits" not in [entry["name"] for entry in metadata["workspace"]]


def test_naive_debug_preserves_instrumented_cta_dispatch(compiler,tmp_path):
    spec = ProgramSpec("softmax",feature_dim=7,strategy="naive",debug_visits=True)
    metadata = compiler.compile(spec,ArtifactCache(tmp_path)).artifact.metadata
    assert metadata["classes"] == ["cta"] and metadata["dispatch_abi"] == "classified-v3"
    assert metadata["optimizations"]["effective"]["classification"] is True


@pytest.mark.parametrize("spec", [
    ProgramSpec("sum", feature_dim=7, hints=Hints(max_len=32)),
    ProgramSpec("max", feature_dim=33, strategy="naive"),
    ProgramSpec("softmax", "bf16", 128, "i32", hints=Hints(min_len=33, max_len=4096), mask=True),
    ProgramSpec("softmax", "bf16", 17, "i32", mask=True, debug_visits=True,
                optimizations=Optimizations(pointwise_fusion=False, feature_tile=4)),
    ProgramSpec("softmax", feature_dim=7, mask=True, alias="inplace"),
], ids=["warp", "cta", "online-cta", "classified-materialized", "inplace"])
def test_real_measured_resources_cover_all_entries_and_bind_custom_budgets(compiler, tmp_path, spec):
    spec = replace(spec, smem_per_cta=4096, regs_budget=16384)
    cache = ArtifactCache(tmp_path)
    program = compiler.compile(spec, cache)
    metadata = program.artifact.metadata
    resources = metadata["resources"]
    assert resources["status"] == "measured" and resources["source"] == "ptxas"
    assert resources["requested"] == {"smem_per_cta": 4096, "regs_budget": 16384}
    launches = metadata["kernels"] + metadata["validation"]["kernels"]
    assert len(resources["kernels"]) == len(launches)
    for resource, launch in zip(resources["kernels"], launches):
        assert resource["name"] == launch["name"]
        assert resource["threads_per_cta"] == prod(launch["block"])
        assert resource["registers_per_cta"] == resource["registers_per_thread"] * prod(launch["block"])
        assert resource["dynamic_shared_bytes"] == launch["shared_bytes"]
        assert resource["shared_bytes_per_cta"] == resource["static_shared_bytes"] + launch["shared_bytes"]
        assert resource["registers_per_cta"] <= spec.regs_budget
        assert resource["shared_bytes_per_cta"] <= spec.smem_per_cta
        assert resource["registers_per_thread"] > 0
    assert not program.cache_hit
    assert compiler.compile(spec, cache).cache_hit


def test_real_resource_exact_boundary_and_one_register_under_cannot_publish(compiler, tmp_path):
    spec = ProgramSpec("sum", feature_dim=17, strategy="naive")
    baseline = compiler.compile(spec, ArtifactCache(tmp_path / "baseline"))
    resources = baseline.artifact.metadata["resources"]["kernels"]
    registers = max(entry["registers_per_cta"] for entry in resources)
    shared = max(1, max(entry["shared_bytes_per_cta"] for entry in resources))
    bounded = replace(spec, smem_per_cta=shared, regs_budget=registers)
    boundary = compiler.compile(bounded, ArtifactCache(tmp_path / "boundary"))
    assert boundary.artifact.metadata["resources"]["requested"] == {"smem_per_cta": shared, "regs_budget": registers}
    assert boundary.artifact.key.digest != baseline.artifact.key.digest
    cache = ArtifactCache(tmp_path / "under")
    with pytest.raises(CompilationError, match="(?i)(resource|budget|register)"):
        compiler.compile(replace(bounded, regs_budget=registers - 1), cache)
    assert not list(cache.root.rglob("manifest.json"))


def test_real_positive_budget_below_old_plan_floor_reaches_measured_gate(compiler, tmp_path):
    # A warp shuffle reduction has no static shared allocation; even one byte
    # is a meaningful accepted budget. A one-register budget cannot hold the
    # required validator and must reject after actual cubin resource analysis.
    spec = ProgramSpec("sum", feature_dim=7, hints=Hints(max_len=32), smem_per_cta=1)
    program = compiler.compile(spec, ArtifactCache(tmp_path / "small-shared"))
    assert all(entry["shared_bytes_per_cta"] == 0 for entry in program.artifact.metadata["resources"]["kernels"])
    cache = ArtifactCache(tmp_path / "small-register")
    with pytest.raises(CompilationError, match="(?i)(resource|budget|register)"):
        compiler.compile(replace(spec, regs_budget=1), cache)
    assert not list(cache.root.rglob("manifest.json"))


@pytest.mark.parametrize("fault", ["partial", "duplicate"])
def test_real_successful_ptxas_with_invalid_resource_log_cannot_publish(compiler, tmp_path, monkeypatch, fault):
    real_ptxas = shutil.which("ptxas")
    if real_ptxas is None:
        pytest.skip("ptxas must be on PATH for assembler-log fault injection")
    fake_bin = tmp_path / "fault-bin"
    fake_bin.mkdir()
    wrapper = fake_bin / "ptxas"
    wrapper.write_text(f'''#!{sys.executable}
import re
import subprocess
import sys
result = subprocess.run([{str(Path(real_ptxas).resolve())!r}, *sys.argv[1:]], capture_output=True)
log = result.stderr.decode()
if "--version" not in sys.argv:
    if {fault!r} == "partial":
        log, changed = re.subn(r"(?m)^.*Used [0-9]+ registers.*\\n?", "", log, count=1)
        assert changed == 1, "fault injection did not locate a ptxas resource record"
    else:
        log += log
sys.stdout.buffer.write(result.stdout)
sys.stderr.write(log)
sys.exit(result.returncode)
''')
    wrapper.chmod(0o755)
    monkeypatch.setenv("PATH", str(fake_bin) + os.pathsep + os.environ["PATH"])
    cache = ArtifactCache(tmp_path / "cache")
    with pytest.raises(CompilationError, match="(?i)(resource|duplicate|missing|incomplete|ptxas)"):
        compiler.compile(ProgramSpec("sum", feature_dim=7, hints=Hints(max_len=32)), cache)
    assert not list(cache.root.rglob("manifest.json"))
