"""Controls/source contracts and host launch order, without GPU computation."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace

import pytest

from ragged import Artifact, CacheKey, CompiledProgram, Optimizations, ProgramSpec, ValidationError
from ragged.cli import main
from ragged.errors import CacheMismatch
from ragged.layout import I64_MAX, resolve_layout
from ragged.optimizations import BOOL_OPTIONS
from test_runtime_checked import host_cuda
from test_runtime_mask import install, with_mask


def controlled_program(options=None, *, base=None, scale=0.5):
    base = base or with_mask()
    options = options or Optimizations(pointwise_fusion=False, feature_tile=2)
    spec = replace(base.spec, optimizations=options, scale=scale)
    key = CacheKey.from_spec(spec,llvm_commit=base.artifact.key.llvm_commit,
                             ptxas_version=base.artifact.key.ptxas_version,compiler_sha256=base.artifact.key.compiler_sha256)
    metadata = deepcopy(base.artifact.metadata)
    materialized = not options.pointwise_fusion and (spec.mask or scale != 1)
    tile = options.resolved_tile(spec.feature_dim)
    metadata.update(d_tile=tile,pointwise_materialized=materialized)
    metadata["optimizations"] = {"requested":options.to_dict(),"effective":{
        "classification":metadata.get("dispatch_abi") == "classified-v3",
        "online_fusion":metadata.get("fusion") == "online", "warp_mapping":"warp" in metadata["classes"],
        "split_mapping":"split" in metadata["classes"], "shuffle":"warp" in metadata["classes"] and options.shuffle,
        "pointwise_fusion":options.pointwise_fusion and (spec.mask or scale != 1),
        "row_pipelining":options.row_pipelining}}
    for index,kernel in enumerate(metadata["kernels"]):
        if metadata.get("dispatch_abi") != "classified-v3" or index >= 2:
            kernel["grid"][1] = (spec.feature_dim + tile - 1)//tile
    if materialized:
        metadata["workspace"].append({"name":"logits","dtype":"f32","shape":["N",spec.feature_dim]})
        for kernel in metadata["kernels"]:
            kernel["args"] = kernel["args"] + ["logits"]
        metadata["kernels"].insert(0,{"name":"pointwise_materialize","role":"materialize","grid":[256,1,1],
            "block":[128,1,1],"shared_bytes":0,"static_shared_bytes":0,"args":metadata["kernels"][0]["args"]})
    return CompiledProgram(spec,Artifact(base.artifact.path,key,base.artifact.cubin,metadata))


def test_checked_precedes_logits_allocation_materialization_and_main(host_cuda,monkeypatch):
    env = host_cuda
    program = install(env,controlled_program(),monkeypatch)
    report = program.run(env.Tensor((10,4),"f32",0x1000),env.Tensor((6,),"i64",0x2000),env.Tensor((10,4),"f32",0x3000),
                         mask=env.Tensor((10,4),"f32",0x4000),stream=env.consumer,mode="checked")
    assert report.kernel_names == ("offsets_init","offsets_check","pointwise_materialize","main_max","main_sum","main_normalize")
    assert report.grids[2] == (256,1,1) and all(grid == (5,2,1) for grid in report.grids[3:])
    assert all(len(call[5]) == 17 for call in env.calls[:2])
    assert all(len(call[5]) == 47 for call in env.calls[2:])
    # Statistics at offsets 0 and 256, logits last at 512, in one scratch allocation.
    scratch = env.allocations[-1]
    assert scratch.shape == (512+10*4*4,) and scratch.dtype == "u8"
    assert all(call[5][-7:] == [scratch.pointer+512]*2+[0,10,4,4,1] for call in env.calls[2:])
    assert env.log.index(("synchronize",202)) < env.log.index(("allocate",env.device,(512+10*4*4,),"u8"))
    assert report.workspace_bytes == 56+2*5*4*4+10*4*4
    assert report.optimizations["requested"] == program.spec.optimizations.to_dict()
    assert report.optimizations["effective"]["pointwise_fusion"] is False
    assert report.pointwise_materialized is True and report.d_tile == 2


def test_invalid_checked_offsets_do_not_materialize_logits(host_cuda,monkeypatch):
    env = host_cuda
    program = install(env,controlled_program(),monkeypatch)
    env.state["status"][1] = 5
    with pytest.raises(ValidationError,match="final_mismatch"):
        program.run(env.Tensor((10,4),"f32",0x1000),env.Tensor((6,),"i64",0x2000),env.Tensor((10,4),"f32",0x3000),
                    mask=env.Tensor((10,4),"f32",0x4000),stream=env.consumer,mode="checked")
    assert [call[0] for call in env.calls] == ["offsets_init","offsets_check"]
    assert all(tensor.shape == (7,) for tensor in env.allocations)


def test_classified_materialize_prefix_keeps_fixed_queue_schedule_and_logits_last(host_cuda,monkeypatch):
    from test_runtime_classified import host_program
    env = host_cuda
    program = install(env,controlled_program(base=with_mask(host_program()),options=Optimizations(pointwise_fusion=False,feature_tile=8)),monkeypatch)
    debug = {}
    report = program.run(env.Tensor((10,33),"f32",0x1000),env.Tensor((6,),"i64",0x2000),env.Tensor((10,33),"f32",0x3000),
                         mask=env.Tensor((10,33),"f32",0x4000),stream=env.consumer,debug_buffers=debug)
    assert len(env.calls) == 14 and all(len(call[5]) == 71 for call in env.calls)
    assert report.kernel_names[:3] == ("pointwise_materialize","init","classify")
    assert all(grid[1] == 5 for grid in report.grids[3:])
    assert list(debug)[-1] == "logits" and debug["logits"].shape == (10,33)
    assert not any(event[0] in ("copy","synchronize") for event in env.log)


def test_controls_are_immutable_roundtrip_and_emit_only_nondefaults():
    options = Optimizations(classification=False,online_fusion=False,feature_tile=4)
    spec = ProgramSpec("softmax",optimizations=options)
    assert ProgramSpec.from_dict(spec.to_dict()) == spec
    assert "classification = false" in spec.to_mlir() and "online_fusion = false" in spec.to_mlir()
    assert "feature_tile = 4 : i64" in spec.to_mlir()
    assert "pointwise_fusion" not in spec.to_mlir()
    assert ProgramSpec("softmax").to_mlir() == ProgramSpec("softmax",optimizations=Optimizations()).to_mlir()
    with pytest.raises(FrozenInstanceError): options.shuffle = False


@pytest.mark.parametrize("name",BOOL_OPTIONS)
def test_each_control_is_in_source_hash(name):
    plain = ProgramSpec("softmax")
    changed = ProgramSpec("softmax",optimizations=Optimizations(**{name:False}))
    assert plain.ir_hash != changed.ir_hash and plain.signature == changed.signature
    with pytest.raises(ValidationError): Optimizations(**{name:1})


@pytest.mark.parametrize("tile",[0,3,64,True,2.0,"4"])
def test_invalid_feature_tiles_fail(tile):
    with pytest.raises(ValidationError): Optimizations(feature_tile=tile)


@pytest.mark.parametrize("kwargs",[
    {"debug_visits":True,"optimizations":Optimizations(classification=False)},
    {"force_fusion":True,"optimizations":Optimizations(online_fusion=False)},
    {"optimizations":{"classification":False}},
])
def test_conflicting_control_requests_fail_before_compilation(kwargs):
    with pytest.raises(ValidationError): ProgramSpec("softmax",**kwargs)


def test_cli_emits_controls(capsys):
    assert main(["softmax","--disable-online-fusion","--disable-pointwise-fusion","--feature-tile","8"]) == 0
    source = capsys.readouterr().out
    assert "online_fusion = false" in source and "pointwise_fusion = false" in source and "feature_tile = 8 : i64" in source


@pytest.mark.parametrize("mutation",["request","effective","tile","logits_width","logits_order","prefix","duplicate_materialize","grid"])
def test_controls_manifest_mismatch_cannot_construct_program(mutation):
    program = controlled_program()
    metadata = deepcopy(program.artifact.metadata)
    if mutation == "request": metadata["optimizations"]["requested"]["shuffle"] = False
    elif mutation == "effective": metadata["optimizations"]["effective"]["warp_mapping"] = True
    elif mutation == "tile": metadata["d_tile"] = 4
    elif mutation == "logits_width": metadata["workspace"][-1]["shape"][1] = 3
    elif mutation == "logits_order": metadata["workspace"][-1],metadata["workspace"][-2] = metadata["workspace"][-2],metadata["workspace"][-1]
    elif mutation == "prefix": metadata["kernels"][0],metadata["kernels"][1] = metadata["kernels"][1],metadata["kernels"][0]
    elif mutation == "duplicate_materialize": metadata["kernels"][1]["role"] = "materialize"
    else: metadata["kernels"][1]["grid"][1] = 1
    with pytest.raises(CacheMismatch):
        CompiledProgram(program.spec,Artifact(program.artifact.path,program.artifact.key,program.artifact.cubin,metadata))


def test_materialized_n_times_d_bytes_reject_overflow_before_allocation():
    program = controlled_program()
    with pytest.raises(ValidationError,match="workspace"):
        resolve_layout(program.artifact.metadata,n=I64_MAX//8,b=1,d=4)
