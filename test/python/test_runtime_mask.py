"""Additive mask ABI orchestration; mock launches do not compute GPU results."""
from copy import deepcopy
from dataclasses import replace

import pytest

from ragged import Artifact, CacheKey, CompiledProgram, ProgramSpec
from ragged.errors import CacheMismatch, RuntimeUnavailable, ValidationError
from test_runtime_checked import checked_program, host_cuda
from test_runtime_classified import host_program


def with_mask(base=None, *, dtype="f32", alias="none"):
    base = base or checked_program()
    spec = replace(base.spec, mask=True, dtype=dtype, alias=alias)
    key = CacheKey.from_spec(spec, llvm_commit=base.artifact.key.llvm_commit,
                             ptxas_version=base.artifact.key.ptxas_version, compiler_sha256=base.artifact.key.compiler_sha256)
    metadata = deepcopy(base.artifact.metadata)
    metadata.update(mask=True, alias_safe=True, fusion="three_pass")
    for kernel in metadata["kernels"]:
        kernel["args"] = kernel["args"][:3] + ["mask"] + kernel["args"][3:]
    return CompiledProgram(spec, Artifact(base.artifact.path,key,base.artifact.cubin,metadata))


def install(env, program, monkeypatch):
    import ragged.runtime as runtime
    old = (0,env.program.artifact.key.digest)
    monkeypatch.setitem(runtime._resident_modules,(0,program.artifact.key.digest),runtime._resident_modules[old])
    return program


@pytest.mark.parametrize("mode", ["validated","checked"])
@pytest.mark.parametrize("dtype", ["f32","bf16"])
def test_mask_is_fourth_main_descriptor_and_absent_from_validation(host_cuda,monkeypatch,mode,dtype):
    env = host_cuda
    program = install(env,with_mask(dtype=dtype),monkeypatch)
    values,offsets,out = env.Tensor((10,4),dtype,0x1000),env.Tensor((6,),"i64",0x2000),env.Tensor((10,4),dtype,0x3000)
    mask = env.Tensor((10,4),"f32",0x4000)
    recorded = []
    mask.record_stream = lambda stream: recorded.append(stream)
    report = program.run(values,offsets,out,stream=env.consumer,mode=mode,mask=mask)
    main = env.calls[-3:]
    assert all(len(call[5]) == 40 for call in main)
    assert all(call[5][19:26] == [0x4000,0x4000,0,10,4,4,1] for call in main)
    assert recorded == [env.consumer]
    if mode == "checked":
        assert len(env.calls) == 5 and all(len(call[5]) == 17 for call in env.calls[:2])
        assert all(0x4000 not in call[5] for call in env.calls[:2])
    assert report.workspace_bytes == 2*5*4*4 + (56 if mode == "checked" else 0)


def test_classified_mask_keeps_all_workspace_arguments_after_four_inputs(host_cuda,monkeypatch):
    env = host_cuda
    program = install(env,with_mask(host_program()),monkeypatch)
    buffers = {}
    report = program.run(env.Tensor((10,33),"f32",0x1000),env.Tensor((6,),"i64",0x2000),
                         env.Tensor((10,33),"f32",0x3000),mask=env.Tensor((10,33),"f32",0x4000),
                         stream=env.consumer,debug_buffers=buffers)
    assert len(env.calls) == 13 and all(len(call[5]) == 64 for call in env.calls)
    assert all(call[5][26:31] == [buffers["class_counts"].pointer]*2+[0,3,1] for call in env.calls)
    # No segment of these 10 rows can be split, so the split partials are empty.
    assert report.workspace_bytes == 24 + 5*(24+2*33*4+4)


@pytest.mark.parametrize("problem", ["wrong_dtype","broadcast","rank","noncontiguous","cpu","device","grad","out","partial_out","offsets"])
def test_mask_tensor_contract_rejects_before_allocation_or_launch(host_cuda,monkeypatch,problem):
    env = host_cuda
    program = install(env,with_mask(),monkeypatch)
    values,offsets,out = env.Tensor((10,4),"f32",0x1000),env.Tensor((6,),"i64",0x2000),env.Tensor((10,4),"f32",0x3000)
    mask = env.Tensor((10,4),"f32",0x4000)
    if problem == "wrong_dtype": mask.dtype = "bf16"
    elif problem == "broadcast": mask.shape = (1,4)
    elif problem == "rank": mask.shape,mask.ndim = (40,),1
    elif problem == "noncontiguous": mask.is_contiguous = lambda: False
    elif problem == "cpu": mask.is_cuda = False
    elif problem == "device": mask.device = type(env.device)(1)
    elif problem == "grad": mask.requires_grad = True
    elif problem == "out": mask.pointer = out.pointer
    elif problem == "partial_out": mask.pointer = out.pointer + 4
    elif problem == "offsets": mask.pointer = offsets.pointer
    with pytest.raises((ValidationError,RuntimeUnavailable)):
        program.run(values,offsets,out,mask=mask,stream=env.consumer,mode="checked")
    assert env.allocations == env.calls == []


@pytest.mark.parametrize("offset", [0,4])
def test_readonly_values_mask_overlap_is_allowed(host_cuda,monkeypatch,offset):
    env = host_cuda
    program = install(env,with_mask(),monkeypatch)
    program.run(env.Tensor((10,4),"f32",0x1000),env.Tensor((6,),"i64",0x2000),env.Tensor((10,4),"f32",0x3000),
                mask=env.Tensor((10,4),"f32",0x1000+offset),stream=env.consumer)
    assert len(env.calls) == 3


def test_inplace_values_mask_alias_is_output_overlap(host_cuda,monkeypatch):
    env = host_cuda
    program = install(env,with_mask(alias="inplace"),monkeypatch)
    values = env.Tensor((10,4),"f32",0x1000)
    with pytest.raises(ValidationError,match="mask must not overlap"):
        program.run(values,env.Tensor((6,),"i64",0x2000),values,mask=values,stream=env.consumer)
    assert env.calls == env.allocations == []


def test_mask_presence_is_required_in_both_directions():
    with pytest.raises(ValidationError,match="mask presence"):
        with_mask().run(None,None,None)
    with pytest.raises(ValidationError,match="mask presence"):
        checked_program().run(None,None,None,mask=object())


@pytest.mark.parametrize("flag", [False,1,"true",None])
def test_mask_manifest_must_bind_source_and_have_boolean_type(flag):
    program = with_mask()
    metadata = deepcopy(program.artifact.metadata)
    metadata["mask"] = flag
    with pytest.raises(CacheMismatch):
        CompiledProgram(program.spec,Artifact(program.artifact.path,program.artifact.key,program.artifact.cubin,metadata))


def test_mask_flag_changes_only_masked_source_hash_and_is_softmax_only():
    plain,masked = ProgramSpec("softmax","bf16",17),ProgramSpec("softmax","bf16",17,mask=True)
    assert plain.ir_hash != masked.ir_hash
    assert "%mask: memref<?x17xf32>" in masked.to_mlir()
    assert '(memref<?x17xbf16>, memref<?xi64>, memref<?x17xbf16>, memref<?x17xf32>)' in masked.to_mlir()
    assert '"ragged.segment_softmax"(%values, %offsets, %out, %mask)' in masked.to_mlir()
    assert ProgramSpec.from_dict(masked.to_dict()) == masked
    with pytest.raises(ValidationError): ProgramSpec("sum",mask=True)
    with pytest.raises(ValidationError): ProgramSpec("softmax",mask=1)
