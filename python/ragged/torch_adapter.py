"""Optional public PyTorch custom-op and contiguous NJT integration.

Call register_torch_ops() explicitly. Importing ragged never imports torch.
Registered custom ops are opaque forward-only calls with shape-only fake impls.
"""
from __future__ import annotations

import threading
from typing import Any

from .errors import ValidationError
from .runtime import CompiledProgram, _torch

_lock = threading.RLock()
_programs: dict[str, CompiledProgram] = {}
_operators: dict[str, Any] = {}


def register_program(program: CompiledProgram) -> str:
    if program.spec.alias != "none":
        raise ValidationError("functional custom ops require an out-of-place specialization")
    token = program.artifact.key.digest
    with _lock:
        _programs[token] = program
    return token


def _check_mode(mode: str) -> None:
    if mode not in ("validated", "checked"):
        raise ValidationError("mode must be validated or checked")


def _execute(values, offsets, token: str, op: str, mode: str, mask=None):
    _check_mode(mode)
    with _lock:
        program = _programs.get(token)
    if program is None:
        raise ValidationError("compiled program token is not registered in this process")
    if program.spec.op != op:
        raise ValidationError("custom op does not match compiled program semantics")
    _check_mask_presence(program.spec.mask, mask)
    rows = values.shape[0] if op == "softmax" else offsets.shape[0] - 1
    output = values.new_empty((rows, values.shape[1]))
    kwargs = {"mode": mode}
    if mask is not None:
        kwargs["mask"] = mask
    program.run(values, offsets, output, **kwargs)
    return output


def register_torch_ops() -> dict[str, Any]:
    with _lock:
        if _operators:
            return dict(_operators)
        torch = _torch()
        # Explicit schemas avoid inspecting stringified annotations across versions.
        @torch.library.custom_op("ragged_mlir::segment_reduce", mutates_args=(),
                                 schema='(Tensor values, Tensor offsets, str kind, str program, str mode="validated") -> Tensor')
        def reduce_op(values, offsets, kind, program, mode="validated"):
            if kind not in ("sum", "max"):
                raise ValidationError("segment_reduce kind must be sum or max")
            return _execute(values, offsets, program, kind, mode)

        @reduce_op.register_fake
        def reduce_fake(values, offsets, kind, program, mode="validated"):
            _check_mode(mode)
            _fake_validate(values, offsets, torch)
            if kind not in ("sum", "max"):
                raise ValidationError("segment_reduce kind must be sum or max")
            return values.new_empty((offsets.shape[0] - 1, values.shape[1]))

        @torch.library.custom_op("ragged_mlir::segment_softmax", mutates_args=(),
                                 schema='(Tensor values, Tensor offsets, str program, str mode="validated", Tensor? mask=None) -> Tensor')
        def softmax_op(values, offsets, program, mode="validated", mask=None):
            return _execute(values, offsets, program, "softmax", mode, mask)

        @softmax_op.register_fake
        def softmax_fake(values, offsets, program, mode="validated", mask=None):
            _check_mode(mode)
            _fake_validate(values, offsets, torch)
            if mask is not None:
                _fake_validate_mask(values, mask, torch)
            return values.new_empty(values.shape)

        _operators.update(segment_reduce=reduce_op, segment_softmax=softmax_op)
        return dict(_operators)


def _fake_validate(values, offsets, torch) -> None:
    if values.ndim != 2 or offsets.ndim != 1:
        raise ValidationError("expected values[N,D] and offsets[B+1]")
    torch._check(offsets.shape[0] >= 1)
    torch._check(values.shape[1] >= 1)
    torch._check(values.shape[1] <= 1024)
    if values.dtype not in (torch.float32, torch.bfloat16):
        raise ValidationError("values must be f32 or bf16")
    if offsets.dtype not in (torch.int32, torch.int64):
        raise ValidationError("offsets must be i32 or i64")
    if not values.is_contiguous() or not offsets.is_contiguous():
        raise ValidationError("values and offsets must be contiguous")
    if values.device != offsets.device:
        raise ValidationError("values and offsets must use the same device")


def _check_mask_presence(expected: bool, mask) -> None:
    if expected != (mask is not None):
        raise ValidationError("mask presence must match the compiled specialization")


def _fake_validate_mask(values, mask, torch) -> None:
    if mask.ndim != 2 or mask.dtype != torch.float32:
        raise ValidationError("mask must be a rank-two f32 tensor")
    torch._check(mask.shape[0] == values.shape[0])
    torch._check(mask.shape[1] == values.shape[1])
    if not mask.is_contiguous() or mask.device != values.device:
        raise ValidationError("mask must be contiguous and use the values device")
    if mask.requires_grad:
        raise ValidationError("autograd is outside the supported forward-only mask contract")


def segment_reduce(values, offsets, kind: str, *, program: CompiledProgram, mode: str = "validated"):
    operators = register_torch_ops()
    return operators["segment_reduce"](values, offsets, kind, register_program(program), mode)


def segment_softmax(values, offsets, *, program: CompiledProgram, mode: str = "validated", mask=None):
    _check_mask_presence(program.spec.mask, mask)
    operators = register_torch_ops()
    return operators["segment_softmax"](values, offsets, register_program(program), mode, mask)


def bind(program: CompiledProgram, *, mode: str = "validated"):
    """Register once, returning a callable that can enter torch.compile graphs.

    Binding belongs outside the captured function. The returned callable only
    invokes the opaque custom op; it performs no registry mutation while tracing.
    """
    _check_mode(mode)
    operators = register_torch_ops()
    token = register_program(program)
    if program.spec.op == "softmax":
        operation = operators["segment_softmax"]
        expects_mask = program.spec.mask
        def execute(values, offsets, mask=None):
            _check_mask_presence(expects_mask, mask)
            return operation(values, offsets, token, mode, mask)
    else:
        operation = operators["segment_reduce"]
        kind = program.spec.op
        def execute(values, offsets):
            return operation(values, offsets, kind, token, mode)
    return execute


def jagged_components(nt):
    torch = _torch()
    if not getattr(nt, "is_nested", False) or nt.layout != torch.jagged:
        raise ValidationError("expected a torch.jagged nested tensor")
    if nt.dim() != 3:
        raise ValidationError("expected NJT shape [B, jagged, D]")
    if nt.lengths() is not None:
        raise ValidationError("NJT with explicit lengths/holes is outside the contiguous offsets contract")
    values, offsets = nt.values(), nt.offsets()
    if values.ndim != 2 or not values.is_contiguous() or not offsets.is_contiguous():
        raise ValidationError("NJT values[N,D] and offsets must be contiguous")
    if nt.shape[-1] != values.shape[-1]:
        raise ValidationError("only the first non-batch NJT dimension may be ragged")
    return values, offsets


def njt_reduce(nt, kind: str, *, program: CompiledProgram, mode: str = "validated"):
    values, offsets = jagged_components(nt)
    return segment_reduce(values, offsets, kind, program=program, mode=mode)


def njt_softmax(nt, *, program: CompiledProgram, mode: str = "validated", mask=None):
    torch = _torch()
    values, offsets = jagged_components(nt)
    output = segment_softmax(values, offsets, program=program, mode=mode, mask=mask)
    return torch.nested.nested_tensor_from_jagged(output, offsets=offsets)
