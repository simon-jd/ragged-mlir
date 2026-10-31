"""Public PyTorch baselines, kept outside the compiler execution path.

BF16 inputs are explicitly promoted before the operation and converted back
afterward, so the arithmetic contract is f32 accumulation and source-dtype output.
These conversions and NJT construction belong to the complete baseline call.
"""
from __future__ import annotations

import torch


def segment_reduce(values: torch.Tensor, offsets: torch.Tensor, kind: str) -> torch.Tensor:
    if kind not in ("sum", "max"):
        raise ValueError("kind must be sum or max")
    work = values.float() if values.dtype == torch.bfloat16 else values
    return torch.segment_reduce(work, kind, offsets=offsets, axis=0).to(values.dtype)


def njt_softmax(values: torch.Tensor, offsets: torch.Tensor, *, min_len=None, max_len=None) -> torch.Tensor:
    work = values.float() if values.dtype == torch.bfloat16 else values
    nt = torch.nested.nested_tensor_from_jagged(work, offsets=offsets, min_seqlen=min_len, max_seqlen=max_len)
    return torch.softmax(nt, dim=1).values().to(values.dtype)


def njt_sum(values: torch.Tensor, offsets: torch.Tensor, *, min_len=None, max_len=None) -> torch.Tensor:
    work = values.float() if values.dtype == torch.bfloat16 else values
    nt = torch.nested.nested_tensor_from_jagged(work, offsets=offsets, min_seqlen=min_len, max_seqlen=max_len)
    return nt.sum(dim=1).to(values.dtype)


def njt_max(values: torch.Tensor, offsets: torch.Tensor, *, min_len=None, max_len=None) -> torch.Tensor:
    """Public NJT-to-padded max with an explicit negative-infinity identity.

    The pinned NJT amax pads with the finite dtype minimum. That corrupts empty
    segments and shorter nonempty all-negative-infinity segments, so an empty
    segment mask alone cannot repair it. Explicit public to_padded_tensor(-inf)
    followed by amax preserves both cases. N=0 uses the identity output directly
    because a zero-width dense reduction is undefined (B=0 returns an empty
    result). Construction, dtype conversion, padding and output allocation all
    belong to the full baseline call. Compilation failures never select another
    backend or silently fall back to eager execution.
    """
    if values.shape[0] == 0:
        return values.new_full((offsets.shape[0] - 1, values.shape[1]), -float("inf"))
    work = values.float() if values.dtype == torch.bfloat16 else values
    nt = torch.nested.nested_tensor_from_jagged(work, offsets=offsets, min_seqlen=min_len, max_seqlen=max_len)
    padded = torch.nested.to_padded_tensor(nt, -float("inf"))
    return torch.amax(padded, dim=1).to(values.dtype)
