"""Handwritten comparison kernels, not the upper bound of Triton performance.

Fixed 32-row tiles and up to 32 features, one program per segment/feature tile.
Softmax uses three globally ordered kernels. There is no host length readback.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
import os
from pathlib import Path
import subprocess

# Triton 3.8 bundles an older assembler. The comparison uses the plan's toolkit.
os.environ.setdefault('TRITON_PTXAS_PATH', '/usr/local/cuda-13.2/bin/ptxas')
os.environ.setdefault('TRITON_LIBDEVICE_PATH', '/usr/local/cuda-13.2/nvvm/libdevice/libdevice.10.bc')
import triton
import triton.language as tl

if triton.__version__ != '3.8.0':
    raise RuntimeError('comparison baseline requires pinned Triton 3.8.0')


@lru_cache(maxsize=1)
def toolchain():
    tool = triton.knobs.nvidia.ptxas
    version = subprocess.run([tool.path, '--version'], capture_output=True, text=True, check=True).stdout
    if 'V13.2.86' not in version:
        raise RuntimeError('Triton baseline requires CUDA 13.2.2 ptxas V13.2.86; bundled fallback is unsupported')
    libdevice = Path(os.environ['TRITON_LIBDEVICE_PATH'])
    return {'triton': triton.__version__, 'ptxas_path': tool.path, 'ptxas_version': version,
            'libdevice_sha256': hashlib.sha256(libdevice.read_bytes()).hexdigest(),
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


@triton.jit
def _maximum_nan(left, right):
    return tl.maximum(left, right, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def _reduce(X, Offsets, Out, Maximum, D: tl.constexpr, OP: tl.constexpr,
            TILE: tl.constexpr, ROWS: tl.constexpr):
    segment = tl.program_id(0).to(tl.int64)
    feature = (tl.program_id(1) * TILE + tl.arange(0, TILE)).to(tl.int64)
    begin = tl.load(Offsets + segment).to(tl.int64)
    end = tl.load(Offsets + segment + 1).to(tl.int64)
    if OP == 1:
        accum = tl.full((ROWS, TILE), -float('inf'), tl.float32)
    else:
        accum = tl.full((ROWS, TILE), 0.0, tl.float32)
    carry = tl.full((ROWS, TILE), 0.0, tl.float32)
    if OP == 2:
        maximum = tl.load(Maximum + segment * D + feature, feature < D, other=0)
    for chunk in range(begin, end, ROWS):
        row = chunk + tl.arange(0, ROWS).to(tl.int64)
        valid = (row[:, None] < end) & (feature[None, :] < D)
        if OP == 1:
            x = tl.load(X + row[:, None] * D + feature[None, :], valid, other=-float('inf')).to(tl.float32)
            accum = tl.maximum(accum, x, propagate_nan=tl.PropagateNan.ALL)
        else:
            x = tl.load(X + row[:, None] * D + feature[None, :], valid, other=0).to(tl.float32)
            if OP == 2:
                x = tl.where(valid, tl.exp(x - maximum[None, :]), 0.0)
            y = x - carry
            new_sum = accum + y
            # Preserve ordinary inf/NaN behavior while compensating finite sums.
            finite = (new_sum == new_sum) & (tl.abs(new_sum) != float('inf'))
            carry = tl.where(finite, (new_sum - accum) - y, 0.0)
            accum = new_sum
    if OP == 1:
        result = tl.reduce(accum, axis=0, combine_fn=_maximum_nan)
    else:
        result = tl.sum(accum, axis=0)
    tl.store(Out + segment * D + feature, result, feature < D)


@triton.jit
def _normalize(X, Offsets, Maximum, Total, Out, D: tl.constexpr,
               TILE: tl.constexpr, ROWS: tl.constexpr):
    segment = tl.program_id(0).to(tl.int64)
    feature = (tl.program_id(1) * TILE + tl.arange(0, TILE)).to(tl.int64)
    begin = tl.load(Offsets + segment).to(tl.int64)
    end = tl.load(Offsets + segment + 1).to(tl.int64)
    maximum = tl.load(Maximum + segment * D + feature, feature < D, other=0)
    total = tl.load(Total + segment * D + feature, feature < D, other=1)
    for chunk in range(begin, end, ROWS):
        row = chunk + tl.arange(0, ROWS).to(tl.int64)
        valid = (row[:, None] < end) & (feature[None, :] < D)
        x = tl.load(X + row[:, None] * D + feature[None, :], valid, other=0).to(tl.float32)
        result = tl.exp(x - maximum[None, :]) / total[None, :]
        tl.store(Out + row[:, None] * D + feature[None, :], result, valid)


def call(values, offsets, operation):
    import torch
    toolchain()
    if operation not in ('sum', 'max', 'softmax'):
        raise ValueError('operation must be sum, max or softmax')
    if (values.ndim != 2 or offsets.ndim != 1 or offsets.numel() < 1
            or not values.is_contiguous() or not offsets.is_contiguous()):
        raise ValueError('baseline requires contiguous values[N,D] and offsets[B+1]')
    if not values.is_cuda or values.device != offsets.device:
        raise ValueError('baseline inputs must be on the same CUDA device')
    if values.dtype not in (torch.float32, torch.bfloat16) or offsets.dtype not in (torch.int32, torch.int64):
        raise ValueError('baseline requires f32/bf16 and i32/i64 offsets')
    if values.requires_grad:
        raise ValueError('baseline is forward-only')
    n, d = values.shape
    b = offsets.numel() - 1
    if not 1 <= d <= 1024 or (b == 0 and n != 0) or n * d > (2**63 - 1) // 4:
        raise ValueError('baseline shape violates contiguous forward contract')
    if offsets.dtype == torch.int32 and n > 2**31 - 1:
        raise ValueError('N exceeds i32 offsets range')
    if b > 2**31 - 1:
        raise ValueError('segment count exceeds CUDA grid x limit')
    with torch.cuda.device(values.device):
        if tuple(torch.cuda.get_device_capability(values.device)) != (8, 9):
            raise ValueError('benchmark target must be sm_89')
        stream = torch.cuda.current_stream(values.device)
        output = torch.empty((n if operation == 'softmax' else b, d), dtype=values.dtype, device=values.device)
        if b == 0:
            return output
        tile = min(32, triton.next_power_of_2(d))
        grid = (b, triton.cdiv(d, tile))
        options = dict(D=d, TILE=tile, ROWS=32, num_warps=4, enable_fp_fusion=False, enable_reflect_ftz=False)
        if operation == 'softmax':
            maximum = torch.empty((b, d), dtype=torch.float32, device=values.device)
            total = torch.empty_like(maximum)
            _reduce[grid](values, offsets, maximum, maximum, OP=1, **options)
            _reduce[grid](values, offsets, total, maximum, OP=2, **options)
            _normalize[grid](values, offsets, maximum, total, output, **options)
        else:
            _reduce[grid](values, offsets, output, output, OP=0 if operation == 'sum' else 1, **options)
        values.record_stream(stream)
        offsets.record_stream(stream)
        return output
