"""Autotuned Triton reference for segmented sum, max and softmax.

Two mappings, and the call times both on its first input of each shape and
keeps the fastest configuration (block rows, feature tile, warps, chunk):

- ``segment``: one program per (segment, feature tile) loops over the rows.
- ``chunked``: every segment is cut into chunks of at most CHUNK rows, one
  program each, so a long segment is spread over many SMs. A one-program map
  kernel builds the chunk-to-segment table on the device (no host readback);
  chunks of multi-chunk segments write partial results that a combine kernel
  merges, while single-chunk segments write their result directly.

Softmax keeps online statistics (a running maximum and a compensated,
rescaled exponential sum) and reads the rows twice. Sums are compensated.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from functools import lru_cache
import os
from pathlib import Path
import subprocess
import time

# Triton 3.8 bundles an older assembler; use the CUDA 13.2 toolkit pinned in
# environment.lock instead.
os.environ.setdefault('TRITON_PTXAS_PATH', '/usr/local/cuda-13.2/bin/ptxas')
os.environ.setdefault('TRITON_LIBDEVICE_PATH', '/usr/local/cuda-13.2/nvvm/libdevice/libdevice.10.bc')
import triton
import triton.language as tl

if triton.__version__ != '3.8.0':
    raise RuntimeError('comparison baseline requires pinned Triton 3.8.0')

OPS = {'sum': 0, 'max': 1, 'softmax': 2}
OPTIONS = dict(enable_fp_fusion=False, enable_reflect_ftz=False)


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
def _kahan(total, carry, x):
    # Compensated addition that keeps ordinary inf/NaN behavior.
    y = x - carry
    new = total + y
    finite = (new == new) & (tl.abs(new) != float('inf'))
    return new, tl.where(finite, (new - total) - y, 0.0)


@triton.jit
def _fold(X, row0, row1, feats, fmask, D: tl.constexpr, OP: tl.constexpr,
          BLOCK_ROWS: tl.constexpr, BLOCK_D: tl.constexpr):
    """Reduce rows [row0, row1): sum, max, or softmax (max, exp-sum) statistics."""
    lanes = tl.arange(0, BLOCK_ROWS).to(tl.int64)
    if OP == 0:
        acc = tl.zeros((BLOCK_ROWS, BLOCK_D), tl.float32)
        carry = tl.zeros((BLOCK_ROWS, BLOCK_D), tl.float32)
        for start in range(row0, row1, BLOCK_ROWS):
            rows = start + lanes
            mask = (rows[:, None] < row1) & fmask[None, :]
            x = tl.load(X + rows[:, None] * D + feats[None, :], mask, other=0.0).to(tl.float32)
            acc, carry = _kahan(acc, carry, x)
        first = tl.sum(acc, axis=0)
        second = tl.zeros((BLOCK_D,), tl.float32)
    elif OP == 1:
        acc = tl.full((BLOCK_ROWS, BLOCK_D), float('-inf'), tl.float32)
        for start in range(row0, row1, BLOCK_ROWS):
            rows = start + lanes
            mask = (rows[:, None] < row1) & fmask[None, :]
            x = tl.load(X + rows[:, None] * D + feats[None, :], mask, other=float('-inf')).to(tl.float32)
            acc = tl.maximum(acc, x, propagate_nan=tl.PropagateNan.ALL)
        first = tl.reduce(acc, 0, _maximum_nan)
        second = tl.zeros((BLOCK_D,), tl.float32)
    else:
        first = tl.full((BLOCK_D,), float('-inf'), tl.float32)   # running maximum
        second = tl.zeros((BLOCK_D,), tl.float32)                # exp-sum against it
        carry = tl.zeros((BLOCK_D,), tl.float32)
        for start in range(row0, row1, BLOCK_ROWS):
            rows = start + lanes
            mask = (rows[:, None] < row1) & fmask[None, :]
            x = tl.load(X + rows[:, None] * D + feats[None, :], mask, other=float('-inf')).to(tl.float32)
            new = _maximum_nan(first, tl.reduce(x, 0, _maximum_nan))
            scale = tl.where(first == new, 1.0, tl.exp(first - new))
            weights = tl.where(x == float('-inf'), 0.0, tl.exp(x - new[None, :]))
            second, carry = _kahan(second * scale, carry * scale, tl.sum(weights, axis=0))
            first = new
    return first, second


@triton.jit
def _write_softmax(X, Out, row0, row1, feats, fmask, maximum, total, D: tl.constexpr,
                   BLOCK_ROWS: tl.constexpr):
    lanes = tl.arange(0, BLOCK_ROWS).to(tl.int64)
    for start in range(row0, row1, BLOCK_ROWS):
        rows = start + lanes
        mask = (rows[:, None] < row1) & fmask[None, :]
        x = tl.load(X + rows[:, None] * D + feats[None, :], mask, other=0.0).to(tl.float32)
        tl.store(Out + rows[:, None] * D + feats[None, :], tl.exp(x - maximum[None, :]) / total[None, :], mask)


@triton.jit
def _segment_kernel(X, Offsets, Out, D: tl.constexpr, OP: tl.constexpr,
                    BLOCK_ROWS: tl.constexpr, BLOCK_D: tl.constexpr):
    segment = tl.program_id(0).to(tl.int64)
    feats = tl.program_id(1).to(tl.int64) * BLOCK_D + tl.arange(0, BLOCK_D)
    fmask = feats < D
    begin = tl.load(Offsets + segment).to(tl.int64)
    end = tl.load(Offsets + segment + 1).to(tl.int64)
    first, second = _fold(X, begin, end, feats, fmask, D, OP, BLOCK_ROWS, BLOCK_D)
    if OP == 2:
        _write_softmax(X, Out, begin, end, feats, fmask, first, second, D, BLOCK_ROWS)
    else:
        tl.store(Out + segment * D + feats, first, fmask)


@triton.jit
def _map_kernel(Offsets, Starts, ChunkSegment, B, CHUNK: tl.constexpr, TILE: tl.constexpr):
    """Starts[s] = first chunk of segment s (max(1, ceil(len/CHUNK)) each); the table maps chunks to segments."""
    carry = tl.full([], 0, tl.int64)
    for base in range(0, B, TILE):
        index = base + tl.arange(0, TILE).to(tl.int64)
        valid = index < B
        begin = tl.load(Offsets + index, valid, other=0).to(tl.int64)
        end = tl.load(Offsets + index + 1, valid, other=0).to(tl.int64)
        count = tl.where(valid, tl.maximum((end - begin + CHUNK - 1) // CHUNK, 1), 0)
        start = carry + tl.cumsum(count, 0) - count
        tl.store(Starts + index, start, valid)
        for k in range(0, tl.max(count, 0)):
            tl.store(ChunkSegment + start + k, index.to(tl.int32), valid & (k < count))
        carry += tl.sum(count, 0)
    tl.store(Starts + B, carry)


@triton.jit
def _chunk_kernel(X, Offsets, Starts, ChunkSegment, Out, PartA, PartB, B, D: tl.constexpr, OP: tl.constexpr,
                  CHUNK: tl.constexpr, BLOCK_ROWS: tl.constexpr, BLOCK_D: tl.constexpr):
    chunk = tl.program_id(0).to(tl.int64)
    feats = tl.program_id(1).to(tl.int64) * BLOCK_D + tl.arange(0, BLOCK_D)
    fmask = feats < D
    if chunk < tl.load(Starts + B):
        segment = tl.load(ChunkSegment + chunk).to(tl.int64)
        first_chunk = tl.load(Starts + segment)
        count = tl.load(Starts + segment + 1) - first_chunk
        begin = tl.load(Offsets + segment).to(tl.int64)
        end = tl.load(Offsets + segment + 1).to(tl.int64)
        row0 = begin + (chunk - first_chunk) * CHUNK
        row1 = tl.minimum(end, row0 + CHUNK)
        first, second = _fold(X, row0, row1, feats, fmask, D, OP, BLOCK_ROWS, BLOCK_D)
        if count == 1:
            if OP == 2:
                _write_softmax(X, Out, row0, row1, feats, fmask, first, second, D, BLOCK_ROWS)
            else:
                tl.store(Out + segment * D + feats, first, fmask)
        else:
            tl.store(PartA + chunk * D + feats, first, fmask)
            if OP == 2:
                tl.store(PartB + chunk * D + feats, second, fmask)


@triton.jit
def _combine_kernel(Starts, Out, PartA, PartB, StatMax, StatSum, D: tl.constexpr, OP: tl.constexpr,
                    BLOCK_C: tl.constexpr, BLOCK_D: tl.constexpr):
    segment = tl.program_id(0).to(tl.int64)
    feats = tl.program_id(1).to(tl.int64) * BLOCK_D + tl.arange(0, BLOCK_D)
    fmask = feats < D
    first_chunk = tl.load(Starts + segment)
    last_chunk = tl.load(Starts + segment + 1)
    if last_chunk - first_chunk > 1:
        lanes = tl.arange(0, BLOCK_C).to(tl.int64)
        if OP == 0:
            acc = tl.zeros((BLOCK_C, BLOCK_D), tl.float32)
            carry = tl.zeros((BLOCK_C, BLOCK_D), tl.float32)
            for base in range(first_chunk, last_chunk, BLOCK_C):
                chunks = base + lanes
                mask = (chunks[:, None] < last_chunk) & fmask[None, :]
                acc, carry = _kahan(acc, carry, tl.load(PartA + chunks[:, None] * D + feats[None, :], mask, other=0.0))
            tl.store(Out + segment * D + feats, tl.sum(acc, axis=0), fmask)
        elif OP == 1:
            acc = tl.full((BLOCK_C, BLOCK_D), float('-inf'), tl.float32)
            for base in range(first_chunk, last_chunk, BLOCK_C):
                chunks = base + lanes
                mask = (chunks[:, None] < last_chunk) & fmask[None, :]
                part = tl.load(PartA + chunks[:, None] * D + feats[None, :], mask, other=float('-inf'))
                acc = tl.maximum(acc, part, propagate_nan=tl.PropagateNan.ALL)
            tl.store(Out + segment * D + feats, tl.reduce(acc, 0, _maximum_nan), fmask)
        else:
            maximum = tl.full((BLOCK_D,), float('-inf'), tl.float32)
            total = tl.zeros((BLOCK_D,), tl.float32)
            carry = tl.zeros((BLOCK_D,), tl.float32)
            for base in range(first_chunk, last_chunk, BLOCK_C):
                chunks = base + lanes
                mask = (chunks[:, None] < last_chunk) & fmask[None, :]
                part_max = tl.load(PartA + chunks[:, None] * D + feats[None, :], mask, other=float('-inf'))
                part_sum = tl.load(PartB + chunks[:, None] * D + feats[None, :], mask, other=0.0)
                new = _maximum_nan(maximum, tl.reduce(part_max, 0, _maximum_nan))
                scale = tl.where(maximum == new, 1.0, tl.exp(maximum - new))
                mass = tl.where(part_max == float('-inf'), 0.0, part_sum * tl.exp(part_max - new[None, :]))
                total, carry = _kahan(total * scale, carry * scale, tl.sum(mass, axis=0))
                maximum = new
            tl.store(StatMax + segment * D + feats, maximum, fmask)
            tl.store(StatSum + segment * D + feats, total, fmask)


@triton.jit
def _normalize_kernel(X, Offsets, Starts, ChunkSegment, StatMax, StatSum, Out, B, D: tl.constexpr,
                      CHUNK: tl.constexpr, BLOCK_ROWS: tl.constexpr, BLOCK_D: tl.constexpr):
    chunk = tl.program_id(0).to(tl.int64)
    feats = tl.program_id(1).to(tl.int64) * BLOCK_D + tl.arange(0, BLOCK_D)
    fmask = feats < D
    if chunk < tl.load(Starts + B):
        segment = tl.load(ChunkSegment + chunk).to(tl.int64)
        first_chunk = tl.load(Starts + segment)
        if tl.load(Starts + segment + 1) - first_chunk > 1:
            row0 = tl.load(Offsets + segment).to(tl.int64) + (chunk - first_chunk) * CHUNK
            row1 = tl.minimum(tl.load(Offsets + segment + 1).to(tl.int64), row0 + CHUNK)
            maximum = tl.load(StatMax + segment * D + feats, fmask, other=0.0)
            total = tl.load(StatSum + segment * D + feats, fmask, other=1.0)
            _write_softmax(X, Out, row0, row1, feats, fmask, maximum, total, D, BLOCK_ROWS)


@dataclass(frozen=True)
class Config:
    mapping: str            # 'segment' or 'chunked'
    block_rows: int
    block_d: int
    num_warps: int
    chunk: int | None = None


def candidates(d):
    """Configurations tried by the tuner for feature width d."""
    widest = triton.next_power_of_2(d)
    configs = []
    for block_d in sorted({min(widest, 32), min(widest, 128)}):
        for elements in (1024, 4096):
            rows = max(16, elements // block_d)
            configs += [Config('segment', rows, block_d, warps) for warps in (4, 8)]
            configs += [Config('chunked', rows, block_d, 4, rows * iterations) for iterations in (4, 32)]
    return configs


_tuned: dict[tuple, dict] = {}
_last: dict | None = None


def last_tuning():
    """The configuration and timings chosen for the most recent call's shape."""
    return _last


def _run(config, values, offsets, op, output):
    import torch
    n, d = values.shape
    b = offsets.numel() - 1
    tiles = triton.cdiv(d, config.block_d)
    shape = dict(D=d, OP=op, BLOCK_D=config.block_d)
    if config.mapping == 'segment':
        _segment_kernel[(b, tiles)](values, offsets, output, BLOCK_ROWS=config.block_rows,
                                    num_warps=config.num_warps, **shape, **OPTIONS)
        return
    chunks = b + triton.cdiv(n, config.chunk)
    starts = torch.empty((b + 1,), dtype=torch.int64, device=values.device)
    table = torch.empty((chunks,), dtype=torch.int32, device=values.device)
    _map_kernel[(1,)](offsets, starts, table, b, CHUNK=config.chunk, TILE=1024, num_warps=4)
    part_max = torch.empty((chunks, d), dtype=torch.float32, device=values.device)
    part_sum = torch.empty_like(part_max) if op == 2 else part_max
    _chunk_kernel[(chunks, tiles)](values, offsets, starts, table, output, part_max, part_sum, b,
                                   CHUNK=config.chunk, BLOCK_ROWS=config.block_rows,
                                   num_warps=config.num_warps, **shape, **OPTIONS)
    stat_max = torch.empty((b, d), dtype=torch.float32, device=values.device) if op == 2 else part_max
    stat_sum = torch.empty_like(stat_max) if op == 2 else part_max
    _combine_kernel[(b, tiles)](starts, output, part_max, part_sum, stat_max, stat_sum,
                                BLOCK_C=32, num_warps=4, **shape, **OPTIONS)
    if op == 2:
        _normalize_kernel[(chunks, tiles)](values, offsets, starts, table, stat_max, stat_sum, output, b,
                                           D=d, CHUNK=config.chunk, BLOCK_ROWS=config.block_rows,
                                           BLOCK_D=config.block_d, num_warps=config.num_warps, **OPTIONS)


def _tune(values, offsets, op, output, warmup=2, repeats=5):
    """Time every candidate on this input and keep the fastest (median of repeats)."""
    import torch
    started = time.perf_counter()
    results = []
    for config in candidates(values.shape[1]):
        try:
            for _ in range(warmup):
                _run(config, values, offsets, op, output)
            samples = []
            for _ in range(repeats):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                _run(config, values, offsets, op, output)
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end))
            results.append((sorted(samples)[len(samples) // 2], config))
        except triton.runtime.errors.OutOfResources:
            continue
    if not results:
        raise RuntimeError('no Triton configuration fits this device')
    best_ms, best = min(results, key=lambda item: item[0])
    return {'config': asdict(best), 'median_ms': best_ms, 'candidates': len(results),
            'tuning_wall_ms': (time.perf_counter() - started) * 1e3,
            'timings': [{'config': asdict(config), 'median_ms': ms} for ms, config in results]}


def call(values, offsets, operation, config: Config | None = None):
    """Run the reference; without a config, the first call per shape autotunes."""
    global _last
    import torch
    toolchain()
    if operation not in OPS:
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
    if b + n > 2**31 - 1:
        raise ValueError('segment and chunk counts exceed the CUDA grid x limit')
    op = OPS[operation]
    with torch.cuda.device(values.device):
        if tuple(torch.cuda.get_device_capability(values.device)) != (8, 9):
            raise ValueError('benchmark target must be sm_89')
        stream = torch.cuda.current_stream(values.device)
        output = torch.empty((n if operation == 'softmax' else b, d), dtype=values.dtype, device=values.device)
        if b == 0:
            return output
        if config is None:
            key = (operation, str(values.dtype), str(offsets.dtype), n, b, d)
            if key not in _tuned:
                _tuned[key] = _tune(values, offsets, op, output)
            _last = _tuned[key]
            config = Config(**_last['config'])
        _run(config, values, offsets, op, output)
        values.record_stream(stream)
        offsets.record_stream(stream)
        return output
