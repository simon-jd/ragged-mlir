"""Comparison-only CUDA and CUB adapters. No host offsets copy or CPU fallback.

All allocation, f32 promotion, library temporary storage, and final dtype casts
are performed inside the callable so the benchmark measures the full cost.
"""
from __future__ import annotations
import ctypes
import hashlib
import json
from pathlib import Path

OP_CODES = {'sum': 0, 'max': 1, 'softmax': 2}


class NativeBaselines:
    def __init__(self, library='build-baselines/libragged_baselines.so'):
        self.path = Path(library).resolve()
        self.lib = ctypes.CDLL(str(self.path))
        ptr, integer, wide = ctypes.c_void_p, ctypes.c_int, ctypes.c_int64
        self.lib.ragged_baseline_version.restype = integer
        self.lib.ragged_baseline_error.argtypes = [integer]
        self.lib.ragged_baseline_error.restype = ctypes.c_char_p
        self.lib.ragged_naive.argtypes = [integer, integer, wide, wide, ptr, ptr, ptr, ptr]
        self.lib.ragged_cub_reduce.argtypes = [integer, integer, wide, ptr, ptr, ptr, ptr,
                                             ctypes.POINTER(ctypes.c_size_t), ptr]
        self.lib.ragged_cub_shifted_exp.argtypes = [integer, wide, wide, ptr, ptr, ptr, ptr, ptr, ptr]
        self.lib.ragged_cub_normalize.argtypes = [wide, ptr, ptr, ptr, ptr, ptr]
        self.lib.ragged_cub_reduce_features.argtypes = [integer, integer, wide, wide, wide, ptr, ptr, ptr, ptr,
                                                        ctypes.POINTER(ctypes.c_size_t), ptr]
        self.lib.ragged_cub_shifted_exp_features.argtypes = [integer, wide, wide, wide, ptr, ptr, ptr, ptr, ptr, ptr]
        for name in ('ragged_naive', 'ragged_cub_reduce', 'ragged_cub_shifted_exp', 'ragged_cub_normalize',
                     'ragged_cub_reduce_features', 'ragged_cub_shifted_exp_features'):
            getattr(self.lib, name).restype = integer
        version = self.lib.ragged_baseline_version()
        if version != 300500:
            raise ValueError(f'baseline requires CCCL 3.5.0, observed CUB_VERSION={version}')
        build = self.path.parent / 'build.json'
        self.provenance = {'library_sha256': hashlib.sha256(self.path.read_bytes()).hexdigest(),
                           'cccl': '3.5.0', 'implementation': 'comparison-only native CUDA/CUB'}
        if build.exists():
            manifest = json.loads(build.read_text())
            if manifest.get('library_sha256') != self.provenance['library_sha256']:
                raise ValueError('baseline library differs from its build manifest')
            self.provenance['build'] = manifest

    def _check(self, code):
        if code:
            raise RuntimeError(self.lib.ragged_baseline_error(code).decode())

    def call(self, method, values, offsets, op):
        import torch
        if method not in ('cuda-naive', 'cub') or op not in OP_CODES:
            raise ValueError('expected cuda-naive/cub and sum/max/softmax')
        if (values.ndim != 2 or offsets.ndim != 1 or offsets.numel() < 1
                or not values.is_contiguous() or not offsets.is_contiguous()):
            raise ValueError('baseline requires contiguous values[N,D] and offsets[B+1]')
        if not values.is_cuda or values.device != offsets.device:
            raise ValueError('baseline inputs must be on the same CUDA device')
        if values.dtype not in (torch.float32, torch.bfloat16) or offsets.dtype not in (torch.int32, torch.int64):
            raise ValueError('baseline requires f32/bf16 and i32/i64 offsets')
        n, d = values.shape
        b = offsets.numel() - 1
        if not 1 <= d <= 1024 or (b == 0 and n != 0) or n * d > (2**63 - 1) // 4:
            raise ValueError('baseline shape violates contiguous forward contract')
        if values.requires_grad:
            raise ValueError('baseline is forward-only')
        if method == 'cub' and d > 1 and b * d > 2**31 - 1:
            raise ValueError('CUB segment count D*B exceeds i32')
        if offsets.dtype == torch.int32 and n > 2**31 - 1:
            raise ValueError('N exceeds i32 offsets range')
        bits = 32 if offsets.dtype == torch.int32 else 64
        with torch.cuda.device(values.device):
            if tuple(torch.cuda.get_device_capability(values.device)) != (8, 9):
                raise ValueError('native baseline compiled only for sm_89')
            stream = torch.cuda.current_stream(values.device)
            ptr = stream.cuda_stream
            promoted = values.float()
            output = torch.empty((n if op == 'softmax' else b, d), dtype=torch.float32, device=values.device)
            if b == 0:
                return output.to(values.dtype)
            if method == 'cuda-naive':
                self._check(self.lib.ragged_naive(OP_CODES[op], bits, b, d, promoted.data_ptr(),
                                                  offsets.data_ptr(), output.data_ptr(), ptr))
            elif d > 1:
                output = self._cub_features(OP_CODES[op], bits, b, n, d, promoted, offsets, ptr, torch)
            elif op != 'softmax':
                self._reduce(OP_CODES[op], bits, b, promoted, offsets, output, ptr, torch)
            else:
                # CUB max, element-parallel exp(x - max), CUB sum, element-parallel division.
                maximum = torch.empty((b, 1), dtype=torch.float32, device=values.device)
                total = torch.empty_like(maximum)
                exp_values = torch.empty_like(promoted)
                segment_ids = torch.empty((n,), dtype=torch.int32, device=values.device)
                self._reduce(1, bits, b, promoted, offsets, maximum, ptr, torch)
                self._check(self.lib.ragged_cub_shifted_exp(bits, b, n, promoted.data_ptr(), offsets.data_ptr(),
                                                            maximum.data_ptr(), exp_values.data_ptr(),
                                                            segment_ids.data_ptr(), ptr))
                self._reduce(0, bits, b, exp_values, offsets, total, ptr, torch)
                self._check(self.lib.ragged_cub_normalize(n, exp_values.data_ptr(), segment_ids.data_ptr(),
                                                          total.data_ptr(), output.data_ptr(), ptr))
            # Foreign kernels do not automatically notify Torch's allocator.
            # Temporaries are allocated and consumed on this same current stream.
            values.record_stream(stream)
            offsets.record_stream(stream)
            return output.to(values.dtype)

    def _reduce(self, op, bits, b, values, offsets, output, stream, torch):
        count = ctypes.c_size_t(0)
        base = [op, bits, b, values.data_ptr(), offsets.data_ptr(), output.data_ptr()]
        self._check(self.lib.ragged_cub_reduce(*base, None, ctypes.byref(count), stream))
        if count.value > 2**63 - 1:
            raise ValueError('CUB workspace size exceeds signed i64')
        workspace = torch.empty((max(1, count.value),), dtype=torch.uint8, device=values.device)
        self._check(self.lib.ragged_cub_reduce(*base, workspace.data_ptr(), ctypes.byref(count), stream))

    def _cub_features(self, op, bits, b, n, d, values, offsets, stream, torch):
        # CUB reduces contiguous runs, so D > 1 transposes to feature-major
        # [D, N], reduces the D*B runs (feature f of segment b is run f*B + b)
        # and transposes the [D, B] result, or for softmax the [D, N] output, back.
        columns = values.t().contiguous()
        if op != 2:
            result = torch.empty((d, b), dtype=torch.float32, device=values.device)
            self._reduce_features(op, bits, b, n, d, columns, offsets, result, stream, torch)
            return result.t().contiguous()
        maximum = torch.empty((d, b), dtype=torch.float32, device=values.device)
        total = torch.empty_like(maximum)
        exp_values = torch.empty_like(columns)
        segment_ids = torch.empty((d, n), dtype=torch.int32, device=values.device)
        self._reduce_features(1, bits, b, n, d, columns, offsets, maximum, stream, torch)
        self._check(self.lib.ragged_cub_shifted_exp_features(bits, b, n, d, columns.data_ptr(), offsets.data_ptr(),
                                                             maximum.data_ptr(), exp_values.data_ptr(),
                                                             segment_ids.data_ptr(), stream))
        self._reduce_features(0, bits, b, n, d, exp_values, offsets, total, stream, torch)
        normalized = torch.empty_like(columns)
        self._check(self.lib.ragged_cub_normalize(n * d, exp_values.data_ptr(), segment_ids.data_ptr(),
                                                  total.data_ptr(), normalized.data_ptr(), stream))
        return normalized.t().contiguous()

    def _reduce_features(self, op, bits, b, n, d, values, offsets, output, stream, torch):
        count = ctypes.c_size_t(0)
        base = [op, bits, b, n, d, values.data_ptr(), offsets.data_ptr(), output.data_ptr()]
        self._check(self.lib.ragged_cub_reduce_features(*base, None, ctypes.byref(count), stream))
        if count.value > 2**63 - 1:
            raise ValueError('CUB workspace size exceeds signed i64')
        workspace = torch.empty((max(1, count.value),), dtype=torch.uint8, device=values.device)
        self._check(self.lib.ragged_cub_reduce_features(*base, workspace.data_ptr(), ctypes.byref(count), stream))
