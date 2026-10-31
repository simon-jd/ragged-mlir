"""AOT C ABI, exact packed inputs and evidence checks for the sparse experiment."""
from __future__ import annotations

import ctypes
import hashlib
import math
import re
from pathlib import Path
import statistics
import time

ROOT = Path(__file__).resolve().parents[1]
DOMAIN = "CUDA-resident contiguous f32 values[N,1] and i64 offsets[B+1] -> CUDA-resident contiguous f32 packed output; completion included"
TIMING_SCOPE = "complete resident-input call through all-stream GPU completion; includes host work and transfers; no process/JIT startup"
CANDIDATE_COSTS = ("input_D2H", "coordinate_construction", "CSR_assembly", "host_allocations", "upstream_GPU_allocations_and_transfers",
                   "all_upstream_reduction_kernels", "host_exp_and_normalize_for_softmax", "packed_output_copy", "output_H2D", "all_stream_completion")
CUB_COSTS = ("output_and_workspace_allocations", "all_CUB_reductions", "exp_and_normalize_kernels_for_softmax", "all_stream_completion")
SOURCE_FILES = ("scripts/run_sparse_comparison.py", "scripts/sparse_comparison_worker.py", "scripts/sparse_comparison_support.py",
                "scripts/sparse_comparison.mlir.in", "scripts/reproduce_gpu.py", "bench/distributions.py", "bench/__init__.py", "baselines/native.py",
                "baselines/cuda/baselines.cu", "oracle/reference.py", "oracle/__init__.py", "environment.lock")


def array_hash(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def coordinates(offsets):
    import numpy as np
    from oracle import validate_offsets
    validate_offsets(offsets, int(offsets[-1]))
    return np.arange(int(offsets[-1]), dtype=np.int64) - np.repeat(offsets[:-1], np.diff(offsets))


def case_identity(values, offsets, coords):
    return {"values_sha256": array_hash(values), "offsets_sha256": array_hash(offsets), "coordinates_sha256": array_hash(coords),
            "B": len(offsets) - 1, "N": len(values), "D": 1, "dtype": "f32", "offsets_dtype": "i64",
            "stored_zero_count": int((values == 0).sum())}


def distribution_qualification(offsets):
    """Operational long-tail scope gate, recomputed from actual saved offsets."""
    import numpy as np
    from oracle import validate_offsets
    validate_offsets(offsets, int(offsets[-1]))
    lengths = np.diff(offsets)
    b = len(lengths)
    short = int((lengths < 32).sum())
    rare_long = int((lengths > 10000).sum())
    classes = {"warp": int((lengths <= 32).sum()), "cta": int(((lengths > 32) & (lengths <= 4096)).sum()),
               "split": int((lengths > 4096).sum())}
    return {"qualified": b > 0 and short * 2 > b and 0 < rare_long * 100 <= b and all(classes.values()),
            "B": b, "N": int(offsets[-1]), "short_lt32_count": short, "long_gt10000_count": rare_long,
            "class_counts": classes,
            "definition": "majority <32; 0 < fraction >10000 <=1%; warp/CTA/split all present; 1% is an operational rare threshold"}


def make_case(b, n, seed):
    import numpy as np
    from bench.distributions import make_lengths, offsets_from_lengths
    lengths = make_lengths("long_tail", b, seed=seed, total_elements=n, mean_length=64)
    offsets = offsets_from_lengths(lengths, "i64")
    values = np.random.default_rng(seed + 1).normal(size=(n, 1)).astype(np.float32)
    # Explicit stored zeros are real input values shared by both candidates.
    values[::257, 0] = 0.0
    return values, offsets, coordinates(offsets)


def semantic_cases():
    import numpy as np
    offsets = np.array([0, 2, 2, 4, 6, 8, 10], dtype=np.int64)
    cases = {
        "empty_nan_infinities_zero": np.array([-2, -4, np.nan, 1, -np.inf, -np.inf, 0, 2, np.inf, 1], np.float32).reshape(-1, 1),
        "positive_infinity_second": np.array([-2, -4, np.nan, 1, -np.inf, -np.inf, 0, 2, 1, np.inf], np.float32).reshape(-1, 1),
        "underflow_stored_entries": np.array([-1000, 0, 1, 2, -1000, 0, 0, 0, 1, 1], np.float32).reshape(-1, 1),
    }
    result = {name: (values, offsets.copy(), coordinates(offsets)) for name, values in cases.items()}
    result["all_empty"] = (np.empty((0, 1), np.float32), np.zeros(7, np.int64), np.empty(0, np.int64))
    return result


def render_module(b, width):
    if type(b) is not int or type(width) is not int or not 1 <= b <= 1_000_000 or not 1 <= width <= 2**31 - 1 or b * width > 2**63 - 1:
        raise ValueError("CSR shape exceeds the supported signed-index bounds")
    return (ROOT / "scripts/sparse_comparison.mlir.in").read_text().replace("@B@", str(b)).replace("@L@", str(width))


class MemRef1(ctypes.Structure):
    _fields_ = [("allocated", ctypes.c_void_p), ("aligned", ctypes.c_void_p), ("offset", ctypes.c_int64),
                ("sizes", ctypes.c_int64 * 1), ("strides", ctypes.c_int64 * 1)]


class DlInfo(ctypes.Structure):
    _fields_ = [("fname", ctypes.c_char_p), ("fbase", ctypes.c_void_p),
                ("sname", ctypes.c_char_p), ("saddr", ctypes.c_void_p)]


def callback_identity(function, expected):
    """Resolve the actual callable address, including symbol interposition."""
    from scripts.reproduce_gpu import digest
    dladdr = ctypes.CDLL(None).dladdr
    dladdr.argtypes, dladdr.restype = [ctypes.c_void_p, ctypes.POINTER(DlInfo)], ctypes.c_int
    address = ctypes.cast(function, ctypes.c_void_p).value
    info = DlInfo()
    if not dladdr(address, ctypes.byref(info)) or not info.fname:
        raise RuntimeError("cannot resolve actual loaded AOT callback")
    path = Path(info.fname.decode()).resolve()
    if path != expected:
        raise RuntimeError("AOT callback resolved to a different shared object")
    return {"path": str(path), "sha256": digest(path), "address": address,
            "symbol": info.sname.decode() if info.sname else None}


def descriptor(array):
    return MemRef1(array.ctypes.data, array.ctypes.data, 0, (ctypes.c_int64 * 1)(array.size), (ctypes.c_int64 * 1)(1))


class SparseLibrary:
    """Persistent upstream-produced AOT module; inputs are borrowed buffers.

    Only direct sparse codegen (enable-runtime-library=false) is supported.
    No values are numerically computed by this adapter.
    """
    def __init__(self, path, b, width):
        self.path, self.b, self.width = Path(path).resolve(), b, width
        started = time.perf_counter_ns()
        self.library = ctypes.CDLL(str(self.path))
        self.maximum = self.library._mlir_ciface_run_max
        self.softmax = self.library._mlir_ciface_run_softmax
        self.maximum.argtypes, self.maximum.restype = [ctypes.POINTER(MemRef1)] * 4, None
        self.softmax.argtypes, self.softmax.restype = [ctypes.POINTER(MemRef1)] * 7 + [ctypes.c_bool], None
        from scripts.reproduce_gpu import digest
        self.provenance = {"path": str(Path(self.library._name).resolve()), "sha256": digest(self.path),
            "callbacks": {name: callback_identity(function, self.path) for name, function in
                (("_mlir_ciface_run_max", self.maximum), ("_mlir_ciface_run_softmax", self.softmax))}}
        self.load_wall_ms = (time.perf_counter_ns() - started) / 1e6

    def call(self, op, values, offsets, *, audit=False):
        import numpy as np
        from oracle import validate_offsets
        if op not in ("max", "softmax") or values.dtype != np.float32 or values.shape[1:] != (1,) or not values.flags.c_contiguous:
            raise ValueError("sparse ABI requires max/softmax and contiguous f32[N,1]")
        if offsets.dtype != np.int64 or len(offsets) != self.b + 1:
            raise ValueError("offsets differ from the specialized CSR shape")
        validate_offsets(offsets, len(values))
        if np.any(np.diff(offsets) > self.width):
            raise ValueError("row coordinate exceeds the specialized CSR width")
        # Always rebuild per call; no cached host offsets/coordinates/output.
        crd = coordinates(offsets)
        out = np.empty((self.b if op == "max" else len(values), 1), dtype=np.float32)
        buffers = [values, offsets, crd, out]
        audit_result = None
        if op == "softmax":
            positions_out = np.empty(self.b + 1 if audit else 0, dtype=np.int64)
            coordinates_out = np.empty(len(values) if audit else 0, dtype=np.int64)
            count = np.empty(1 if audit else 0, dtype=np.int64)
            buffers += [positions_out, coordinates_out, count]
            descriptors = [descriptor(array) for array in buffers]
            self.softmax(*[ctypes.byref(item) for item in descriptors], audit)
            if audit:
                np.testing.assert_array_equal(positions_out, offsets)
                np.testing.assert_array_equal(coordinates_out, crd)
                if int(count[0]) != len(values):
                    raise AssertionError("sparse output dropped explicit stored entries")
                audit_result = {"positions_sha256": array_hash(positions_out), "coordinates_sha256": array_hash(coordinates_out),
                                "stored_entry_count": int(count[0]), "explicit_stored_zero_preserved": True}
        else:
            descriptors = [descriptor(array) for array in buffers]
            self.maximum(*[ctypes.byref(item) for item in descriptors])
        return out, audit_result


def candidate_cuda_callable(library, values, offsets):
    """Every hot call crosses both domains; all crossings are intentionally timed."""
    import torch
    if (not values.is_cuda or not offsets.is_cuda or values.device != offsets.device
            or values.dtype != torch.float32 or offsets.dtype != torch.int64
            or values.ndim != 2 or values.shape[1] != 1 or offsets.ndim != 1
            or offsets.numel() != library.b + 1 or not values.is_contiguous() or not offsets.is_contiguous()
            or values.requires_grad):
        raise ValueError("hybrid resident boundary requires contiguous f32[N,1] and i64[B+1] on the same CUDA device")
    def call(op):
        import torch
        host_values = values.detach().cpu().numpy()
        host_offsets = offsets.detach().cpu().numpy()
        host_out, _ = library.call(op, host_values, host_offsets, audit=False)
        return torch.from_numpy(host_out).to(device=values.device, dtype=torch.float32)
    return call


def measure(call, complete, warmup, repeats):
    for _ in range(warmup):
        output = call()
        complete()
        del output
    samples = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        output = call()
        complete()
        samples.append((time.perf_counter_ns() - started) / 1e6)
        del output
    return {"samples_ms": samples, "count": len(samples), "median_ms": statistics.median(samples),
            "warmup_calls": warmup, "scope": TIMING_SCOPE}


def valid_timing(timing, warmup, repeats):
    if not isinstance(timing, dict) or type(warmup) is not int or warmup < 0 or type(repeats) is not int or repeats < 1:
        return False
    samples = timing.get("samples_ms")
    return (timing.get("scope") == TIMING_SCOPE and isinstance(samples, list) and len(samples) == repeats and type(timing.get("count")) is int
            and timing["count"] == repeats and type(timing.get("warmup_calls")) is int and timing["warmup_calls"] == warmup
            and all(type(value) in (int, float) and math.isfinite(value) and value > 0 for value in samples)
            and type(timing.get("median_ms")) in (int, float) and math.isfinite(timing["median_ms"])
            and math.isclose(timing["median_ms"], statistics.median(samples), rel_tol=1e-12, abs_tol=1e-12))


def scope_decision(proof):
    """A directed sparse-vs-CUB recommendation, never a claim that hybrid is pure GPU."""
    reasons = []
    if proof.get("distribution_qualified") is not True:
        reasons.append("actual offsets did not qualify the required long-tail distribution")
    if proof.get("actual_device_execution") is not True or proof.get("parent_artifact_validation") is not True:
        reasons.append("actual device execution and independently bound artifacts are required")
    if proof.get("backend") != "upstream-sparse-gpu-hybrid" or proof.get("all_complete_costs_included") is not True:
        reasons.append("official hybrid identity and all complete-call costs are required")
    warmup, repeats = proof.get("warmup"), proof.get("repeats")
    if type(warmup) is not int or warmup < 0 or type(repeats) is not int or repeats < 1:
        reasons.append("requested sample counts are missing")
    identity = proof.get("input")
    if not isinstance(identity, dict) or any(not isinstance(identity.get(key), str) or re.fullmatch(r"[0-9a-f]{64}", identity[key]) is None for key in
            ("values_sha256", "offsets_sha256", "coordinates_sha256")):
        reasons.append("exact saved input hashes are missing")
    for op in ("max", "softmax"):
        if proof.get("gpu_semantic_oracle", {}).get(op) is not True:
            reasons.append(op + " GPU semantic cases did not pass for both methods")
        if proof.get("cpu_oracle", {}).get(op) is not True:
            reasons.append(op + " CPU raw-output oracle did not pass")
        for method, costs in (("candidate", CANDIDATE_COSTS), ("cub", CUB_COSTS)):
            cell = proof.get("operations", {}).get(op, {}).get(method, {})
            if (cell.get("status") != "measured" or cell.get("oracle_passed") is not True
                    or cell.get("input") != identity or cell.get("domain") != DOMAIN or cell.get("costs") != list(costs)
                    or type(repeats) is not int or type(warmup) is not int or not valid_timing(cell.get("timing"), warmup, repeats)):
                reasons.append(op + "/" + method + " lacks same-input, same-domain, oracle-qualified full-call timing")
    result = {"status": "pending", "ratios_sparse_over_cub": None, "recommendation": None, "reasons": reasons,
              "threshold": 1.5, "comparison": "strictly less than 1.5; directed sparse median complete wall / CUB median complete wall",
              "scope": "this measured D1 f32 long-tail workload only; candidate is hybrid, never pure GPU"}
    if reasons:
        return result
    ratios = {op: proof["operations"][op]["candidate"]["timing"]["median_ms"] /
              proof["operations"][op]["cub"]["timing"]["median_ms"] for op in ("max", "softmax")}
    if any(not math.isfinite(value) or value <= 0 for value in ratios.values()):
        result["reasons"].append("ratio is nonfinite or nonpositive")
        return result
    result.update(status="eligible", ratios_sparse_over_cub=ratios,
                  recommendation="consider_sparse_mapping_extension" if all(value < 1.5 for value in ratios.values())
                  else "retain_separate_scope_for_this_workload")
    return result
