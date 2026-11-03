"""Optional FBGEMM_GPU 1.9.0 comparison adapter.

Two public operators are invoked: fbgemm::jagged_softmax (softmax at any D)
and fbgemm::segment_sum_csr (sum at D=1; at D>1 it sums every feature of a
segment together, which is not a per-feature segment sum).
Inputs follow the benchmark's validated-offset contract; supplied max_len must
already bound every segment. No device offsets are inspected on the host.
Every numeric result still requires the common oracle.
"""
from __future__ import annotations

import copy
from functools import lru_cache
import hashlib
import importlib
from importlib import metadata
from pathlib import Path
import re

PINNED_VERSION = "1.9.0"
INT32_MAX = 2**31 - 1
# The i32-offset kernel computes (length + 127) before division by 128.
MAX_LENGTH = INT32_MAX - 127
KNOWN_RISKS = (
    "CUDA >= 13 uses cuda::maximum<index_t> in the scalar_t block maximum: logits outside the offsets' "
    "integer range, or +inf after a finite logit, give NaN (test/gpu/test_fbgemm_baseline.py)",
    "FBGEMM_GPU publishes no CUDA 13.2 build; the 1.9.0+cu130 wheel runs on the CUDA 13.2 runtime "
    "of Torch 2.14.1+cu132 under CUDA minor-version compatibility",
    "upstream max_L truncates computation and leaves excluded output rows unwritten",
)


class OptionalDependencyUnavailable(RuntimeError):
    """Only the optional fbgemm-gpu distribution is absent; no fallback runs."""


def _release(version) -> str:
    if not isinstance(version, str) or re.fullmatch(r"1\.9\.0(?:\+[A-Za-z0-9._-]+)?", version) is None:
        raise RuntimeError(f"FBGEMM comparison requires release 1.9.0, observed {version!r}")
    return version


@lru_cache(maxsize=1)
def _load():
    try:
        package_version = metadata.version("fbgemm-gpu")
    except metadata.PackageNotFoundError as error:
        raise OptionalDependencyUnavailable("optional fbgemm-gpu==1.9.0 distribution is not installed") from error
    _release(package_version)
    # Import errors/undefined symbols are real failures, not dependency absence.
    package = importlib.import_module("fbgemm_gpu")
    reported_version = _release(getattr(package, "__version__", None))
    torch = importlib.import_module("torch")
    try:
        operator = torch.ops.fbgemm.jagged_softmax
    except AttributeError as error:
        raise RuntimeError("installed FBGEMM package lacks public fbgemm::jagged_softmax") from error
    module_file = Path(package.__file__).resolve()
    libraries = sorted({p.resolve() for p in module_file.parent.rglob("*.so") if p.is_file()})
    if not libraries:
        raise RuntimeError("FBGEMM installation contains no auditable native libraries")
    provenance = {
        "fbgemm_gpu": package_version, "reported_module_version": reported_version,
        "torch": str(torch.__version__), "torch_cuda": torch.version.cuda,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "package_init": str(module_file), "package_init_sha256": hashlib.sha256(module_file.read_bytes()).hexdigest(),
        # This is the package library inventory, not a claim that every file is
        # loaded by this operation. Hashes identify the installed binaries.
        "native_libraries": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in libraries],
        "native_libraries_scope": "installed package inventory; no claim every file is loaded by this operation",
        "compatibility": "unverified", "device_semantics": "unqualified",
        "device_semantics_scope": "the general input domain is unqualified; each benchmark case must separately pass the fixed oracle",
        "operators": ["fbgemm::jagged_softmax", "fbgemm::segment_sum_csr"], "known_risks": list(KNOWN_RISKS),
    }
    return operator, provenance


def toolchain() -> dict:
    """Load the actual optional package and report identity, never qualification."""
    return copy.deepcopy(_load()[1])


def configuration(n: int, d: int, b: int, *, max_len: int | None = None) -> dict:
    """Shape-only admission and explicit no-truncation premise for a call."""
    if any(type(x) is not int or x < 0 for x in (n, d, b)):
        raise ValueError("FBGEMM dimensions must be nonnegative integers")
    if not 1 <= d <= 1024 or (b == 0 and n != 0):
        raise ValueError("FBGEMM requires D in [1,1024] and valid packed batch shape")
    if n * d > INT32_MAX or b + 1 > INT32_MAX or n > MAX_LENGTH:
        raise ValueError("FBGEMM requires 32-bit accessor and length-rounding bounds")
    if max_len is not None and (type(max_len) is not int or not 0 <= max_len <= MAX_LENGTH):
        raise ValueError("max_len must be an integer in the safe signed-i32 range")
    if max_len == 0 and n != 0:
        raise ValueError("max_len=0 would truncate a nonempty packed input")
    return {
        "max_L": n if max_len is None else max_len,
        "max_L_source": "shape_N_upper_bound" if max_len is None else "caller_verified_length_bound",
        "offsets_contract": "caller validated [0,N] monotone offsets with endpoints 0 and N",
        "no_truncation": "max_L=N is unconditional for valid offsets" if max_len is None else
                         "caller must have verified every segment length <= max_len",
        "accumulation_dtype": "f32", "bf16_promotion_in_call": True,
        "empty_input": "packed empty output allocation; no foreign kernel",
        "device_semantics": "unqualified", "known_risks": list(KNOWN_RISKS),
    }


def _invoke(values, offsets, max_L, operator):
    # Kept separate from preflight so host tests can inspect dtype/argument
    # routing without simulating CUDA or claiming foreign numerical execution.
    if values.shape[0] == 0:
        return values.new_empty(values.shape)
    work = values.float()
    result = operator(work, offsets, max_L)
    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError("fbgemm::jagged_softmax must return (values, offsets)")
    output, returned_offsets = result
    if (output.shape != values.shape or output.dtype != work.dtype or output.device != values.device
            or returned_offsets.shape != offsets.shape or returned_offsets.dtype != offsets.dtype
            or returned_offsets.device != offsets.device or returned_offsets.data_ptr() != offsets.data_ptr()):
        raise RuntimeError("FBGEMM output violates its public values/offsets contract")
    return output.to(values.dtype)


def call(values, offsets, operation: str = "softmax", *, max_len: int | None = None):
    """Complete forward call, including f32 promotion and source-dtype cast.

    Passing a smaller bound is legal only when the caller has validated it from
    the original workload. This function never obtains such a bound by reading
    CUDA offsets. Shape/range guards cannot establish offset contents.
    """
    if operation not in ("softmax", "sum"):
        raise ValueError("FBGEMM adapter supports jagged softmax and the D=1 segment sum")
    torch = importlib.import_module("torch")
    if (values.ndim != 2 or offsets.ndim != 1 or offsets.numel() < 1
            or not values.is_contiguous() or not offsets.is_contiguous()):
        raise ValueError("FBGEMM requires contiguous values[N,D] and offsets[B+1]")
    if values.dtype not in (torch.float32, torch.bfloat16) or offsets.dtype not in (torch.int32, torch.int64):
        raise ValueError("FBGEMM adapter requires f32/bf16 values and i32/i64 offsets")
    if values.requires_grad:
        raise ValueError("FBGEMM comparison is forward-only")
    config = configuration(values.shape[0], values.shape[1], offsets.numel() - 1, max_len=max_len)
    if not values.is_cuda or values.device != offsets.device:
        raise ValueError("FBGEMM inputs must be on the same CUDA device")
    with torch.cuda.device(values.device):
        if tuple(torch.cuda.get_device_capability(values.device)) != (8, 9):
            raise ValueError("FBGEMM comparison target must be sm_89")
        operator, _ = _load()
        if operation == "sum":
            return _segment_sum(values, offsets, torch)
        return _invoke(values, offsets, config["max_L"], operator)


def _segment_sum(values, offsets, torch):
    # segment_sum_csr(batch_size, csr_seg, values) sums values[csr_seg[s] * batch_size :
    # csr_seg[s + 1] * batch_size] for each segment s; batch_size=1 is the D=1 segment sum.
    if values.shape[1] != 1:
        raise ValueError("fbgemm::segment_sum_csr sums whole segments; only D=1 is a per-feature sum")
    result = torch.ops.fbgemm.segment_sum_csr(1, offsets, values.float().view(-1))
    if result.shape != (offsets.numel() - 1,):
        raise RuntimeError("fbgemm::segment_sum_csr must return one value per segment")
    return result.view(-1, 1).to(values.dtype)
