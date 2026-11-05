"""Real CUDA NJT max baseline; compilation failures never fall back to eager."""
from __future__ import annotations

import numpy as np
import pytest

from oracle import assert_matches, quantize_bf16


@pytest.fixture(scope="module")
def njt_max_environment():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("NJT max baseline unverified: no CUDA device")
    if tuple(torch.cuda.get_device_capability()) not in ((8, 6), (8, 9)):
        pytest.skip("NJT max baseline requires sm_86 or sm_89")
    from baselines.torch_reference import njt_max
    return torch, njt_max


@pytest.fixture(autouse=True)
def fresh_dynamo(njt_max_environment):
    # Each parametrized shape is a separate fullgraph compilation; without a
    # reset they accumulate against Dynamo's per-function recompile limit.
    njt_max_environment[0]._dynamo.reset()


@pytest.mark.parametrize("dtype", ["f32", "bf16"])
@pytest.mark.parametrize("d", [1, 7, 16, 128, 1024])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("case", ["mixed_special", "negative", "all_empty", "zero_segments"])
def test_njt_max_shared_oracle(njt_max_environment, dtype, d, compiled, case):
    torch, njt_max = njt_max_environment
    offsets = np.array([0, 0, 2, 5, 6, 6], dtype=np.int64)
    values = -np.arange(1, 6*d + 1, dtype=np.float32).reshape(6, d) / 8
    if case == "mixed_special":
        values[0:2] = -np.inf
        values[2] = np.inf
        values[4, 0] = np.nan
    elif case == "all_empty":
        values, offsets = np.empty((0, d), np.float32), np.array([0, 0, 0], np.int64)
    elif case == "zero_segments":
        values, offsets = np.empty((0, d), np.float32), np.array([0], np.int64)
    if dtype == "bf16":
        values = quantize_bf16(values)
    lengths = np.diff(offsets)
    minimum = int(lengths.min()) if len(lengths) else 0
    maximum = int(lengths.max()) if len(lengths) else 0
    def function(v, o):
        return njt_max(v, o, min_len=minimum, max_len=maximum)
    operation = torch.compile(function, fullgraph=True) if compiled else function
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        tensor = torch.from_numpy(values).to("cuda", dtype=torch.float32 if dtype == "f32" else torch.bfloat16)
        bounds = torch.from_numpy(offsets).to("cuda")
        output = operation(tensor, bounds)
    stream.synchronize()
    assert output.dtype == tensor.dtype and output.device == tensor.device
    assert_matches(output.float().cpu().numpy(), values, offsets, operation="max", input_dtype=dtype)
