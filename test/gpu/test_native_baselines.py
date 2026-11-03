"""Execute comparison-only baselines against the same independent oracle."""
import os
import numpy as np
import pytest

from baselines.native import NativeBaselines
from oracle import assert_matches, quantize_bf16


@pytest.fixture(scope='module')
def native_environment():
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('native baseline correctness unverified: no CUDA device')
    if tuple(torch.cuda.get_device_capability()) != (8, 9):
        pytest.skip('native baseline requires sm_89')
    return torch, NativeBaselines(os.environ.get('RAGGED_BASELINE_LIBRARY',
                                                 'build-baselines/libragged_baselines.so'))


@pytest.mark.parametrize('method,d', [('cuda-naive', 1), ('cuda-naive', 7),
                                     ('cuda-naive', 128), ('cuda-naive', 1024), ('cub', 1),
                                     ('cub', 7), ('cub', 128)])
@pytest.mark.parametrize('op', ['sum', 'max', 'softmax'])
@pytest.mark.parametrize('dtype', ['f32', 'bf16'])
@pytest.mark.parametrize('bits', [32, 64])
def test_native_baseline_mixed_lengths(native_environment, method, d, op, dtype, bits):
    torch, adapter = native_environment
    offsets = np.array([0, 0, 1, 33, 4130, 4130, 4161], dtype=f'i{bits//8}')
    values = np.random.default_rng(821).normal(size=(4161, d)).astype(np.float32)
    if dtype == 'bf16':
        values = quantize_bf16(values)
    tensor = torch.from_numpy(values).to('cuda', dtype=torch.float32 if dtype == 'f32' else torch.bfloat16)
    bounds = torch.from_numpy(offsets).to('cuda')
    output = adapter.call(method, tensor, bounds, op)
    assert_matches(output.float().cpu().numpy(), values, offsets, operation=op, input_dtype=dtype)


@pytest.mark.parametrize('method', ['cuda-naive', 'cub'])
@pytest.mark.parametrize('op', ['sum', 'max', 'softmax'])
@pytest.mark.parametrize('offsets,values', [
    ([0], []), ([0, 0, 0], []),
    ([0, 2, 4, 6, 8, 8], [np.nan, 1, np.inf, -np.inf, -np.inf, -np.inf, -3, -5]),
    ([0, 131073], [0.1] * 131073),
])
def test_native_baseline_boundaries(native_environment, method, op, offsets, values):
    torch, adapter = native_environment
    values = np.array(values, dtype=np.float32).reshape(-1, 1)
    offsets = np.array(offsets, dtype=np.int64)
    # All work is submitted on a nondefault stream, including input creation.
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        tensor = torch.from_numpy(values).to('cuda')
        bounds = torch.from_numpy(offsets).to('cuda')
        output = adapter.call(method, tensor, bounds, op)
    stream.synchronize()
    assert_matches(output.float().cpu().numpy(), values, offsets, operation=op, input_dtype='f32')
