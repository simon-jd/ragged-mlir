"""Pinned handwritten Triton comparison against the shared float64 oracle."""
import numpy as np
import pytest
from oracle import assert_matches, quantize_bf16


@pytest.fixture(scope='module')
def triton_environment(tmp_path_factory):
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('Triton baseline unverified: no CUDA device')
    if tuple(torch.cuda.get_device_capability()) != (8, 9):
        pytest.skip('Triton comparison requires sm_89')
    import os
    os.environ['TRITON_CACHE_DIR'] = str(tmp_path_factory.mktemp('triton-baseline'))
    from baselines import triton_reference
    triton_reference.toolchain()
    return torch, triton_reference


@pytest.mark.parametrize('op', ['sum', 'max', 'softmax'])
@pytest.mark.parametrize('dtype', ['f32', 'bf16'])
@pytest.mark.parametrize('offset_type', ['i32', 'i64'])
@pytest.mark.parametrize('d', [1, 7, 128, 1024])
def test_triton_mixed_lengths(triton_environment, op, dtype, offset_type, d):
    torch, adapter = triton_environment
    offsets = np.array([0, 0, 1, 33, 4130, 4130, 4161], dtype=np.int32 if offset_type == 'i32' else np.int64)
    values = np.random.default_rng(622).normal(size=(4161, d)).astype(np.float32)
    if dtype == 'bf16':
        values = quantize_bf16(values)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        tensor = torch.from_numpy(values).to('cuda', dtype=torch.float32 if dtype == 'f32' else torch.bfloat16)
        bounds = torch.from_numpy(offsets).to('cuda')
        result = adapter.call(tensor, bounds, op)
    stream.synchronize()
    assert_matches(result.float().cpu().numpy(), values, offsets, operation=op, input_dtype=dtype)


@pytest.mark.parametrize('op', ['sum', 'max', 'softmax'])
@pytest.mark.parametrize('offsets,values', [
    ([0], []), ([0, 0, 0], []),
    ([0, 2, 4, 6, 8, 8], [np.nan, 1, np.inf, -np.inf, -np.inf, -np.inf, -3, -5]),
    ([0, 131073], [0.1] * 131073),
])
def test_triton_boundary_semantics(triton_environment, op, offsets, values):
    torch, adapter = triton_environment
    values = np.array(values, dtype=np.float32).reshape(-1, 1)
    offsets = np.array(offsets, dtype=np.int64)
    tensor = torch.from_numpy(values).to('cuda')
    bounds = torch.from_numpy(offsets).to('cuda')
    result = adapter.call(tensor, bounds, op)
    assert_matches(result.float().cpu().numpy(), values, offsets, operation=op, input_dtype='f32')
