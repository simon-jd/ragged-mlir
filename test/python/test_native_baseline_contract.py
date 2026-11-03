"""Real native-library loading and host preflight; never claims device success."""
import ctypes
from pathlib import Path

import pytest
from baselines.native import NativeBaselines


@pytest.fixture(scope='module')
def baseline():
    path = Path('build-baselines/libragged_baselines.so')
    if not path.exists():
        pytest.skip('comparison library has not been built')
    return NativeBaselines(path)


def test_actual_library_pinned_version_and_provenance(baseline):
    assert baseline.lib.ragged_baseline_version() == 300500
    assert len(baseline.provenance['library_sha256']) == 64
    assert baseline.provenance['build']['exit_code'] == 0
    assert baseline.provenance['build']['source_unchanged']


@pytest.mark.parametrize('op,bits,b,d', [(3, 64, 1, 1), (0, 16, 1, 1), (0, 64, -1, 1), (0, 64, 1, 0)])
def test_native_invalid_scalar_contract_rejected_before_cuda(baseline, op, bits, b, d):
    assert baseline.lib.ragged_naive(op, bits, b, d, None, None, None, None) != 0


def test_native_empty_calls_need_no_device(baseline):
    assert baseline.lib.ragged_naive(0, 64, 0, 1, None, None, None, None) == 0
    count = ctypes.c_size_t(42)
    assert baseline.lib.ragged_cub_reduce(0, 64, 0, None, None, None, None, ctypes.byref(count), None) == 0
    assert count.value == 0


def test_feature_major_cub_rejects_invalid_shapes_and_needs_no_device_when_empty(baseline):
    count = ctypes.c_size_t(42)
    assert baseline.lib.ragged_cub_reduce_features(0, 64, 0, 0, 128, None, None, None, None,
                                                   ctypes.byref(count), None) == 0
    assert count.value == 0
    # Softmax is not a CUB reduction, and D*B must fit CUB's i32 segment count.
    assert baseline.lib.ragged_cub_reduce_features(2, 64, 1, 1, 128, None, None, None, None,
                                                   ctypes.byref(count), None) != 0
    assert baseline.lib.ragged_cub_reduce_features(0, 64, 2**24, 1, 1024, None, None, None, None,
                                                   ctypes.byref(count), None) != 0


def test_native_adapter_rejects_real_cpu_tensors(baseline):
    torch = pytest.importorskip('torch')
    with pytest.raises(ValueError, match='same CUDA'):
        baseline.call('cuda-naive', torch.ones(4, 1), torch.tensor([0, 4]), 'sum')
