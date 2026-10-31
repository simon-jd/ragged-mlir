"""Real pinned toolchain and adapter preflight without a CUDA simulation."""
from pathlib import Path

import pytest


def test_triton_uses_pinned_assembler_and_records_source():
    pytest.importorskip('triton')
    from baselines.triton_reference import toolchain
    evidence = toolchain()
    assert evidence['triton'] == '3.8.0'
    assert 'V13.2.86' in evidence['ptxas_version']
    assert Path(evidence['ptxas_path']).is_file()
    assert len(evidence['libdevice_sha256']) == 64
    assert len(evidence['source_sha256']) == 64


def test_triton_adapter_rejects_real_cpu_tensors():
    pytest.importorskip('triton')
    torch = pytest.importorskip('torch')
    from baselines.triton_reference import call
    with pytest.raises(ValueError, match='same CUDA'):
        call(torch.ones(4, 1), torch.tensor([0, 4]), 'sum')
