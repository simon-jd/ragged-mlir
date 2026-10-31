"""Malformed assembler evidence must not silently become zero resource use."""
import pytest

from scripts.ptxas_resources import parse_resource_log


LOG = """ptxas info    : Compiling entry function 'kernel' for 'sm_89'
ptxas info    : Function properties for kernel
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 19 registers, used 0 barriers, 512 bytes cmem[0]
"""


def test_omitted_shared_memory_means_zero_but_required_fields_exist():
    assert parse_resource_log(LOG, ["kernel"])["kernel"] == {
        "registers_per_thread": 19, "static_shared_bytes": 0,
        "stack_frame_bytes": 0, "spill_store_bytes": 0, "spill_load_bytes": 0,
    }


@pytest.mark.parametrize("corrupt", [
    "", LOG + LOG, LOG.replace("sm_89", "sm_90"),
    LOG.replace("Function properties for kernel", "Function properties for wrong"),
    LOG.replace("0 bytes spill loads", ""),
    LOG.replace("Used 19 registers", ""),
    LOG + "Used 21 registers\n",
    LOG.replace("512 bytes cmem[0]", "0 bytes smem, 99999 bytes smem, 512 bytes cmem[0]"),
])
def test_corrupt_resource_logs_reject(corrupt):
    with pytest.raises(ValueError):
        parse_resource_log(corrupt, ["kernel"])


@pytest.mark.parametrize("expected", [[], ["missing"], ["kernel", "kernel"], ["kernel", "extra"]])
def test_entry_coverage_must_be_exact(expected):
    with pytest.raises(ValueError):
        parse_resource_log(LOG, expected)
