"""Data fairness and measurement evidence checks, without simulated CUDA."""
from types import SimpleNamespace

import numpy as np
import pytest

from bench.run import prepare_case, summarize
from bench.public_histogram import sample_histogram
from scripts.run_gpu_validation import junit_counts


def test_fixed_elements_case_records_actual_source_precision_and_input_identity():
    args = SimpleNamespace(distribution="long_tail", segments=31, seed=711,
                           total_elements=4097, mean_length=64, offsets_dtype="i32",
                           dtype="bf16", d=7, op="sum")
    values, offsets, evidence = prepare_case(args)
    repeat_values, repeat_offsets, repeat_evidence = prepare_case(args)
    assert values.shape == (4097, 7)
    assert offsets.dtype == np.int32
    assert offsets[-1] == 4097
    assert np.all(values.view(np.uint32) & 0xFFFF == 0)
    assert np.array_equal(values, repeat_values)
    assert np.array_equal(offsets, repeat_offsets)
    assert evidence == repeat_evidence
    assert sum(evidence["histogram_counts"]) == 31


@pytest.mark.parametrize("samples", [[], [-1.0], [float("nan")], [float("inf")]])
def test_invalid_measurements_are_never_reported_as_timings(samples):
    with pytest.raises(ValueError):
        summarize(samples)


def test_measurement_summary_preserves_slow_samples():
    summary = summarize([1.0, 2.0, 100.0])
    assert summary["median"] == 2
    assert summary["max"] == 100
    assert summary["samples"] == [1.0, 2.0, 100.0]


def test_skipped_gpu_cases_are_distinct_from_sanitizer_success(tmp_path):
    path = tmp_path / "results.xml"
    path.write_text('<testsuites><testsuite><testcase/><testcase><skipped/></testcase>'
                    '<testcase><failure/></testcase><testcase><error/></testcase></testsuite></testsuites>')
    assert junit_counts(path) == {"tests": 4, "skipped": 1, "failures": 1, "errors": 1}


def test_empirical_sampling_preserves_exact_total_beyond_float_integer_range(tmp_path):
    path = tmp_path / "histogram.json"
    path.write_text('{"length_counts": [[1, 15], [16, 3], [1000, 1]]}')
    n = 2**62 + 37
    lengths = sample_histogram(path, 53, seed=92, total_elements=n)
    assert sum(map(int, lengths)) == n
    assert np.array_equal(lengths, sample_histogram(path, 53, seed=92, total_elements=n))
    native = sample_histogram(path, 53, seed=92, total_elements=None)
    assert set(native) <= {1, 16, 1000}


@pytest.mark.parametrize("pairs", ['[]', '[[0, 1]]', '[[1, -1]]', '[[1, true]]', '[[1.5, 2]]'])
def test_invalid_public_histogram_is_rejected(tmp_path, pairs):
    path = tmp_path / "histogram.json"
    path.write_text('{"length_counts": ' + pairs + '}')
    with pytest.raises(ValueError):
        sample_histogram(path, 2, seed=0, total_elements=10)


@pytest.mark.parametrize('dtype', ['f32', 'bf16'])
def test_complete_torch_baseline_keeps_preprocessing_f32_and_original_output_dtype(dtype):
    torch = pytest.importorskip('torch')
    from bench.run import make_call
    from oracle import assert_matches
    values = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32 if dtype == 'f32' else torch.bfloat16)
    offsets = torch.tensor([0, 0, 3], dtype=torch.int64)
    args = SimpleNamespace(scale=0.1234567, op='sum', d=1)
    call, _ = make_call('torch-segment', args, torch, values, offsets, None)
    actual = call()
    assert actual.dtype == values.dtype
    expected_input = values.float().numpy() * np.float32(args.scale)
    assert_matches(actual.float().numpy(), expected_input, offsets.numpy(), operation='sum', input_dtype=dtype)


@pytest.mark.parametrize('dtype', ['f32', 'bf16'])
def test_complete_njt_baseline_includes_scale_mask_and_shared_length_hints(dtype):
    torch = pytest.importorskip('torch')
    from bench.run import make_call
    from oracle.masked import assert_masked_softmax_matches
    values = torch.tensor([[1.0, -2.0], [2.0, -3.0], [-1.0, 4.0]],
                          dtype=torch.float32 if dtype == 'f32' else torch.bfloat16)
    offsets = torch.tensor([0, 2, 3], dtype=torch.int64)
    mask = torch.tensor([[0.125, -0.2], [0.125, -0.3], [-1.0, -float('inf')]])
    args = SimpleNamespace(scale=0.1234567, op='softmax', d=2, offsets_dtype='i64',
                           length_bounds={'min_len': 1, 'max_len': 2})
    call, _ = make_call('njt-eager', args, torch, values, offsets, None, mask)
    actual = call()
    assert actual.dtype == values.dtype
    assert_masked_softmax_matches(actual.float().numpy(), values.float().numpy(), offsets.numpy(), mask.numpy(),
                                  scale=args.scale, input_dtype=dtype)


@pytest.mark.parametrize('dtype', ['f32', 'bf16'])
def test_complete_njt_max_baseline_preserves_empty_identity_and_f32_preprocessing(dtype):
    torch = pytest.importorskip('torch')
    from bench.run import make_call
    from oracle import assert_matches
    values = torch.tensor([[-float('inf'), -3.0], [-float('inf'), -2.0], [1.0, float('nan')]],
                          dtype=torch.float32 if dtype == 'f32' else torch.bfloat16)
    offsets = torch.tensor([0, 0, 1, 3], dtype=torch.int64)
    args = SimpleNamespace(scale=0.25, op='max', d=2, offsets_dtype='i64',
                           length_bounds={'min_len': 0, 'max_len': 2})
    call, evidence = make_call('njt-eager', args, torch, values, offsets, None)
    actual = call()
    assert actual.dtype == values.dtype
    assert 'padding' in evidence['baseline_strategy']
    expected_input = values.float().numpy() * np.float32(args.scale)
    assert_matches(actual.float().numpy(), expected_input, offsets.numpy(), operation='max', input_dtype=dtype)
