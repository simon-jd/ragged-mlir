"""Scheduling changes that must not change results: row pipelining, device-sized
persistent grids and concurrent segment-class chains.

Row pipelining folds each lane's rows in the same order as the rolled loop, and
the class chains touch disjoint segments, so every comparison here is bitwise,
not within the oracle tolerance.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from bench.distributions import offsets_from_lengths
from oracle import assert_matches
from ragged import Optimizations, ProgramSpec
import ragged.runtime as runtime
from test_correctness import gpu_environment, prepared

# Every class, lengths around the 8-row unroll and the 32/4096 class bounds,
# and a split segment whose eight parts have different remainders.
LENGTHS = np.array([0, 1, 7, 8, 9, 31, 32, 33, 63, 64, 65, 257, 1000, 4095, 4096, 4097, 9001, 30011], dtype=np.int64)


def device_case(torch, spec, lengths, seed):
    offsets = offsets_from_lengths(lengths, spec.offsets_dtype)
    values = prepared(np.random.default_rng(seed).standard_normal((int(offsets[-1]), spec.feature_dim)) * 4, spec.dtype)
    dtype = torch.float32 if spec.dtype == "f32" else torch.bfloat16
    return values, offsets, torch.from_numpy(values).to("cuda", dtype), torch.from_numpy(offsets).cuda()


def run(torch, program, device_values, device_offsets, **options):
    spec = program.spec
    if spec.alias == "inplace":
        output = device_values.clone()
        source = output
    else:
        rows = len(device_values) if spec.op == "softmax" else len(device_offsets) - 1
        output = torch.empty((rows, spec.feature_dim), device="cuda", dtype=device_values.dtype)
        source = device_values
    report = program.run(source, device_offsets, output, **options)
    torch.cuda.current_stream().synchronize()
    return output, report


def bits(torch, tensor):
    return tensor.view(torch.int32 if tensor.dtype == torch.float32 else torch.int16)


def worker_counts(torch, program):
    """Device-sized worker counts of a loaded program, from a one-segment run."""
    spec = program.spec
    values = torch.zeros((1, spec.feature_dim), device="cuda", dtype=torch.float32 if spec.dtype == "f32" else torch.bfloat16)
    offsets = torch.tensor([0, 1], device="cuda", dtype=torch.int64 if spec.offsets_dtype == "i64" else torch.int32)
    run(torch, program, values, offsets)
    return runtime._workers[(0, program.artifact.key.digest)]


@pytest.mark.parametrize("op,alias", [("sum", "none"), ("max", "none"), ("softmax", "none"), ("softmax", "inplace")])
@pytest.mark.parametrize("dtype", ["f32", "bf16"])
@pytest.mark.parametrize("d", [1, 7, 128])
def test_row_pipelining_is_bitwise_identical_to_the_rolled_loop(gpu_environment, op, alias, dtype, d):
    torch, compile_spec, _, _ = gpu_environment
    spec = ProgramSpec(op, dtype, d, alias=alias)
    values, offsets, device_values, device_offsets = device_case(torch, spec, LENGTHS, 89 + d)
    pipelined, report = run(torch, compile_spec(spec), device_values, device_offsets)
    rolled, _ = run(torch, compile_spec(replace(spec, optimizations=Optimizations(row_pipelining=False))),
                    device_values, device_offsets)
    assert report.optimizations["effective"]["row_pipelining"] is True
    assert torch.equal(bits(torch, pipelined), bits(torch, rolled))
    assert_matches(pipelined.float().cpu().numpy(), values, offsets, operation=op, input_dtype=dtype)


@pytest.mark.parametrize("op", ["sum", "max", "softmax"])
@pytest.mark.parametrize("d", [1, 128])
def test_concurrent_class_chains_are_bitwise_identical_to_serial(gpu_environment, op, d):
    torch, compile_spec, _, _ = gpu_environment
    spec = ProgramSpec(op, "f32", d)
    program = compile_spec(spec)
    _, _, device_values, device_offsets = device_case(torch, spec, np.resize(LENGTHS, 200), 7)
    concurrent, report = run(torch, program, device_values, device_offsets)
    serial, serial_report = run(torch, program, device_values, device_offsets, concurrent=False)
    assert report.classes == ("warp", "cta", "split") and report.streams == 3 and serial_report.streams == 1
    assert torch.equal(bits(torch, concurrent), bits(torch, serial))


@pytest.mark.parametrize("op", ["sum", "softmax"])
def test_concurrent_chains_observe_prior_work_and_precede_later_work_on_the_caller_stream(gpu_environment, op):
    # The values are produced on the caller's stream behind a long matmul and
    # consumed there right after the call; the class streams must wait for the
    # producer (fork) and the consumer for every class stream (join).
    torch, compile_spec, _, _ = gpu_environment
    spec = ProgramSpec(op, "f32", 128)
    program = compile_spec(spec)
    values, offsets, _, device_offsets = device_case(torch, spec, np.resize(LENGTHS, 120), 11)
    stream = torch.cuda.Stream()
    host = torch.from_numpy(values).pin_memory()
    delay = torch.randn((4096, 4096), device="cuda")
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        device_values = torch.zeros(values.shape, device="cuda")
        for _ in range(4):
            delay = delay @ delay / 64
        device_values.copy_(host, non_blocking=True)
        rows = len(values) if op == "softmax" else len(offsets) - 1
        output = torch.full((rows, 128), float("nan"), device="cuda")
        report = program.run(device_values, device_offsets, output, stream=stream)
        copied = output.clone()
    stream.synchronize()
    assert report.streams == 3
    assert_matches(copied.cpu().numpy(), values, offsets, operation=op, input_dtype="f32")


@pytest.mark.parametrize("op", ["sum", "softmax"])
def test_cuda_graph_replays_concurrent_class_chains(gpu_environment, op):
    torch, compile_spec, _, _ = gpu_environment
    spec = ProgramSpec(op, "f32", 33)
    program = compile_spec(spec)
    values, offsets, device_values, device_offsets = device_case(torch, spec, np.resize(LENGTHS, 64), 5)
    rows = len(values) if op == "softmax" else len(offsets) - 1
    output = torch.empty((rows, 33), device="cuda")
    capture = torch.cuda.Stream()
    with torch.cuda.stream(capture):
        program.run(device_values, device_offsets, output, stream=capture)   # load the module, create side streams
    capture.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture):
        report = program.run(device_values, device_offsets, output, stream=capture)
    assert report.streams == 3
    replacement = prepared(np.random.default_rng(6).standard_normal(values.shape), "f32")
    device_values.copy_(torch.from_numpy(replacement))
    output.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert_matches(output.cpu().numpy(), replacement, offsets, operation=op, input_dtype="f32")


@pytest.mark.parametrize("d", [1, 128])
def test_worker_grids_are_one_resident_wave(gpu_environment, d):
    torch, compile_spec, _, _ = gpu_environment
    program = compile_spec(ProgramSpec("softmax", "f32", d))
    workers = worker_counts(torch, program)
    module = runtime._resident_modules[(0, program.artifact.key.digest)]
    kernels = {kernel["name"]: kernel for kernel in program.artifact.metadata["kernels"]}
    assert set(workers) == {name for name, kernel in kernels.items() if "segment_class" in kernel}
    for name, count in workers.items():
        kernel = kernels[name]
        groups = kernel["grid"][1] * kernel["grid"][2]
        assert count == max(1, module.resident_blocks(name, kernel["block"][0], 0) // groups)
    _, _, device_values, device_offsets = device_case(torch, program.spec, np.resize(LENGTHS, 6000), 3)
    _, report = run(torch, program, device_values, device_offsets)
    for name, grid in zip(report.kernel_names, report.grids):
        if name in workers:
            assert grid[0] == min(6000, workers[name])


@pytest.mark.parametrize("op", ["sum", "max", "softmax"])
def test_persistent_workers_drain_queues_longer_than_the_grid(gpu_environment, op):
    # Each class holds twice as many segments as the grid of its reduction
    # workers, so every worker takes several queue entries and reuses its
    # shared storage. (Split combine workers read partials, not shared memory.)
    torch, compile_spec, _, _ = gpu_environment
    spec = ProgramSpec(op, "f32", 3)
    program = compile_spec(spec)
    workers = worker_counts(torch, program)
    most = {segment_class: max(count for name, count in workers.items()
                               if f"_{segment_class}_" in name and "_combine_" not in name)
            for segment_class in ("warp", "cta", "split")}
    lengths = np.concatenate([np.resize([0, 1, 32], 2 * most["warp"]), np.full(2 * most["cta"], 33),
                              np.full(2 * most["split"], 4097)]).astype(np.int64)
    lengths = np.random.default_rng(31).permutation(lengths)
    values, offsets, device_values, device_offsets = device_case(torch, spec, lengths, 31)
    output, report = run(torch, program, device_values, device_offsets)
    assert report.classes == ("warp", "cta", "split")
    assert_matches(output.cpu().numpy(), values, offsets, operation=op, input_dtype="f32")


def test_native_launch_plans_reject_malformed_steps(gpu_environment):
    # Every rejection here happens before any driver call, so the test stays
    # clean under compute-sanitizer's API error reporting.
    torch, compile_spec, _, _ = gpu_environment
    program = compile_spec(ProgramSpec("sum", "f32", 3))
    worker_counts(torch, program)
    module = runtime._resident_modules[(0, program.artifact.key.digest)]
    kernel = program.artifact.metadata["kernels"][0]["name"]
    step = (kernel, (1, 1, 1), (128, 1, 1), 0, 0)
    with pytest.raises(RuntimeError, match="grid dimensions must be nonzero"):
        module.add_plan([(kernel, (0, 1, 1), (128, 1, 1), 0, 0)], [], [])
    with pytest.raises(RuntimeError, match="block"):
        module.add_plan([(kernel, (1, 1, 1), (2048, 1, 1), 0, 0)], [], [])
    with pytest.raises(RuntimeError, match="at most eight side lanes"):
        module.add_plan([(kernel, (1, 1, 1), (128, 1, 1), 0, 9)], [], [])
    with pytest.raises(RuntimeError, match="relocation is out of range"):
        module.add_plan([step], [0, 0], [2])
    with pytest.raises(RuntimeError, match="unknown launch plan"):
        module.run_plan(1 << 20, torch.cuda.current_stream().cuda_stream, [], 0)
