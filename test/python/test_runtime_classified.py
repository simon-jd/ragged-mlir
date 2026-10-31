"""Host launch orchestration, without simulating GPU kernel computations."""
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from math import prod
from pathlib import Path
from types import SimpleNamespace

import pytest

from ragged import Artifact, CacheKey, CompiledProgram, ProgramSpec
from ragged.errors import ValidationError
from native_fake import PlanModule


def host_program(*, debug=True):
    spec = ProgramSpec("softmax", feature_dim=33, debug_visits=debug)
    key = CacheKey.from_spec(spec, llvm_commit="host-test", ptxas_version="host-test", compiler_sha256="3"*64)
    workspace = [{"name": "class_counts", "shape": [3], "dtype": "i64"},
                 {"name": "segment_ids", "shape": [3, "B"], "dtype": "i64"},
                 {"name": "partials", "shape": ["B_split", 264], "dtype": "f32"},
                 *[{"name": name, "shape": ["B", 33], "dtype": "f32"} for name in ("max_stats", "sum_stats")]]
    if debug:
        workspace.append({"name": "visits", "shape": ["B"], "dtype": "i32"})
    args = ["values", "offsets", "out", *[entry["name"] for entry in workspace]]
    kernels = [{"name": "init", "role": "init", "grid": [1, 1, 1], "block": [128, 1, 1], "args": args},
               {"name": "classify", "role": "classify", "grid": ["ceil_B_128", 1, 1], "block": [128, 1, 1], "args": args}]
    phases = {"warp": ("max_stats", "sum_stats", "normalize"),
              "cta": ("max_stats", "sum_stats", "normalize"),
              "split": ("partial_max", "combine_max_stats", "partial_exp_sum", "combine_sum_stats", "normalize")}
    for segment_class, class_phases in phases.items():
        for phase in class_phases:
            role = f"{segment_class}_{phase}"
            z = 8 if segment_class == "split" and (phase.startswith("partial_") or phase == "normalize") else 1
            kernels.append({"name": role, "role": role, "segment_class": segment_class,
                            "grid": ["min_B_resident", 2, z], "block": [32 if segment_class == "warp" else 128, 1, 1], "args": args})
    metadata = {"schema": 1, "abi": "mlir-memref64-v1", "dispatch_abi": "classified-v3",
                "backend": "classified-persistent", "classes": list(phases), "thresholds": [32, 4096],
                "fusion": "three_pass", "alias_safe": True, "split_parts": 8, "debug_visits": debug,
                "workspace": workspace, "kernels": kernels, "compiler_sha256": key.compiler_sha256,
                "toolchain": {name: getattr(key, name) for name in ("llvm_commit", "ptxas_version", "dialect_version", "chip", "features")}}
    return CompiledProgram(spec, Artifact(Path("host-contract"), key, b"\x7fELFhost-contract", metadata))


@pytest.fixture
def host_cuda(monkeypatch):
    import ragged.runtime as runtime
    @dataclass(frozen=True)
    class Device:
        index: int = 0
    device = Device()
    current, allocations, launches, recordings, events = [], [], [], [], []
    class Stream:
        def __init__(self, handle, priority=0):
            self.device, self.cuda_stream, self.priority = device, handle, priority
    class Event:
        def record(self, stream): events.append(("record", stream.cuda_stream, self))
        def query(self): return True
    producer, consumer = Stream(101), Stream(202)
    current.append(producer)
    @contextmanager
    def stream_context(stream):
        previous, current[0] = current[0], stream
        try:
            yield
        finally:
            current[0] = previous
    sizes = {"i64": 8, "f32": 4, "i32": 4, "u8": 1}
    class Tensor:
        # No item(), cpu(), numpy(), or synchronize() methods: tests fail if
        # runtime orchestration attempts any device data read or host wait.
        is_cuda, requires_grad = True, False
        def __init__(self, shape, dtype, pointer):
            self.shape, self.dtype, self.pointer = shape, dtype, pointer
            self.device, self.ndim = device, len(shape)
        def is_contiguous(self): return True
        def numel(self): return prod(self.shape)
        def element_size(self): return sizes[self.dtype]
        def data_ptr(self): return self.pointer
        def stride(self): return (self.shape[1], 1) if self.ndim == 2 else (1,)
        def record_stream(self, stream): recordings.append((self, stream))
        def __getitem__(self, part):
            return Tensor((part.stop - part.start,), self.dtype, self.pointer + part.start * sizes[self.dtype])
        def view(self, form):
            if isinstance(form, str):
                return Tensor((self.numel() * sizes[self.dtype] // sizes[form],), form, self.pointer)
            assert prod(form) == self.numel()
            return Tensor(tuple(form), self.dtype, self.pointer)
    def allocate(shape, *, dtype, device):
        assert current[0] is consumer
        tensor = Tensor(shape, dtype, 0x1000000 + len(allocations) * 0x10000000)
        allocations.append(tensor)
        return tensor
    fake_torch = SimpleNamespace(Tensor=Tensor, float32="f32", bfloat16="bf16", int32="i32", int64="i64", uint8="u8",
        empty=allocate, cuda=SimpleNamespace(Stream=Stream, Event=Event, device=lambda _: nullcontext(),
        get_device_capability=lambda _: (8, 9), is_current_stream_capturing=lambda: False, current_stream=lambda _: current[0], stream=stream_context))
    program = host_program()
    monkeypatch.setattr(runtime, "_torch", lambda: fake_torch)
    module = PlanModule(lambda *a: launches.append(a), events)
    monkeypatch.setattr(runtime, "_resident_modules", {(0, program.artifact.key.digest): module})
    monkeypatch.setattr(runtime, "_workers", {})
    monkeypatch.setattr(runtime, "_plans", {})
    return SimpleNamespace(program=program, Tensor=Tensor, producer=producer, consumer=consumer, current=current,
                           allocations=allocations, launches=launches, recordings=recordings, torch=fake_torch,
                           events=events, module=module)


def test_classified_scratch_full_descriptors_and_debug_handles(host_cuda):
    env = host_cuda
    # At most three of the 257 segments can exceed 4,096 of the 12,291 rows.
    n, b, d = 4097 * 3, 257, 33
    buffers = {}
    report = env.program.run(env.Tensor((n, d), "f32", 0x1000000), env.Tensor((b+1,), "i64", 0x2000000),
                             env.Tensor((n, d), "f32", 0x3000000), stream=env.consumer, debug_buffers=buffers)
    assert env.current[0] is env.producer
    assert list(buffers) == ["class_counts", "segment_ids", "partials", "max_stats", "sum_stats", "visits"]
    assert [(tensor.shape, tensor.dtype) for tensor in buffers.values()] == [
        ((3,), "i64"), ((3, b), "i64"), ((3, 8*d), "f32"), ((b, d), "f32"), ((b, d), "f32"), ((b,), "i32")]
    # One scratch allocation on the consumer stream, buffers at 256-byte offsets.
    [scratch] = env.allocations
    assert scratch.dtype == "u8"
    pointers = [tensor.pointer - scratch.pointer for tensor in buffers.values()]
    assert pointers[0] == 0 and all(offset % 256 == 0 for offset in pointers) and pointers == sorted(pointers)
    sizes = [tensor.numel() * tensor.element_size() for tensor in buffers.values()]
    assert all(after - before < size + 256 for before, after, size in zip(pointers, pointers[1:], sizes))
    assert scratch.shape == (pointers[-1] + sizes[-1],)
    assert report.workspace_bytes == 24 + b * (24 + 2*d*4 + 4) + 3 * 8*d*4 == sum(sizes)
    assert len(env.launches) == 13
    assert report.grids[:2] == ((1, 1, 1), (3, 1, 1))
    # Worker grid x is min(B, resident blocks // feature tiles // parts).
    threads = {name: 32 if name.startswith("warp") else 128 for name in report.kernel_names}
    for name, grid in zip(report.kernel_names[2:], report.grids[2:]):
        assert grid[:2] == (min(b, 128 * (24 if threads[name] == 32 else 12) // (2 * grid[2])), 2)
    assert tuple(call[0] for call in env.launches) == report.kernel_names
    for call in env.launches:
        assert len(call[5]) == 57
        # Rank-one i64 counts descriptor, then rank-two i64 queues.
        assert call[5][19:24] == [buffers["class_counts"].pointer] * 2 + [0, 3, 1]
        assert call[5][24:31] == [buffers["segment_ids"].pointer] * 2 + [0, 3, b, b, 1]
        assert call[5][-5:] == [buffers["visits"].pointer] * 2 + [0, b, 1]
    # Only caller-owned tensors need the consumer stream recorded.
    assert len(env.recordings) == 3 and all(stream is env.consumer for _, stream in env.recordings)


def test_class_chains_fork_after_classify_and_join_on_the_caller_stream(host_cuda):
    env = host_cuda
    n, b, d = 400, 257, 33
    report = env.program.run(env.Tensor((n, d), "f32", 0x100000), env.Tensor((b+1,), "i64", 0x200000),
                             env.Tensor((n, d), "f32", 0x300000), stream=env.consumer, debug_buffers={})
    assert report.streams == 3
    caller = env.consumer.cuda_stream
    cta, split = (env.module.side_stream(caller, lane) for lane in (1, 2))
    def launches(stream):
        return [event[2] for event in env.events if event[0] == "launch" and event[1] == stream]
    # CTA first, then split on their own lanes; warp stays on the caller's stream.
    assert launches(caller) == ["init", "classify", "warp_max_stats", "warp_sum_stats", "warp_normalize"]
    assert launches(split) == ["split_partial_max", "split_combine_max_stats", "split_partial_exp_sum",
                               "split_combine_sum_stats", "split_normalize"]
    assert launches(cta) == ["cta_max_stats", "cta_sum_stats", "cta_normalize"]
    assert report.kernel_names == ("init", "classify", *launches(cta), *launches(split), *launches(caller)[2:])
    assert env.events.index(("record", caller, "fork")) == env.events.index(("launch", caller, "classify")) + 1
    # One native call per run, and the plan is built once per problem size.
    env.program.run(env.Tensor((n, d), "f32", 0x100000), env.Tensor((b+1,), "i64", 0x200000),
                    env.Tensor((n, d), "f32", 0x300000), stream=env.consumer, debug_buffers={})
    assert len(env.module.plans) == 1 and len(env.launches) == 26


def test_serial_classes_launch_in_manifest_order_on_the_caller_stream(host_cuda):
    env = host_cuda
    report = env.program.run(env.Tensor((400, 33), "f32", 0x100000), env.Tensor((258,), "i64", 0x200000),
                             env.Tensor((400, 33), "f32", 0x300000), stream=env.consumer, debug_buffers={},
                             concurrent=False)
    assert report.streams == 1
    assert not [event for event in env.events if event[0] != "launch"]
    assert all(call[4] == env.consumer.cuda_stream for call in env.launches)
    assert [call[0] for call in env.launches] == [kernel["name"] for kernel in env.program.artifact.metadata["kernels"]]


def test_b0_performs_no_allocation_driver_work_or_stream_query(host_cuda):
    env = host_cuda
    def forbidden(*_):
        raise AssertionError("B=0 must not invoke CUDA")
    env.torch.cuda.device = forbidden
    buffers = {}
    report = env.program.run(env.Tensor((0, 33), "f32", 0x1000), env.Tensor((1,), "i64", 0x2000),
                             env.Tensor((0, 33), "f32", 0x3000), stream=env.consumer, debug_buffers=buffers)
    assert not env.allocations and not env.launches and not env.recordings and not buffers
    assert report.classes == report.grids == report.kernel_names == () and report.workspace_bytes == 0


def test_b0_with_values_rejected_without_side_effects(host_cuda):
    env = host_cuda
    with pytest.raises(ValidationError, match="B=0 requires N=0"):
        env.program.run(env.Tensor((1, 33), "f32", 0x1000), env.Tensor((1,), "i64", 0x2000),
                        env.Tensor((1, 33), "f32", 0x3000), stream=env.consumer)
    assert not env.allocations and not env.launches


@pytest.mark.parametrize("debug,buffers", [(False, {}), (True, []), (True, {"visits": object()})])
def test_debug_buffer_contract_rejected_before_tensor_access(debug, buffers):
    with pytest.raises(ValidationError, match="debug_buffers"):
        host_program(debug=debug).run(None, None, None, debug_buffers=buffers)


def test_debug_visits_is_a_typed_ir_specialization():
    plain, debug = ProgramSpec("sum"), ProgramSpec("sum", debug_visits=True)
    assert plain.ir_hash != debug.ir_hash
    assert "debug_visits = true" in debug.to_mlir()
    assert ProgramSpec.from_dict(debug.to_dict()) == debug
    with pytest.raises(ValidationError, match="debug_visits"):
        ProgramSpec("sum", debug_visits=1)
