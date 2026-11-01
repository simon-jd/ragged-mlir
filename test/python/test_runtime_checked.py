"""Checked ABI and host orchestration tests, never numeric GPU simulation."""
import gc
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import dataclass
from math import prod
from pathlib import Path
from types import SimpleNamespace

import pytest

from ragged import Artifact, CacheKey, CompiledProgram, Hints, ProgramSpec
from ragged.errors import CacheMismatch, RuntimeUnavailable, ValidationError
from ragged.layout import I64_MAX, resolve_layout
from ragged.validation import CONDITIONS, raise_for_status, validate_validation_metadata
from native_fake import PlanModule


def validation_metadata():
    return {"abi": "offsets-check-v1", "min_len": 0, "max_len": 32,
            "conditions": list(CONDITIONS), "sentinel": I64_MAX,
            "workspace": [{"name": "validation_status", "dtype": "i64", "shape": [7]}],
            "kernels": [{"name": "offsets_" + role, "role": role,
                         "grid": [1 if role == "init" else "ceil_B1_128", 1, 1],
                         "block": [128, 1, 1], "shared_bytes": 0, "static_shared_bytes": 0,
                         "args": ["values", "offsets", "validation_status"]}
                        for role in ("init", "check")]}


def checked_program(validation=True):
    spec = ProgramSpec("softmax", feature_dim=4, strategy="naive", hints=Hints(32, 0, True))
    key = CacheKey.from_spec(spec, llvm_commit="host-test", ptxas_version="host-test", compiler_sha256="4"*64)
    args = ["values", "offsets", "out", "max_stats", "sum_stats"]
    metadata = {"schema": 1, "abi": "mlir-memref64-v1", "classes": ["cta"], "compiler_sha256": key.compiler_sha256,
                "toolchain": {name: getattr(key, name) for name in ("llvm_commit", "ptxas_version", "dialect_version", "chip", "features")},
                "kernels": [{"name": "main_" + name, "grid": ["B", 1, 1], "block": [128, 1, 1], "args": args}
                            for name in ("max", "sum", "normalize")],
                "workspace": [{"name": name, "shape": ["B", 4], "dtype": "f32"} for name in args[3:]]}
    if validation:
        metadata["validation"] = validation_metadata()
    return CompiledProgram(spec, Artifact(Path("host-checked-contract"), key, b"\x7fELFhost-contract", metadata))


@pytest.fixture
def host_cuda(monkeypatch):
    import ragged.runtime as runtime
    @dataclass(frozen=True)
    class Device:
        index: int = 0
    device, log, allocations, calls = Device(), [], [], []
    state = {"status": [I64_MAX]*7, "synchronized": False}
    class Stream:
        def __init__(self, handle, priority=0):
            self.device, self.cuda_stream, self.priority = device, handle, priority
        def synchronize(self):
            assert self is consumer
            log.append(("synchronize", self.cuda_stream))
            state["synchronized"] = True
    producer, consumer = Stream(101), Stream(202)
    current = [producer]
    @contextmanager
    def stream_context(stream):
        previous, current[0] = current[0], stream
        try:
            yield
        finally:
            current[0] = previous
    sizes = {"i64": 8, "f32": 4, "i32": 4, "bf16": 2, "u8": 1}
    class Tensor:
        requires_grad = False
        def __init__(self, shape, dtype, pointer, tensor_device=device):
            self.shape, self.dtype, self.pointer = shape, dtype, pointer
            self.device, self.ndim = tensor_device, len(shape)
            self.is_cuda = tensor_device != "cpu"
        def is_contiguous(self): return True
        def numel(self): return prod(self.shape)
        def element_size(self): return sizes[self.dtype]
        def data_ptr(self): return self.pointer
        def stride(self): return (self.shape[1], 1) if self.ndim == 2 else (1,)
        def record_stream(self, stream):
            assert stream is consumer
        def __getitem__(self, part):
            return Tensor((part.stop - part.start,), self.dtype, self.pointer + part.start * sizes[self.dtype], self.device)
        def view(self, form):
            if isinstance(form, str):
                return Tensor((self.numel() * sizes[self.dtype] // sizes[form],), form, self.pointer, self.device)
            return Tensor(tuple(form), self.dtype, self.pointer, self.device)
        def copy_(self, source, *, non_blocking):
            assert current[0] is consumer and self.device == "cpu" and source.is_cuda
            assert self.shape == source.shape == (7,) and self.dtype == source.dtype == "i64"
            assert non_blocking is True
            log.append(("copy", 56))
            return self
        def tolist(self):
            assert self.device == "cpu" and self.shape == (7,) and state["synchronized"]
            log.append(("read_host_status",))
            return state["status"].copy()
    def empty(shape, *, dtype, device, pin_memory=False):
        assert current[0] is consumer
        assert pin_memory is (device == "cpu")
        tensor = Tensor(shape, dtype, 0x100000 + len(allocations)*0x100000, device)
        allocations.append(tensor)
        log.append(("allocate", device, shape, dtype))
        return tensor
    class Event:
        def record(self, stream): log.append(("record", stream.cuda_stream))
        def query(self): return True
    fake_torch = SimpleNamespace(Tensor=Tensor, float32="f32", bfloat16="bf16", int32="i32", int64="i64", uint8="u8", empty=empty,
        cuda=SimpleNamespace(Stream=Stream, Event=Event, device=lambda _: nullcontext(), get_device_capability=lambda _: (8,9), is_current_stream_capturing=lambda: False,
                             current_stream=lambda _: current[0], stream=stream_context))
    def launch(*args):
        assert args[4] in [consumer.cuda_stream] + [PlanModule.side_stream(consumer.cuda_stream, lane) for lane in (1, 2)]
        calls.append(args)
        log.append(("launch", args[0]))
    program = checked_program()
    monkeypatch.setattr(runtime, "_torch", lambda: fake_torch)
    module = PlanModule(launch)
    # Programs of earlier tests that are still in reference cycles release
    # their module key when collected; collect them before installing ours.
    gc.collect()
    monkeypatch.setattr(runtime, "_resident_modules", {(0, program.artifact.key.digest): module})
    monkeypatch.setattr(runtime, "_workers", {})
    monkeypatch.setattr(runtime, "_plans", {})
    def run(*, b=5, n=10, mode="checked", program=program):
        return program.run(Tensor((n,4), "f32", 0x1000), Tensor((b+1,), "i64", 0x2000),
                           Tensor((n,4), "f32", 0x3000), stream=consumer, mode=mode)
    return SimpleNamespace(run=run, program=program, calls=calls, allocations=allocations, log=log,
                           state=state, current=current, producer=producer, consumer=consumer,
                           Tensor=Tensor, device=device)


def test_checked_copy_is_fixed_size_and_only_actual_stream_synchronizes(host_cuda):
    env = host_cuda
    report = env.run(b=128)
    assert env.current[0] is env.producer
    assert [call[0] for call in env.calls] == ["offsets_init", "offsets_check", "main_max", "main_sum", "main_normalize"]
    assert all(len(call[5]) == 17 for call in env.calls[:2])
    assert all(len(call[5]) == 33 for call in env.calls[2:])
    assert report.grids[:2] == ((1,1,1), (2,1,1))
    assert report.mode == "checked" and report.validation_workspace_bytes == 56
    assert report.workspace_bytes == 56 + 2*128*4*4
    assert report.validation_host_ms is not None and report.validation_host_ms >= 0
    assert report.device_elapsed_ms is None
    assert [event for event in env.log if event[0] == "copy"] == [("copy", 56)]
    assert [event for event in env.log if event[0] == "synchronize"] == [("synchronize", 202)]
    read = env.log.index(("read_host_status",))
    assert all(index > read for index, event in enumerate(env.log) if event[0] == "allocate" and event[-1] == "f32")


def test_checked_errors_report_all_conditions_and_never_allocate_or_launch_main(host_cuda):
    env = host_cuda
    env.state["status"] = [0,5,2,3,1,2,4]
    with pytest.raises(ValidationError) as caught:
        env.run()
    assert caught.value.errors == dict(zip(CONDITIONS, [0,5,2,3,1,2,4]))
    assert caught.value.condition == "first_nonzero" and caught.value.index == 0
    assert all(f"{condition} at index {index}" in str(caught.value) for condition,index in caught.value.errors.items())
    assert [call[0] for call in env.calls] == ["offsets_init", "offsets_check"]
    assert len(env.allocations) == 2 and all(tensor.shape == (7,) for tensor in env.allocations)
    assert env.current[0] is env.producer


def test_validated_path_does_not_allocate_launch_copy_or_wait_for_validation(host_cuda):
    env = host_cuda
    # Bogus device-status fixture must remain unread in validated mode.
    env.state["status"] = [0]*7
    report = env.run(mode="validated")
    assert [call[0] for call in env.calls] == ["main_max", "main_sum", "main_normalize"]
    assert not any(event[0] in ("copy", "synchronize", "read_host_status") for event in env.log)
    # Both statistics buffers share one scratch allocation, the second at 256 bytes.
    assert [(tensor.shape, tensor.dtype) for tensor in env.allocations] == [((256 + 5*4*4,), "u8")]
    assert report.validation_workspace_bytes == 0 and report.validation_host_ms is None


def test_b0_checked_still_runs_validation(host_cuda):
    env = host_cuda
    report = env.run(b=0,n=0)
    assert report.classes == () and report.kernel_names == ("offsets_init", "offsets_check")
    assert report.grids == ((1,1,1),(1,1,1)) and report.workspace_bytes == 56
    assert len(env.allocations) == 2


def test_b0_nonempty_values_checked_reports_device_final_mismatch(host_cuda):
    env = host_cuda
    env.state["status"][1] = 0
    with pytest.raises(ValidationError) as caught:
        env.run(b=0,n=1)
    assert caught.value.errors == {"final_mismatch": 0}
    assert len(env.calls) == 2


def test_legacy_artifact_is_validated_compatible_but_cannot_be_checked():
    program = checked_program(validation=False)
    with pytest.raises(RuntimeUnavailable, match="offsets-check-v1"):
        program.run(None, None, None, mode="checked")


@pytest.mark.parametrize("field,value", [
    ("abi", "offsets-check-v2"), ("sentinel", 2**63), ("sentinel", float(I64_MAX)),
    ("min_len", True), ("min_len", -1), ("max_len", 2**63), ("max_len", "32"),
    ("conditions", list(reversed(CONDITIONS))), ("workspace", [{"name":"validation_status","dtype":"i32","shape":[7]}]),
    ("workspace", [{"name":"validation_status","dtype":"i64","shape":[7.0]}]),
])
def test_validation_manifest_rejects_unsafe_status_contract(field,value):
    metadata = validation_metadata()
    metadata[field] = value
    with pytest.raises(CacheMismatch):
        validate_validation_metadata(metadata)


@pytest.mark.parametrize("mutation", ["order", "args", "grid", "float_grid", "duplicate", "shared", "missing_bound", "extra_field"])
def test_validation_manifest_rejects_unsafe_schedule(mutation):
    metadata = validation_metadata()
    if mutation == "order": metadata["kernels"].reverse()
    elif mutation == "args": metadata["kernels"][1]["args"] = ["values","offsets","out"]
    elif mutation == "grid": metadata["kernels"][1]["grid"][0] = "ceil_B_128"
    elif mutation == "float_grid": metadata["kernels"][0]["grid"][0] = 1.0
    elif mutation == "duplicate": metadata["kernels"][1]["name"] = metadata["kernels"][0]["name"]
    elif mutation == "shared": metadata["kernels"][1]["shared_bytes"] = False
    elif mutation == "missing_bound": del metadata["min_len"]
    else: metadata["executable_expression"] = "B+1"
    with pytest.raises(CacheMismatch):
        validate_validation_metadata(metadata)


def test_validation_bounds_are_bound_to_source_even_when_offsets_checked_true():
    program = checked_program()
    metadata = deepcopy(program.artifact.metadata)
    metadata["validation"]["max_len"] = 4096
    with pytest.raises(CacheMismatch, match="source hints"):
        CompiledProgram(program.spec, Artifact(program.artifact.path, program.artifact.key, program.artifact.cubin, metadata))


@pytest.mark.parametrize("b,grid", [(0,1),(126,1),(127,1),(128,2),(255,2),(256,3)])
def test_checked_ceiling_includes_terminal_offset(b,grid):
    _,grids,nbytes = resolve_layout(validation_metadata(), n=0,b=b,d=4)
    assert grids == ((1,1,1),(grid,1,1)) and nbytes == 56


@pytest.mark.parametrize("b", [I64_MAX, 128*(2**31-1)])
def test_validation_grid_and_b_plus_one_cannot_overflow(b):
    with pytest.raises(ValidationError):
        resolve_layout(validation_metadata(),n=0,b=b,d=4)


@pytest.mark.parametrize("status", [[I64_MAX]*6, [-1]+[I64_MAX]*6, [I64_MAX,4]+[I64_MAX]*5,
                                    [I64_MAX]*4+[5,I64_MAX,I64_MAX], [True]+[I64_MAX]*6])
def test_corrupt_status_cannot_be_treated_as_success(status):
    with pytest.raises(ValidationError):
        raise_for_status(status,b=5)
