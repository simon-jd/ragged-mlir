"""Host orchestration contract only: no numerical GPU execution is simulated."""
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from math import prod
from pathlib import Path
from types import SimpleNamespace

from ragged import Artifact, CacheKey, CompiledProgram, ProgramSpec
from native_fake import PlanModule


def test_internal_workspace_allocates_on_explicit_consumer_stream(monkeypatch):
    import ragged.runtime as runtime
    @dataclass(frozen=True)
    class Device:
        index: int = 0
    device = Device()
    class Stream:
        def __init__(self, handle):
            self.device, self.cuda_stream = device, handle
    producer, consumer = Stream(101), Stream(202)
    current = [producer]
    allocations, launches, recordings = [], [], []
    @contextmanager
    def stream_context(stream):
        previous, current[0] = current[0], stream
        try:
            yield
        finally:
            current[0] = previous
    class Tensor:
        is_cuda, requires_grad = True, False
        def __init__(self, shape, dtype, pointer):
            self.shape, self.dtype, self.pointer = shape, dtype, pointer
            self.device, self.ndim = device, len(shape)
        def is_contiguous(self): return True
        def numel(self): return prod(self.shape)
        def element_size(self): return 8 if self.dtype == "i64" else 4
        def data_ptr(self): return self.pointer
        def stride(self): return (self.shape[1], 1) if self.ndim == 2 else (1,)
        def record_stream(self, stream): recordings.append((self, stream))
    def allocate(shape, *, dtype, device):
        # The pre-fix code allocates here with producer current, which fails.
        assert current[0] is consumer, "scratch can reuse producer-stream memory with pending writes"
        tensor = Tensor(shape, dtype, 0x10000 + len(allocations) * 0x1000)
        allocations.append(tensor)
        return tensor
    fake_torch = SimpleNamespace(Tensor=Tensor, float32="f32", bfloat16="bf16", int32="i32", int64="i64", uint8="u8",
        empty=allocate, cuda=SimpleNamespace(Stream=Stream, device=lambda _: nullcontext(),
        get_device_capability=lambda _: (8, 9), is_current_stream_capturing=lambda: False,
        current_stream=lambda _: current[0], stream=stream_context))
    spec = ProgramSpec("softmax", feature_dim=4, strategy="naive")
    key = CacheKey.from_spec(spec, llvm_commit="host-test", ptxas_version="host-test", compiler_sha256="1"*64)
    args = ["values", "offsets", "out", "max_stats", "sum_stats"]
    metadata = {"schema": 1, "abi": "mlir-memref64-v1", "compiler_sha256": key.compiler_sha256,
        "toolchain": {name: getattr(key, name) for name in ("llvm_commit", "ptxas_version", "dialect_version", "chip", "features")},
        "kernels": [{"name": name, "grid": ["B", 1, 1], "block": [128, 1, 1], "args": args}
                    for name in ("maximum", "sum", "normalize")],
        "workspace": [{"name": name, "shape": ["B", 4], "dtype": "f32"} for name in args[3:]]}
    program = CompiledProgram(spec, Artifact(Path("synthetic-host-contract"), key, b"\x7fELFhost-contract", metadata))
    # Only launch argument orchestration is recorded; no result values exist.
    monkeypatch.setattr(runtime, "_torch", lambda: fake_torch)
    monkeypatch.setattr(runtime, "_resident_modules", {(0, key.digest): PlanModule(lambda *a: launches.append(a))})
    monkeypatch.setattr(runtime, "_plans", {})
    report = program.run(Tensor((6, 4), "f32", 0x1000), Tensor((4,), "i64", 0x2000),
                         Tensor((6, 4), "f32", 0x3000), stream=consumer)
    assert current[0] is producer
    # One scratch allocation holds both statistics buffers.
    assert len(allocations) == 1 and report.workspace_bytes == 2 * 3 * 4 * 4
    assert len(launches) == 3 and all(call[4] == consumer.cuda_stream and len(call[5]) == 33 for call in launches)
    assert len(recordings) == 3 and all(stream is consumer for _, stream in recordings)
