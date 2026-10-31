"""Frontend unit checks, separate from numerical compiler/GPU evidence."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from ragged import (ArtifactCache, CacheKey, CacheMismatch, CompilationError, Compiler,
                    ExecutionReport, Hints, ProgramSpec, ToolchainUnavailable, ValidationError)
from ragged.cli import main
from ragged.runtime import memref_arguments


@pytest.mark.parametrize("kwargs", [
    {"op": "mean"}, {"dtype": "f16"}, {"feature_dim": 0}, {"feature_dim": 1025},
    {"feature_dim": True}, {"feature_dim": 2.0}, {"offsets_dtype": "u32"},
    {"strategy": "magic"}, {"chip": "sm_90"}, {"alias": "inplace"},
])
def test_reject_invalid_program(kwargs):
    with pytest.raises(ValidationError):
        ProgramSpec(**({"op": "sum"} | kwargs))


@pytest.mark.parametrize("kwargs", [
    {"max_len": -1}, {"min_len": True}, {"max_len": 2**63},
    {"max_len": 4, "min_len": 5}, {"offsets_checked": 1},
])
def test_reject_invalid_hints(kwargs):
    with pytest.raises(ValidationError):
        Hints(**kwargs)


@pytest.mark.parametrize("op", ["sum", "max", "softmax"])
@pytest.mark.parametrize("dtype", ["f32", "bf16"])
@pytest.mark.parametrize("offset_type", ["i32", "i64"])
def test_roundtrip_and_stable_ir(op, dtype, offset_type):
    spec = ProgramSpec(op, dtype, 17, offset_type, Hints(63, 0, True))
    restored = ProgramSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
    assert restored == spec
    assert restored.to_mlir() == spec.to_mlir()
    assert f"memref<?x17x{dtype}>" in spec.to_mlir()
    assert f"memref<?x{offset_type}>" in spec.to_mlir()
    assert "max_len = 63 : i64" in spec.to_mlir()
    assert '"ragged.segment_' in spec.to_mlir()
    with pytest.raises(ValidationError):
        spec.to_mlir("main;malformed")


def test_hints_change_hash_not_signature():
    first = ProgramSpec("sum", hints=Hints(max_len=32))
    second = ProgramSpec("sum", hints=Hints(max_len=64))
    assert first.signature == second.signature
    assert first.ir_hash != second.ir_hash


def test_import_does_not_require_torch():
    source = str(Path(__file__).resolve().parents[2] / "python")
    subprocess.run([sys.executable, "-c", f"import sys; sys.path.insert(0, {source!r}); import ragged; assert 'torch' not in sys.modules"], check=True)


def test_missing_compiler_is_not_cpu_fallback():
    with pytest.raises(ToolchainUnavailable, match="not found"):
        Compiler("/does/not/exist/ragged-compile").lower(ProgramSpec("sum"))


def test_cli_source_and_diagnostic(capsys):
    assert main(["softmax", "--feature-dim", "13", "--dtype", "bf16"]) == 0
    assert "memref<?x13xbf16>" in capsys.readouterr().out
    assert main(["sum", "--feature-dim", "0"]) == 1
    assert "feature_dim" in capsys.readouterr().err


def test_memref_view_pointer_is_not_offset_twice():
    class View:
        shape = (3, 4)
        def data_ptr(self): return 1040
        def stride(self): return (4, 1)
        def storage_offset(self): return 10
    assert memref_arguments(View()) == [1040, 1040, 0, 3, 4, 4, 1]


def test_report_distinguishes_enqueue_and_device_time():
    report = ExecutionReport(("cta",), (), ((12, 1, 1),), 0, "key", host_enqueue_ms=0.4)
    data = json.loads(report.to_json())
    assert data["host_enqueue_ms"] == 0.4
    assert data["device_elapsed_ms"] is None


@pytest.fixture
def compiler_transport(monkeypatch):
    """Synthetic compiler transport only; test bytes never reach the Driver API."""
    from types import SimpleNamespace
    state = {
        "actual": {"llvm_commit": "tested-llvm", "llvm_version": "23.1.3", "ptxas_version": "tested-ptxas",
                   "dialect_version": "1", "chip": "sm_89", "features": "+ptx80"},
        "commands": [], "stale_provenance": False,
        "compiler_hash": "1" * 64, "change_compiler": False,
    }
    compiler = Compiler("synthetic-compiler")
    monkeypatch.setattr(compiler, "_executable", lambda: "synthetic-compiler")
    monkeypatch.setattr(compiler, "_compiler_hash", lambda: state["compiler_hash"])
    def invoke(command, **kwargs):
        state["commands"].append(command)
        if "--toolchain-json" in command:
            return SimpleNamespace(returncode=0, stdout=json.dumps(state["actual"]).encode(), stderr=b"")
        output = Path(command[command.index("-o") + 1])
        output.write_bytes(b"\x7fELFsynthetic-transport-test")
        metadata = {"schema": 1, "abi": "mlir-memref64-v1", "toolchain": dict(state["actual"]),
                    "kernels": [{"name": "sum_kernel", "grid": ["B", 1, 1], "block": [128, 1, 1],
                                 "args": ["values", "offsets", "out"]}]}
        if state["stale_provenance"]:
            metadata["toolchain"]["ptxas_version"] = "stale-ptxas"
        if "metadata_override" in state:
            metadata.update(state["metadata_override"])
        if "resources" not in metadata and not state.get("omit_resources", False):
            from math import prod
            metadata["resources"] = {"status": "measured", "source": "ptxas",
                "requested": {"smem_per_cta": 49152, "regs_budget": 65536},
                "kernels": [{"name": kernel["name"], "registers_per_thread": 16,
                             "threads_per_cta": prod(kernel["block"]), "registers_per_cta": 16 * prod(kernel["block"]),
                             "static_shared_bytes": 0, "dynamic_shared_bytes": kernel.get("shared_bytes", 0),
                             "shared_bytes_per_cta": kernel.get("shared_bytes", 0), "stack_frame_bytes": 0,
                             "spill_store_bytes": 0, "spill_load_bytes": 0}
                            for kernel in metadata["kernels"] + metadata.get("validation", {}).get("kernels", [])]}
        if state["change_compiler"]:
            state["compiler_hash"] = "2" * 64
        output.with_suffix(".cubin.json").write_text(json.dumps(metadata))
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
    monkeypatch.setattr("ragged.compiler.subprocess.run", invoke)
    return compiler, state


def test_compile_uses_measured_provenance_and_passes_target(compiler_transport, tmp_path):
    compiler, state = compiler_transport
    spec, cache = ProgramSpec("sum"), ArtifactCache(tmp_path)
    program = compiler.compile(spec, cache)
    assert not program.cache_hit
    assert program.artifact.key.ptxas_version == "tested-ptxas"
    assert all("--features=+ptx80" in command and "--chip=sm_89" in command for command in state["commands"])
    hit = compiler.compile(spec, cache)
    assert hit.cache_hit
    assert sum("--emit=cubin" in command for command in state["commands"]) == 1
    state["actual"]["ptxas_version"] = "new-ptxas"
    updated = compiler.compile(spec, cache)
    assert not updated.cache_hit
    assert updated.artifact.key.digest != program.artifact.key.digest


@pytest.mark.parametrize("field", ["llvm_commit", "ptxas_version"])
def test_caller_cannot_fabricate_compiler_provenance(compiler_transport, tmp_path, field):
    compiler, state = compiler_transport
    with pytest.raises(CompilationError, match="actual"):
        compiler.compile(ProgramSpec("sum"), ArtifactCache(tmp_path), **{field: "invented"})
    assert not list(tmp_path.rglob("manifest.json"))


def test_provenance_change_during_compile_is_rejected(compiler_transport, tmp_path):
    compiler, state = compiler_transport
    state["stale_provenance"] = True
    with pytest.raises(CacheMismatch, match="provenance"):
        compiler.compile(ProgramSpec("sum"), ArtifactCache(tmp_path))
    assert not list(tmp_path.rglob("manifest.json"))


def test_compiler_binary_change_invalidates_cache(compiler_transport, tmp_path):
    compiler, state = compiler_transport
    cache, spec = ArtifactCache(tmp_path), ProgramSpec("sum")
    first = compiler.compile(spec, cache)
    state["compiler_hash"] = "2" * 64
    second = compiler.compile(spec, cache)
    assert not second.cache_hit
    assert first.artifact.key.digest != second.artifact.key.digest


def test_compiler_binary_change_during_compile_cannot_publish(compiler_transport, tmp_path):
    compiler, state = compiler_transport
    state["change_compiler"] = True
    with pytest.raises(CompilationError, match="changed during compilation"):
        compiler.compile(ProgramSpec("sum"), ArtifactCache(tmp_path))
    assert not list(tmp_path.rglob("manifest.json"))


def test_source_workspace_mismatch_cannot_publish(compiler_transport, tmp_path):
    compiler, state = compiler_transport
    args = ["values", "offsets", "out", "max_stats", "sum_stats"]
    state["metadata_override"] = {
        "workspace": [{"name": name, "shape": ["B", 7], "dtype": "f32"} for name in args[3:]],
        "kernels": [{"name": "three_pass", "grid": ["B", 1, 1], "block": [128, 1, 1], "args": args}],
    }
    with pytest.raises(CacheMismatch, match="workspace"):
        compiler.compile(ProgramSpec("softmax", feature_dim=128), ArtifactCache(tmp_path))
    assert not list(tmp_path.rglob("manifest.json"))


def test_source_debug_mismatch_cannot_publish(compiler_transport, tmp_path):
    compiler, _ = compiler_transport
    with pytest.raises(CacheMismatch, match="debug visits"):
        compiler.compile(ProgramSpec("sum", debug_visits=True), ArtifactCache(tmp_path))
    assert not list(tmp_path.rglob("manifest.json"))
