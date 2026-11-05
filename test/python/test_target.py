"""The target chip (sm_86 or sm_89, default sm_89) through spec, compiler,
runtime and cost model; precompiled artifacts loaded without a compiler."""
from contextlib import nullcontext
from dataclasses import dataclass, replace
import json
import os
from math import prod
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest

from bench import calibrate
from bench.precompile import specs
from bench.run import build_parser, ragged_spec
from native_fake import PlanModule
from ragged import (Artifact, ArtifactCache, CacheKey, CacheMiss, CompiledProgram, Compiler, ProgramSpec,
                    RuntimeUnavailable, ToolchainUnavailable, ValidationError)
from ragged.model import CHIPS, DEFAULT_CHIP, runs_on
from ragged.selection import CostModel, features

TOOLCHAIN = {"llvm_commit": "c" * 40, "llvm_version": "23.1.3", "ptxas_version": "13.2", "dialect_version": "1",
             "features": "+ptx80"}


def test_specs_default_to_sm89_and_accept_sm86():
    assert ProgramSpec("sum").chip == DEFAULT_CHIP == "sm_89" and set(CHIPS) == {"sm_86", "sm_89"}
    for chip in ("sm_80", "sm_87", "sm_90", "89"):
        with pytest.raises(ValidationError, match="sm_86 or sm_89"):
            ProgramSpec("sum", chip=chip)
    sm89, sm86 = ProgramSpec("max", feature_dim=7), ProgramSpec("max", feature_dim=7, chip="sm_86")
    # The source is the same; the target separates cache entries and signatures.
    assert sm86.ir_hash == sm89.ir_hash and sm86.signature.endswith(":sm_86") and sm89.signature.endswith(":sm_89")
    keys = [CacheKey.from_spec(spec, llvm_commit="l", ptxas_version="p") for spec in (sm89, sm86)]
    assert keys[0].digest != keys[1].digest and keys[1].chip == "sm_86"
    assert ProgramSpec.from_dict(sm86.to_dict()) == sm86


def test_chips_have_the_same_per_cta_limits():
    # Compute capabilities 8.6 and 8.9 share these limits; the defaults equal them.
    for limits in CHIPS.values():
        assert (limits.shared_per_cta, limits.registers_per_cta, limits.threads_per_cta) == (49152, 65536, 1024)
    spec = ProgramSpec("sum", chip="sm_86")
    assert (spec.smem_per_cta, spec.regs_budget) == (49152, 65536)


@pytest.mark.parametrize("chip,capability,expected", [
    ("sm_89", (8, 9), True), ("sm_86", (8, 6), True), ("sm_86", (8, 9), True),
    ("sm_89", (8, 6), False), ("sm_86", (9, 0), False), ("sm_89", (7, 5), False)])
def test_cubins_run_on_their_chip_and_later_minor_versions(chip, capability, expected):
    assert runs_on(chip, capability) is expected


def synthetic_program(chip):
    spec = ProgramSpec("sum", feature_dim=4, strategy="naive", chip=chip)
    key = CacheKey.from_spec(spec, llvm_commit="host-test", ptxas_version="host-test", compiler_sha256="1" * 64)
    metadata = {"schema": 1, "abi": "mlir-memref64-v1", "chip": chip, "compiler_sha256": key.compiler_sha256,
                "toolchain": {name: getattr(key, name)
                              for name in ("llvm_commit", "ptxas_version", "dialect_version", "chip", "features")},
                "kernels": [{"name": "sum", "grid": ["B", 1, 1], "block": [128, 1, 1],
                             "args": ["values", "offsets", "out"]}]}
    return CompiledProgram(spec, Artifact(Path("synthetic-target"), key, b"\x7fELFtarget", metadata)), key


@pytest.mark.parametrize("chip,capability,runs", [
    ("sm_86", (8, 6), True), ("sm_86", (8, 9), True), ("sm_89", (8, 9), True), ("sm_89", (8, 6), False)])
def test_runtime_checks_the_device_and_loads_the_module_for_the_chip(monkeypatch, chip, capability, runs):
    import ragged.runtime as runtime

    @dataclass(frozen=True)
    class Device:
        index: int = 0

    class Stream:
        device, cuda_stream = Device(), 7

    class Tensor:
        is_cuda, requires_grad, device = True, False, Device()

        def __init__(self, shape, dtype, pointer):
            self.shape, self.dtype, self.pointer, self.ndim = shape, dtype, pointer, len(shape)

        def is_contiguous(self): return True
        def numel(self): return prod(self.shape)
        def element_size(self): return 8 if self.dtype == "i64" else 4
        def data_ptr(self): return self.pointer
        def stride(self): return (self.shape[1], 1) if self.ndim == 2 else (1,)
        def record_stream(self, stream): pass

    stream = Stream()
    fake_torch = SimpleNamespace(
        Tensor=Tensor, float32="f32", bfloat16="bf16", int32="i32", int64="i64", uint8="u8",
        empty=lambda shape, *, dtype, device: Tensor(shape, dtype, 0x9000),
        cuda=SimpleNamespace(Stream=Stream, device=lambda _: nullcontext(), get_device_capability=lambda _: capability,
                             is_current_stream_capturing=lambda: False, current_stream=lambda _: stream,
                             stream=lambda _: nullcontext()))
    loaded, launches = [], []

    def driver_module(cubin, target="sm_89"):
        loaded.append(target)
        return PlanModule(lambda *a: launches.append(a))

    program, key = synthetic_program(chip)
    monkeypatch.setattr(runtime, "_torch", lambda: fake_torch)
    monkeypatch.setattr(runtime, "_checked_devices", set())
    monkeypatch.setattr(runtime, "_resident_modules", {})
    monkeypatch.setattr(runtime, "_plans", {})
    monkeypatch.setitem(sys.modules, "ragged._native", SimpleNamespace(DriverModule=driver_module))
    call = lambda: program.run(Tensor((6, 4), "f32", 0x1000), Tensor((4,), "i64", 0x2000),  # noqa: E731
                               Tensor((3, 4), "f32", 0x3000), stream=stream)
    if not runs:
        with pytest.raises(RuntimeUnavailable, match=f"{chip} does not run on the current device"):
            call()
        assert not loaded
        return
    call()
    assert loaded == [chip] and len(launches) == 1


def test_cost_model_counts_bytes_beyond_the_given_l2():
    model = CostModel.load()
    assert model.l2_mib == 72.0
    # 64 MiB of D=64 rows plus 16 MiB of output: within a 72 MiB L2 by 8 MiB short, beyond a 6 MiB one by 74.
    n, b, d = 262144, 65536, 64
    assert features(n, b, d, 64, 32, op="sum")[5] == 8.0
    assert features(n, b, d, 64, 32, op="sum", l2_mib=6.0)[5] == 74.0
    chosen, info = model.choose("sum", n, b, d, 64, 32)
    assert info["l2_mib"] == 72.0
    swapped, info6 = model.choose("sum", n, b, d, 64, 32, l2_mib=6.0)
    assert info6["l2_mib"] == 6.0
    assert info6["predicted_us"] != info["predicted_us"]
    with pytest.raises(ValueError):
        model.choose("sum", n, b, d, 64, 32, l2_mib=0)


def write_record(path, chips=("sm_86",), sha="2" * 64):
    path.write_text(json.dumps({"schema": 1, "compiler_sha256": sha,
                                "toolchains": [dict(TOOLCHAIN, chip=chip) for chip in chips]}))
    return path


def test_toolchain_record_identifies_without_a_compiler(tmp_path):
    compiler = Compiler("no-such-ragged-compile", toolchain_record=write_record(tmp_path / "record.json"))
    assert compiler.toolchain(chip="sm_86") == dict(TOOLCHAIN, chip="sm_86")
    assert compiler._compiler_hash() == "2" * 64
    with pytest.raises(ToolchainUnavailable, match="no sm_89"):
        compiler.toolchain()
    with pytest.raises(ToolchainUnavailable, match="runs no compiler"):
        compiler.lower(ProgramSpec("sum", chip="sm_86"))
    with pytest.raises(CacheMiss, match="was not precompiled"):
        compiler.compile(ProgramSpec("sum", chip="sm_86"), ArtifactCache(tmp_path / "cache"))
    with pytest.raises(ToolchainUnavailable, match="written from the real compiler"):
        compiler.record_toolchain(tmp_path / "again.json")


@pytest.mark.parametrize("data", [{}, {"schema": 1, "compiler_sha256": "x", "toolchains": [{}]},
                                  {"schema": 2, "compiler_sha256": "2" * 64, "toolchains": []},
                                  {"schema": 1, "compiler_sha256": "2" * 64, "toolchains": [{"chip": "sm_86"}]}])
def test_malformed_toolchain_records_are_rejected(tmp_path, data):
    path = tmp_path / "record.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ToolchainUnavailable):
        Compiler(toolchain_record=path)


def test_bench_targets_the_chip_it_is_given():
    args = build_parser().parse_args(["--op", "softmax", "--d", "16", "--output", "unused"])
    args.length_bounds = {}
    assert args.chip == "sm_89" and ragged_spec("ragged-auto", args, 1.0, False).chip == "sm_89"
    args = build_parser().parse_args(["--chip", "sm_86", "--op", "softmax", "--d", "16", "--output", "unused"])
    args.length_bounds = {}
    spec = ragged_spec("ragged-naive", args, 1.0, False)
    assert spec.chip == "sm_86" and spec.strategy == "naive"
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--chip", "sm_90", "--output", "unused"])


def test_precompile_lists_both_schedules_of_every_case_once():
    selected = specs(["heldout2"], [711, 712, 713], "sm_86")
    # 8 shapes x 3 operations, auto and naive; distributions and seeds share programs.
    assert len(selected) == 8 * 3 * 2
    assert {spec.chip for spec in selected.values()} == {"sm_86"}
    assert {spec.strategy for spec in selected.values()} == {"auto", "naive"}


def run_file(root, group, name, seed, l2_bytes):
    path = root / group / name / f"seed-{seed}" / "rep-0" / "results.json"
    path.parent.mkdir(parents=True)
    case = {"op": "sum", "elements": 2**20, "segments": 4096, "D": 64, "max_length": 512, "dtype": "f32"}
    driver = {} if l2_bytes is None else {"l2_cache_size_bytes": l2_bytes}
    path.write_text(json.dumps({"case": case, "cuda_driver": driver, "methods": [
        {"method": "ragged-classified", "status": "measured", "complete_call_cuda_event_ms": {"median": 0.05}},
        {"method": "ragged-naive", "status": "measured", "complete_call_cuda_event_ms": {"median": 0.04}}]}))


def test_calibration_takes_the_l2_size_from_the_runs(tmp_path):
    run_file(tmp_path, "heldout2", "a", 711, 6 << 20)
    found = calibrate.collect(tmp_path, "warm", "heldout2")
    assert found[("a", 711)]["l2_mib"] == 6.0
    assert calibrate.l2_of([(key, entry) for key, entry in found.items()]) == 6.0
    run_file(tmp_path, "main", "b", 711, None)
    with pytest.raises(SystemExit, match="L2 size"):
        calibrate.collect(tmp_path, "warm", "main")
    with pytest.raises(SystemExit, match="different L2 sizes"):
        calibrate.l2_of([(("x",), {"l2_mib": 6.0}), (("y",), {"l2_mib": 72.0})])


@pytest.mark.skipif(not shutil.which(os.environ.get(
    "RAGGED_COMPILE", str(Path(__file__).resolve().parents[2] / "build/bin/ragged-compile"))),
    reason="real ragged-compile is not built")
def test_sm86_programs_load_from_a_record_without_the_compiler(tmp_path):
    executable = os.environ.get("RAGGED_COMPILE", str(Path(__file__).resolve().parents[2] / "build/bin/ragged-compile"))
    compiler, cache = Compiler(executable), ArtifactCache(tmp_path / "cache")
    spec = ProgramSpec("softmax", "bf16", 24, chip="sm_86")
    compiled = compiler.compile(spec, cache)
    assert compiled.artifact.metadata["chip"] == compiled.artifact.key.chip == "sm_86"
    assert compiled.artifact.metadata["toolchain"]["chip"] == "sm_86" and compiled.naive is not None
    assert ".target sm_86" in compiler.lower(spec, emit="ptx").output.decode()
    record = tmp_path / "toolchain.json"
    compiler.record_toolchain(record, chips=("sm_86",))
    loader = Compiler("no-such-ragged-compile", toolchain_record=record)
    loaded = loader.compile(spec, cache)
    assert loaded.cache_hit and loaded.artifact.cubin == compiled.artifact.cubin
    assert loaded.naive.artifact.cubin == compiled.naive.artifact.cubin
    with pytest.raises(CacheMiss):
        loader.compile(replace(spec, feature_dim=25), cache)
    # The same source for sm_89 is another artifact.
    assert compiler.compile(replace(spec, chip="sm_89"), cache).artifact.cubin != compiled.artifact.cubin
