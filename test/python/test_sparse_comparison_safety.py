"""Sparse callable safety and evidence regressions, not GPU results."""
import numpy as np
import pytest
from types import SimpleNamespace
from pathlib import Path

from scripts import sparse_comparison_support as support


def test_sparse_cabi_rejects_strided_offsets_before_stride_one_descriptor():
    # Logical offsets are valid; the physical adjacent words are not. A native
    # descriptor with hard-coded stride one would silently change the CSR.
    offsets = np.array([0, 999, 2, 999, 4], np.int64)[::2]
    values = np.ones((4, 1), np.float32)
    library = object.__new__(support.SparseLibrary)
    library.b, library.width = 2, 2
    calls = []
    library.maximum = lambda *args: calls.append(args)
    with pytest.raises(ValueError):
        library.call("max", values, offsets)
    assert not calls


def test_cuda_resident_adapter_rejects_host_tensors_before_calling_mlir():
    # This is an actual CPU tensor guard check. No AOT module is loaded, and the
    # stub records entry only; it computes no output or simulated GPU result.
    import torch
    values = torch.ones((4, 1), dtype=torch.float32)
    offsets = torch.tensor([0, 2, 4], dtype=torch.int64)
    class Library:
        def call(self, *args, **kwargs):
            raise AssertionError("host-domain inputs reached the claimed CUDA-resident callable")
    with pytest.raises(ValueError):
        support.candidate_cuda_callable(Library(), values, offsets)("max")


def test_sparse_stage_evidence_does_not_persist_arbitrary_inherited_environment(tmp_path, monkeypatch):
    from scripts import run_sparse_comparison as driver
    # Deliberately synthetic: never read the real process environment or any
    # actual credential. The child may receive environment, but evidence must
    # record only explicit non-sensitive provenance overrides.
    sentinel = "synthetic-private-value-not-a-real-credential"
    inherited = {"PATH": "/usr/bin", "CUDA_ROOT": "/old/cuda", "PRIVATE_TOKEN": sentinel}
    monkeypatch.setattr(driver.os, "environ", inherited)
    env = driver.child_environment(Path("/synthetic/cuda"))
    recorder = driver.Recorder(tmp_path, {"stages": []}, env, 1.0,
                        lambda *args, **kwargs: SimpleNamespace(returncode=0))
    recorder.run("host-guard-only", ["synthetic-no-process"])
    contents = (tmp_path / "comparison.json").read_text()
    assert sentinel not in contents and "PRIVATE_TOKEN" not in contents
    assert "/synthetic/cuda" in contents


def test_saved_inputs_cannot_label_multiple_features_as_the_d1_comparison_domain(tmp_path):
    from scripts import run_sparse_comparison as driver
    values = np.ones((4, 2), np.float32)
    offsets = np.array([0, 2, 4], np.int64)
    coords = support.coordinates(offsets)
    path = tmp_path / "invalid-feature-width.npz"
    np.savez(path, values=values, offsets=offsets, coordinates=coords)
    record = driver.file_identity(path) | {"identity": support.case_identity(values, offsets, coords)}
    assert record["identity"]["D"] == 1  # An invalid declared identity is not evidence.
    with pytest.raises((driver.Failed, ValueError)):
        driver.load_saved_case(record)


def test_recorder_itself_never_serializes_unlisted_environment_fields(tmp_path):
    from scripts import run_sparse_comparison as driver
    # Defence at the evidence sink, independent of the normal caller's filter.
    sentinel = "synthetic-recorder-secret-not-a-real-credential"
    env = {"CUDA_ROOT": "/synthetic/cuda", "PRIVATE_TOKEN": sentinel}
    recorder = driver.Recorder(tmp_path, {"stages": []}, env, 1.0,
                               lambda *args, **kwargs: SimpleNamespace(returncode=0))
    recorder.run("synthetic-no-process", ["synthetic-command"])
    contents = (tmp_path / "comparison.json").read_text()
    assert sentinel not in contents and "PRIVATE_TOKEN" not in contents
    assert "/synthetic/cuda" in contents


@pytest.mark.parametrize("name,prefix", [("llvm-version", "LLVM version"),
                                        ("translate-version", "LLVM version"),
                                        ("clang-version", "clang version")])
@pytest.mark.parametrize("text", ["{prefix} 23.1.30", "{prefix} 123.1.3",
    "{prefix} 23.1.3-custom", "a note mentions 23.1.3 but is not a version banner",
    "{prefix} 23.1.3\n{prefix} 23.1.3", "{prefix} 23.1.3\n{prefix} 24.0.0"])
def test_primary_compiler_version_must_be_exact_unique_banner(name, prefix, text):
    from scripts import run_sparse_comparison as driver
    with pytest.raises(driver.Blocked):
        driver.validate_tool_version(name, text.format(prefix=prefix))


@pytest.mark.parametrize("name,text,expected", [
    ("llvm-version", "LLVM (http://llvm.org/):\n  LLVM version 23.1.3\n  Optimized build.", {"version": "23.1.3"}),
    ("clang-version", "clang version 23.1.3\nTarget: x86_64-unknown-linux-gnu", {"version": "23.1.3"}),
    ("ptxas-version", "ptxas: NVIDIA (R) Ptx optimizing assembler\nCuda compilation tools, release 13.2, V13.2.86\nBuild cuda_13.2", {"release": "13.2", "version": "13.2.86"}),
])
def test_pinned_primary_tool_version_banners_are_accepted(name, text, expected):
    from scripts import run_sparse_comparison as driver
    assert driver.validate_tool_version(name, text) == expected


@pytest.mark.parametrize("text", [
    "Cuda compilation tools, release 13.2, V13.2.860",
    "Cuda compilation tools, release 13.3, V13.2.86",
    "information only: V13.2.86",
    "Cuda compilation tools, release 13.2, V13.2.86\nCuda compilation tools, release 13.2, V13.2.86",
])
def test_ptxas_version_cannot_be_substring_or_conflicting_release(text):
    from scripts import run_sparse_comparison as driver
    with pytest.raises(driver.Blocked):
        driver.validate_tool_version("ptxas-version", text)


def _cub_build_fixture(tmp_path):
    """Synthetic metadata-only fixture; bytes are never loaded or run."""
    from scripts import run_sparse_comparison as driver
    library = tmp_path / "synthetic-never-loaded.so"
    library.write_bytes(b"synthetic manifest ingestion only, not a shared object")
    cuda = tmp_path / "synthetic-cuda"
    source = driver.ROOT / "baselines/cuda/baselines.cu"
    manifest = {"schema": 1, "exit_code": 0, "source_unchanged": True,
        "cccl_expected": "3.5.0", "source_sha256": driver.digest(source),
        "library_sha256": driver.digest(library),
        "nvcc_version": "Cuda compilation tools, release 13.2, V13.2.86\n",
        "command": [str(cuda / "bin/nvcc"), "-arch=sm_89", str(source), "-o", str(library)]}
    return driver, library, cuda, manifest


def test_cub_producer_manifest_binds_current_native_source_and_library(tmp_path):
    import json
    driver, library, cuda, manifest = _cub_build_fixture(tmp_path)
    path = tmp_path / "build.json"
    path.write_text(json.dumps(manifest))
    proof = driver.validate_cub_build(library, cuda)
    assert proof == driver.file_identity(path) | {"manifest": manifest}


@pytest.mark.parametrize("fault", ["missing", "stale-source", "stale-library", "wrong-target",
    "wrong-cuda", "conflicting-target", "version-substring", "build-failed", "duplicate-json-key"])
def test_cub_unbound_or_conflicting_producer_cannot_qualify(tmp_path, fault):
    import json
    driver, library, cuda, manifest = _cub_build_fixture(tmp_path)
    path = tmp_path / "build.json"
    if fault == "stale-source":
        manifest["source_sha256"] = "0" * 64
    elif fault == "stale-library":
        manifest["library_sha256"] = "0" * 64
    elif fault == "wrong-target":
        manifest["command"][1] = "-arch=sm_90"
    elif fault == "wrong-cuda":
        manifest["command"][0] = str(tmp_path / "other-cuda/bin/nvcc")
    elif fault == "conflicting-target":
        manifest["command"].insert(2, "--gpu-architecture=sm_90")
    elif fault == "version-substring":
        manifest["nvcc_version"] = "Cuda compilation tools, release 13.2, V13.2.860"
    elif fault == "build-failed":
        manifest["exit_code"] = 1
    if fault != "missing":
        text = json.dumps(manifest)
        if fault == "duplicate-json-key":
            text = text[:-1] + ', "exit_code": 0}'
        path.write_text(text)
    with pytest.raises((driver.Blocked, driver.Failed, ValueError)):
        driver.validate_cub_build(library, cuda)
