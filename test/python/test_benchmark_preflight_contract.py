"""Independent host contracts; synthetic CUDA availability is never device evidence."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import venv

import numpy as np
import pytest

from bench import matrix


def arguments(tmp_path, python=None, *, pending_only=False):
    argv = ["--output", str(tmp_path / "output"), "--ops", "sum", "--dtypes", "f32", "--ds", "16",
            "--offsets-dtypes", "i64", "--distributions", "uniform", "--normalizations", "fixed_elements",
            "--sizes", "small", "--methods", "sparse-gpu" if pending_only else "ragged-auto"]
    if python is not None:
        argv.extend(("--python", str(python)))
    return matrix.parser().parse_args(argv)


def forbidden(*args, **kwargs):
    raise AssertionError("parent preflight or benchmark child must not execute in this contract")


def alternate_venv(tmp_path, *, available):
    """A real distinct venv, using a local Torch stub and existing NumPy only."""
    directory = tmp_path / "alternate env"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(directory)
    packages = directory / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    # Only reuse dependencies; the venv's own site-packages comes first.
    (packages / "test-numpy.pth").write_text(str(Path(np.__file__).resolve().parents[1]) + "\n")
    marker = tmp_path / "imported-torch.json"
    source = f'''import json, sys
from pathlib import Path
from types import SimpleNamespace
Path({str(marker)!r}).write_text(json.dumps({{"executable": sys.executable, "prefix": sys.prefix}}))
print("synthetic alternate Torch fixture diagnostic", file=sys.stderr)
__version__ = "synthetic-alternate-Torch-not-GPU-evidence"
version = SimpleNamespace(cuda="synthetic-CUDA")
cuda = SimpleNamespace(is_available=lambda: {available!r}, current_device=lambda: 0,
    get_device_properties=lambda ordinal: SimpleNamespace(name="synthetic venv contract only",
        major=8, minor=9, total_memory=1024, multi_processor_count=1))
'''
    (packages / "torch.py").write_text(source)
    python = directory / "bin" / "python"
    assert python.resolve() == Path(sys.executable).resolve()
    assert str(python) != sys.executable
    return python, marker


@pytest.mark.parametrize("available", [False, True])
def test_real_alternate_venv_retains_symlink_identity_and_uses_its_torch(tmp_path, monkeypatch, available):
    python, marker = alternate_venv(tmp_path, available=available)
    args = arguments(tmp_path, python, pending_only=available)
    monkeypatch.setattr(matrix, "gpu_preflight", forbidden)
    report, code = matrix.run_matrix(args, runner=forbidden)
    assert code == 2
    assert report["status"] == ("partial" if available else "blocked")
    result = report["preflight"]
    assert result["status"] == ("ready" if available else "blocked")
    imported = json.loads(marker.read_text())
    assert imported == {"executable": str(python), "prefix": str(python.parent.parent)}
    probe = result["python_preflight_probe"]
    assert probe["command"][0] == str(python) and probe["exit_code"] == 0
    assert "synthetic alternate Torch fixture diagnostic" in probe["stderr"]
    assert json.loads(probe["stdout"])["status"] == result["status"]
    if available:
        assert result["torch"] == "synthetic-alternate-Torch-not-GPU-evidence"
        assert result["torch_cuda"] == "synthetic-CUDA"
        assert result["device"]["name"] == "synthetic venv contract only"
    assert report == json.loads((args.output / "matrix.json").read_text())
    assert not (args.output / "cases").exists()


def test_current_absolute_interpreter_can_probe_directly(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    calls = []
    expected = {"status": "blocked", "reason": "synthetic direct interpreter fixture"}
    monkeypatch.setattr(matrix, "gpu_preflight", lambda: calls.append(True) or expected)
    monkeypatch.setattr(matrix.subprocess, "run", forbidden)
    report, code = matrix.run_matrix(args, runner=forbidden)
    assert code == 2 and calls == [True] and report["preflight"] == expected


def test_relative_python_is_resolved_by_child_cwd_not_parent_cwd(tmp_path, monkeypatch):
    args = arguments(tmp_path, "venv/bin/python")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(matrix.sys, "executable", str(tmp_path / "venv/bin/python"))
    monkeypatch.setattr(matrix, "gpu_preflight", forbidden)
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout='{"status":"blocked","reason":"child cwd fixture"}', stderr="")
    monkeypatch.setattr(matrix.subprocess, "run", run)
    result = matrix._requested_python_preflight(args)
    assert len(calls) == 1
    assert calls[0][0][0] == "venv/bin/python" and calls[0][1]["cwd"] == matrix.ROOT
    assert result["reason"] == "child cwd fixture"


@pytest.mark.parametrize("timeout,expected", [(2.5, 2.5), (3600.0, 60.0)])
def test_probe_has_child_import_paths_and_bounded_timeout(tmp_path, monkeypatch, timeout, expected):
    args = arguments(tmp_path, tmp_path / "other/python")
    args.timeout = timeout
    monkeypatch.setenv("PYTHONPATH", "preserved-import-path")
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout='{"status":"blocked","reason":"fixture"}', stderr="warning\n")
    monkeypatch.setattr(matrix.subprocess, "run", run)
    result = matrix._requested_python_preflight(args)
    command, options = calls[0]
    assert command[0] == args.python and command[1] == "-c"
    assert options["cwd"] == matrix.ROOT and options["timeout"] == expected
    assert options["env"]["PYTHONPATH"].split(os.pathsep) == [str(matrix.ROOT / "python"), str(matrix.ROOT), "preserved-import-path"]
    assert options["capture_output"] is True and options["text"] is True and options["check"] is False
    assert result["python_preflight_probe"]["stderr"] == "warning\n"


@pytest.mark.parametrize("stdout", [
    "not JSON", "[]", "null", '{"status":"unknown"}', '{"status":"blocked"}',
    '{"status":"blocked","reason":4}',
    '{"status":"blocked","reason":"x","status":"ready"}',
    '{"status":"blocked","reason":"x","nested":{"a":1,"a":2}}',
    '{"status":"blocked","reason":"x","nested":[NaN]}',
    '{"status":"blocked","reason":"x","nested":[Infinity]}',
    '{"status":"blocked","reason":"x","nested":[1e999]}',
    '{"status":"blocked","reason":"x","nested":[-1e999]}',
    '{"status":"blocked","reason":"x"}\nstdout contamination',
])
def test_malformed_probe_is_archived_as_blocked_not_launched(tmp_path, monkeypatch, stdout):
    args = arguments(tmp_path, tmp_path / "different/python")
    monkeypatch.setattr(matrix.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout=stdout, stderr="fixture stderr"))
    report, code = matrix.run_matrix(args, runner=forbidden)
    assert code == 2 and report["status"] == "blocked"
    probe = report["preflight"]["python_preflight_probe"]
    assert probe["stdout"] == stdout and probe["stderr"] == "fixture stderr" and probe["exit_code"] == 0
    assert "execution_error" in probe
    assert json.loads((args.output / "matrix.json").read_text()) == report
    assert not (args.output / "cases").exists()


@pytest.mark.parametrize("fault", ["nonzero", "timeout", "missing"])
def test_unsuccessful_probe_keeps_diagnostics_and_never_starts_benchmark(tmp_path, monkeypatch, fault):
    args = arguments(tmp_path, tmp_path / "different/python")
    stdout = '{"status":"ready"}'
    def run(command, **kwargs):
        if fault == "nonzero":
            return SimpleNamespace(returncode=9, stdout=stdout, stderr="fixture failure")
        if fault == "timeout":
            raise subprocess.TimeoutExpired(command, 1, output=b"partial stdout", stderr=b"partial stderr")
        raise FileNotFoundError("fixture missing interpreter")
    monkeypatch.setattr(matrix.subprocess, "run", run)
    report, code = matrix.run_matrix(args, runner=forbidden)
    assert code == 2 and report["status"] == "blocked"
    probe = report["preflight"]["python_preflight_probe"]
    assert "execution_error" in probe
    if fault == "nonzero":
        assert (probe["exit_code"], probe["stdout"], probe["stderr"]) == (9, stdout, "fixture failure")
    elif fault == "timeout":
        assert (probe["exit_code"], probe["stdout"], probe["stderr"]) == (None, "partial stdout", "partial stderr")
    else:
        assert probe["exit_code"] is None and "FileNotFoundError" in probe["execution_error"]
    assert report == json.loads((args.output / "matrix.json").read_text())


def test_plan_only_does_not_execute_requested_interpreter(tmp_path, monkeypatch):
    args = arguments(tmp_path, tmp_path / "missing-python")
    args.plan_only = True
    monkeypatch.setattr(matrix, "gpu_preflight", forbidden)
    monkeypatch.setattr(matrix.subprocess, "run", forbidden)
    report, code = matrix.run_matrix(args, runner=forbidden)
    assert code == 0 and report["status"] == "planned" and report["preflight"] is None
    assert report["rows"][0]["command"][0] == args.python
