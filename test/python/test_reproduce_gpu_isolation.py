"""Narrow GPU reproduction checks on synthetic evidence; no GPU claims.

Reuses orchestration fixtures only, never invokes the existing contract suite or
downloads dependencies. Local subprocess tests execute only the host Python.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import zipfile

import pytest

from scripts import reproduce_gpu as reproduction
from test.python.test_reproduce_gpu_contract import _comparison_fixture, _minimal_source, _pipeline


def test_loaded_native_must_be_the_exact_member_of_the_installed_fresh_wheel(tmp_path, monkeypatch):
    def move_loaded_native(output, source, stream):
        path = output / "results/fresh-identity.json"
        record = json.loads(path.read_text())
        foreign = output / "venv/foreign/_native.so"
        foreign.parent.mkdir()
        foreign.write_bytes(b"synthetic foreign native module; never loaded or executed\n")
        record["modules"]["ragged._native"] = {
            "file": str(foreign), "sha256": reproduction.digest(foreign)}
        path.write_text(json.dumps(record))

    _, report, calls, code = _pipeline(
        tmp_path, monkeypatch, mutate_stage="fresh-identity", mutate=move_loaded_native)
    assert code == 1, "a fresh wheel on disk cannot qualify a different imported native module"
    assert report["stages"][-1]["name"] == "fresh-identity"
    assert report["stages"][-1]["status"] == "failed"
    assert "gpu-correctness" not in [name for name, *_ in calls]


@pytest.mark.parametrize("member", ["ragged/__init__.py", "ragged/../escape.py"])
def test_fresh_wheel_rejects_duplicate_or_escaping_package_member(tmp_path, monkeypatch, member):
    def corrupt_member(output, source, stream):
        wheel, = (output / "wheels").glob("*.whl")
        with zipfile.ZipFile(wheel, "a") as archive:
            if member == "ragged/__init__.py":
                with pytest.warns(UserWarning, match="Duplicate name"):
                    archive.writestr(member, b"synthetic ambiguous package member\n")
            else:
                archive.writestr(member, b"synthetic escaping package member\n")
    _, report, calls, code = _pipeline(
        tmp_path, monkeypatch, mutate_stage="build-wheel", mutate=corrupt_member)
    assert code == 1 and report["stages"][-1]["name"] == "build-wheel"
    assert "install-wheel" not in [name for name, *_ in calls]


def test_junit_same_count_cannot_substitute_another_test_identity(tmp_path):
    xml = tmp_path / "gpu.xml"
    xml.write_text('<testsuites><testsuite tests="1" errors="0" failures="0" skipped="0">'
                   '<testcase classname="test.gpu.test_correctness" name="test_unrelated">'
                   '<properties><property name="ragged_nodeid" '
                   'value="test/gpu/test_correctness.py::test_unrelated"/></properties></testcase>'
                   '</testsuite></testsuites>')
    identity = tmp_path / "identity.json"
    identity.write_text(json.dumps({"pytest_selection": {
        "collected": 1,
        "nodeids": ["test/gpu/test_correctness.py::test_required"],
        "deselected": [],
    }}))
    with pytest.raises(reproduction.Failed):
        reproduction.validate_pytest_selection(xml, identity)


@pytest.mark.parametrize("fault", ["bool_count", "duplicate_property", "duplicate_execution", "missing_property"])
def test_junit_collection_requires_unique_unambiguous_nodeid_proof(tmp_path, fault):
    node = "test_probe.py::test_required"
    prop = f'<property name="ragged_nodeid" value="{node}"/>'
    properties = prop * 2 if fault == "duplicate_property" else "" if fault == "missing_property" else prop
    case = '<testcase classname="test_probe" name="test_required"><properties>' + properties + '</properties></testcase>'
    xml = tmp_path / "gpu.xml"
    xml.write_text('<testsuites><testsuite>' + case * (2 if fault == "duplicate_execution" else 1) + '</testsuite></testsuites>')
    nodes = [node, "test_probe.py::test_other"] if fault == "duplicate_execution" else [node]
    identity = tmp_path / "identity.json"
    identity.write_text(json.dumps({"pytest_selection": {
        "collected": True if fault == "bool_count" else len(nodes), "nodeids": nodes, "deselected": []}}))
    with pytest.raises(reproduction.Failed):
        reproduction.validate_pytest_selection(xml, identity)


def test_actual_host_pytest_worker_records_matching_parameterized_nodeids(tmp_path):
    """Runs two actual CPU assertions; mocked identity is never GPU evidence."""
    source = tmp_path / "source"
    source.mkdir()
    test_file = source / "test_probe.py"
    test_file.write_text("import pytest\n@pytest.mark.parametrize('value', [1, 2])\n"
                         "def test_host_assertion(value):\n    assert value > 0\n")
    ini, xml, identity = tmp_path / "pytest.ini", tmp_path / "actual.xml", tmp_path / "identity.json"
    ini.write_text("[pytest]\n")
    root = Path(__file__).resolve().parents[2]
    code = (
        "import sys; sys.path.insert(0,sys.argv.pop(1)); "
        "from scripts import reproduce_gpu_worker as worker; "
        "worker.identity=lambda *args: {'status':'verified','synthetic_identity':True}; "
        "raise SystemExit(worker.main(sys.argv[1:]))"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", code, str(root), "pytest", "--source", str(source),
         "--venv", str(tmp_path / "synthetic-venv"), "--expected-uuid", "synthetic-no-GPU",
         "--identity-output", str(identity), "--", "-q", "-c", str(ini),
         "--rootdir", str(source), f"--junitxml={xml}", str(test_file)],
        cwd=tmp_path, env={"PATH": "/usr/bin:/bin", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
        capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    assert reproduction.validate_pytest_selection(xml, identity) == {
        "tests": 2, "skipped": 0, "failures": 0, "errors": 0}
    selection = json.loads(identity.read_text())["pytest_selection"]
    assert selection["nodeids"] == ["test_probe.py::test_host_assertion[1]",
                                    "test_probe.py::test_host_assertion[2]"]


def test_bounded_run_timeout_kills_real_child_and_grandchild(tmp_path):
    """Actual host process-group cleanup; no mocked wait/kill and no GPU work."""
    pid_file = tmp_path / "pids.json"
    code = (
        "import json,os,pathlib,subprocess,sys,time; "
        "child=subprocess.Popen([sys.executable,'-I','-c','import time; time.sleep(30)']); "
        "pathlib.Path(sys.argv[1]).write_text(json.dumps([os.getpid(),child.pid])); "
        "time.sleep(30)"
    )
    pids = []
    started = time.monotonic()
    try:
        with (tmp_path / "timeout.log").open("w") as stream:
            with pytest.raises(subprocess.TimeoutExpired) as error:
                reproduction.bounded_run(
                    [sys.executable, "-I", "-c", code, str(pid_file)],
                    cwd=tmp_path, env={"PATH": "/usr/bin:/bin"}, stdout=stream,
                    stderr=subprocess.STDOUT, timeout=1)
        assert error.value.returncode == -signal.SIGKILL
        assert time.monotonic() - started < 10
        pids = json.loads(pid_file.read_text())
        # A killed grandchild can briefly remain a zombie until init reaps it;
        # it must not retain a live process or continue executing code.
        for pid in pids:
            status = Path(f"/proc/{pid}/stat")
            deadline = time.monotonic() + 1
            while True:
                try:
                    state = status.read_text().split(")", 1)[1].strip().split()[0]
                except (FileNotFoundError, ProcessLookupError):
                    # A process that exits between open and read raises ESRCH.
                    break
                if state == "Z":
                    break
                # SIGKILL delivery to the grandchild need not finish before
                # wait() reaps the direct child; allow its scheduler one second.
                assert time.monotonic() < deadline, f"child {pid} remained live after process-group SIGKILL"
                time.sleep(0.01)
    finally:
        # Cleanup only the child PIDs created and retained by this test.
        if not pids and pid_file.exists():
            pids = json.loads(pid_file.read_text())
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def _worker_fixture(tmp_path, monkeypatch):
    """Real worker proof logic around explicitly synthetic benchmark output.

    GPU identity and the benchmark call are stubs; the worker's strict JSON,
    NPZ, metrics and implementation validation functions are not replaced.
    """
    from scripts import reproduce_gpu_worker as worker
    from bench import run as benchmark

    output = tmp_path / "worker-fixture"
    output.mkdir()
    source = _minimal_source(output / "source")
    for relative in ("bench/run.py", "baselines/triton_reference.py", "baselines/torch_reference.py"):
        (source / relative).write_text("synthetic source identity fixture; never executed\n")
    for relative in ("build/bin/ragged-compile", "native/libragged_baselines.so"):
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic executable identity fixture; never executed\n")
    destination = output / "comparisons/sum"
    forwarded = ["--op", "sum", "--dtype", "f32", "--offsets-dtype", "i64", "--d", "1",
                 "--distribution", "long_tail", "--segments", "128", "--total-elements", "8192",
                 "--seed", "711", "--length-hints", "--warmup", "3", "--repeats", "10",
                 "--compiler", str(output / "build/bin/ragged-compile"),
                 "--native-library", str(output / "native/libragged_baselines.so"),
                 "--methods", *reproduction.COMMON_METHODS, "torch-segment", "--output", str(destination)]
    _comparison_fixture(output, source, "sum", ["--", *forwarded])
    results = destination / "results.json"
    data = json.loads(results.read_text())
    for cell in data["methods"]:
        for field in ("complete_call_cuda_event_ms", "complete_call_wall_until_complete_ms"):
            cell[field] = {"samples": [1.0] * 10, "count": 10, "median": 1.0,
                           "min": 1.0, "max": 1.0, "p90": 1.0}
    results.write_text(json.dumps(data))
    uuid = "GPU-00010203-0405-0607-0809-0a0b0c0d0e0f"
    record = {"status": "verified", "torch": data["torch"], "torch_cuda": data["torch_cuda"],
              "torch_device": data["device"], "driver": {"driver_supported_cuda_version": 13020}}
    monkeypatch.setattr(worker, "identity", lambda *args: dict(record))
    def synthetic_benchmark_exit():
        raise SystemExit(0)
    monkeypatch.setattr(benchmark, "main", synthetic_benchmark_exit)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setattr(sys, "argv", sys.argv.copy())
    identity = output / "identity.json"
    command = ["benchmark", "--source", str(source), "--venv", str(output / "venv"),
               "--expected-uuid", uuid, "--identity-output", str(identity), "--", *forwarded]
    return worker, command, results, identity


def test_worker_actual_proof_path_accepts_consistent_synthetic_evidence(tmp_path, monkeypatch):
    worker, command, results, identity = _worker_fixture(tmp_path, monkeypatch)
    assert worker.main(command) == 0
    proof = json.loads(identity.read_text())["benchmark_data_validation"]
    assert proof["result_sha256"] == reproduction.digest(results)
    assert proof["case_npz_sha256"] == reproduction.digest(results.parent / "case.npz")
    assert proof["validated_methods"] == [*reproduction.COMMON_METHODS, "torch-segment"]


@pytest.mark.parametrize("fault", ["short_samples", "wrong_warmup", "npz_changed", "foreign_native",
                                  "allocator_mismatch", "overflow_json", "duplicate_json"])
def test_worker_actual_proof_path_rejects_inconsistent_evidence(tmp_path, monkeypatch, fault):
    worker, command, results, identity = _worker_fixture(tmp_path, monkeypatch)
    record = json.loads(results.read_text())
    first = record["methods"][0]
    if fault == "short_samples":
        for field in ("complete_call_cuda_event_ms", "complete_call_wall_until_complete_ms"):
            first[field]["samples"] = [1.0] * 3
            first[field]["count"] = 3
    elif fault == "wrong_warmup":
        first["warmup_calls"] = 0
    elif fault == "npz_changed":
        import numpy as np
        npz = results.parent / "case.npz"
        with np.load(npz, allow_pickle=False) as saved:
            values, offsets = saved["values"], saved["offsets"]
        values[0, 0] = 99.0
        np.savez(npz, values=values, offsets=offsets)
    elif fault == "foreign_native":
        next(cell for cell in record["methods"] if cell["method"] == "cub")["library_sha256"] = "0" * 64
    elif fault == "allocator_mismatch":
        first["allocator_peak_allocated_bytes"] += 1
    results.write_text(json.dumps(record))
    if fault == "overflow_json":
        results.write_text(results.read_text().replace('"max_abs_error": 0.0', '"max_abs_error": 1e999', 1))
    elif fault == "duplicate_json":
        results.write_text(results.read_text().replace('"status": "complete"', '"status": "complete", "status": "complete"', 1))
    with pytest.raises((RuntimeError, ValueError)):
        worker.main(command)
    failed = json.loads(identity.read_text())
    assert failed["status"] == "failed" and "benchmark_data_validation" not in failed
