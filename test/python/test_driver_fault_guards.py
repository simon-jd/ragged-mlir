"""Host-only checks of the guardrails around the real Driver-fault tests.

Diagnostic strings here are parser fixtures, not observed CUDA errors. The
subprocess checks run only host Python and never assemble or load GPU code.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import gpu_validation_scope as scope
from scripts import reproduce_gpu as reproduction
from scripts import run_gpu_validation as validation


PROBE = Path(__file__).resolve().parents[1] / "gpu/driver_fault_probe.py"
_spec = importlib.util.spec_from_file_location("independent_driver_fault_probe", PROBE)
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)


@pytest.mark.parametrize("mode,name,code", [
    ("corrupt_elf", "CUDA_ERROR_INVALID_IMAGE", 200),
    ("wrong_target", "CUDA_ERROR_NO_BINARY_FOR_GPU", 209),
    ("runtime_wrong_target", "CUDA_ERROR_NO_BINARY_FOR_GPU", 209),
])
def test_driver_parser_accepts_only_explicit_expected_branch(mode, name, code):
    message = f"cuModuleLoadData: {name} ({code}): parser fixture; not observed CUDA"
    assert probe.validate_driver_rejection(mode, message) == {
        "operation": "cuModuleLoadData", "name": name, "code": code,
        "description": "parser fixture; not observed CUDA"}


@pytest.mark.parametrize("mode,message", [
    ("corrupt_elf", "cuModuleLoadData: CUDA_ERROR_OUT_OF_MEMORY (2): out of memory"),
    ("wrong_target", "cuModuleLoadData: CUDA_ERROR_UNKNOWN (999): unknown error"),
    ("runtime_wrong_target", "cuModuleLoadData: CUDA_ERROR_INVALID_IMAGE (200): invalid image"),
    ("corrupt_elf", "cuModuleLoadData: CUDA_ERROR_INVALID_IMAGE (209): wrong numeric code"),
    ("corrupt_elf", "cuModuleLoadData: CUDA_ERROR_NO_BINARY_FOR_GPU (200): wrong name"),
    ("wrong_target", "cuLaunchKernel: CUDA_ERROR_NO_BINARY_FOR_GPU (209): wrong operation"),
    ("valid", "cuModuleLoadData: CUDA_ERROR_INVALID_IMAGE (200): invalid image"),
    ("corrupt_elf", "cuModuleLoadData: CUDA_ERROR_INVALID_IMAGE (200): invalid image\ntrailing"),
    ("wrong_target", "prefix cuModuleLoadData: CUDA_ERROR_NO_BINARY_FOR_GPU (209): no binary"),
    ("wrong_target", "cuModuleLoadData: CUDA_ERROR_NO_BINARY_FOR_GPU (209): "),
])
def test_driver_parser_rejects_unrelated_failure_and_ambiguous_diagnostic(mode, message):
    with pytest.raises(AssertionError):
        probe.validate_driver_rejection(mode, message)


def test_offline_tool_timeout_is_real_failure_and_retains_partial_output(tmp_path):
    command = [sys.executable, "-I", "-u", "-c",
               "import time; print('host-only-started', flush=True); time.sleep(10)"]
    with pytest.raises(RuntimeError, match="timed out"):
        probe._run_logged(command, tmp_path, "host-timeout", timeout=0.25)
    assert json.loads((tmp_path / "host-timeout.command.json").read_text()) == command
    assert (tmp_path / "host-timeout.stdout.txt").read_text() == "host-only-started\n"
    assert json.loads((tmp_path / "host-timeout.exit.json").read_text()) == {
        "timed_out": True, "timeout_seconds": 0.25}


def test_offline_tool_nonzero_is_failure_and_retains_actual_exit(tmp_path):
    command = [sys.executable, "-I", "-c", "import sys; print('host-only failure', file=sys.stderr); sys.exit(7)"]
    with pytest.raises(RuntimeError, match=r"failed \(7\)"):
        probe._run_logged(command, tmp_path, "host-failure", timeout=5)
    assert (tmp_path / "host-failure.stderr.txt").read_text() == "host-only failure\n"
    assert json.loads((tmp_path / "host-failure.exit.json").read_text()) == {
        "exit_code": 7, "timed_out": False}


def test_offline_tool_success_records_actual_host_command(tmp_path):
    command = [sys.executable, "-I", "-c", "print('host-only success')"]
    result = probe._run_logged(command, tmp_path, "host-success", timeout=5)
    assert result.returncode == 0 and result.stdout == "host-only success\n"
    assert json.loads((tmp_path / "host-success.command.json").read_text()) == command
    assert json.loads((tmp_path / "host-success.exit.json").read_text()) == {
        "exit_code": 0, "timed_out": False}


def test_full_scope_partition_retains_each_node_once_with_only_api_faults_suppressed():
    main = "test/gpu/test_example.py::test_device"
    requested = [scope.API_NODEIDS[1], main, *scope.SAFE_NODEIDS,
                 scope.API_NODEIDS[0], scope.API_NODEIDS[2]]
    groups = scope.partition_sanitizer_nodeids(requested)
    assert groups["main"] == [main]
    assert set(groups["native_safe"]) == set(scope.SAFE_NODEIDS)
    assert set(groups["loader_api"]) == set(scope.API_NODEIDS)
    flattened = [node for nodes in groups.values() for node in nodes]
    assert sorted(flattened) == sorted(requested)


@pytest.mark.parametrize("nodes,full", [
    ([*scope.FAULT_NODEIDS, scope.FAULT_FUNCTION + "[new_fault]"], True),
    (list(scope.FAULT_NODEIDS[:-1]), True),
    ([*scope.FAULT_NODEIDS, scope.API_NODEIDS[0]], True),
    ([], False),
])
def test_scope_rejects_unknown_missing_duplicated_or_empty_evidence(nodes, full):
    with pytest.raises(ValueError):
        scope.partition_sanitizer_nodeids(nodes, require_all_faults=full)


def test_filtered_selection_preserves_actual_empty_groups_without_inventing_tests():
    assert scope.partition_sanitizer_nodeids([scope.API_NODEIDS[0]], require_all_faults=False) == {
        "main": [], "native_safe": [], "loader_api": [scope.API_NODEIDS[0]]}


@pytest.mark.parametrize("old,new", [
    ("^cuModuleLoadData$", ".*"),
    ("^CUDA_ERROR_INVALID_IMAGE$", "^CUDA_ERROR_.*$"),
    ("<result>200</result>", ""),
    ("<kind>Api</kind>", "<kind>Memory</kind>"),
])
def test_suppression_validator_rejects_broadening(tmp_path, old, new):
    original = Path(__file__).resolve().parents[1] / "gpu/driver-load-api-suppressions.xml"
    path = tmp_path / "broadened.xml"
    path.write_text(original.read_text().replace(old, new, 1))
    with pytest.raises(ValueError):
        scope.validate_suppression_file(path)


def test_actual_host_pytest_plugin_records_exact_partition_and_junit_identity(tmp_path):
    """Seven host assertions use synthetic names; this is not CUDA acceptance."""
    root = Path(__file__).resolve().parents[2]
    tests = tmp_path / "test/gpu"
    tests.mkdir(parents=True)
    (tests / "test_driver_faults.py").write_text(
        "import pytest\n@pytest.mark.parametrize('mode', " + repr(list(probe.MODES)) + ")\n"
        "def test_real_driver_faults_in_isolated_process(mode):\n    assert bool(mode)\n")
    (tests / "test_host_scope.py").write_text("def test_host_assertion():\n    assert 2+2==4\n")
    ini, xml, selection = tmp_path / "pytest.ini", tmp_path / "actual.xml", tmp_path / "selection.json"
    ini.write_text("[pytest]\n")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-c", str(ini), "--rootdir", str(tmp_path),
         "-p", "scripts.gpu_validation_scope", f"--junitxml={xml}", "test/gpu"],
        cwd=tmp_path, env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONPATH": str(root),
                           "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", scope.SELECTION_ENV: str(selection)},
        capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    nodes = scope.collected_nodeids(json.loads(selection.read_text()))
    assert len(nodes) == 7
    assert len(scope.partition_sanitizer_nodeids(nodes)["main"]) == 1
    assert reproduction.validate_pytest_selection(xml, selection, expected_nodeids=nodes) == {
        "tests": 7, "skipped": 0, "failures": 0, "errors": 0}
    wrong = [*nodes[:-1], "test/gpu/test_other.py::test_not_executed"]
    with pytest.raises(reproduction.Failed, match="explicitly requested sanitizer scope"):
        reproduction.validate_pytest_selection(xml, selection, expected_nodeids=wrong)


def test_runner_interruption_records_terminal_failure_not_stale_running(tmp_path, monkeypatch):
    """Injected orchestration interruption; no child, Torch, or CUDA execution."""
    fake_torch = SimpleNamespace(__version__="synthetic-host-only", __file__=__file__,
        cuda=SimpleNamespace(is_available=lambda: True, get_device_capability=lambda: (8, 9)))
    real_import = validation.importlib.import_module
    monkeypatch.setattr(validation.importlib, "import_module",
                        lambda name: fake_torch if name == "torch" else real_import(name))
    output = tmp_path / "interrupted"
    args = validation.parser().parse_args(["--tools", "plain", "--output", str(output)])
    def interrupted_runner(*args, **kwargs):
        raise KeyboardInterrupt("synthetic orchestration interruption; no CUDA process")
    try:
        code = validation.run(args, preflight=lambda **kwargs: {"status": "ready", "synthetic": True},
                              runner=interrupted_runner)
    except KeyboardInterrupt:
        code = None
    record = json.loads((output / "results.json").read_text())
    assert code == 1 and record["status"] == "failed"
    assert record["stages"][-1]["name"] == "collection"
    assert record["stages"][-1]["status"] == "failed"
    assert "KeyboardInterrupt" in record["stages"][-1]["error"]
    assert record["runs"] == []


def _assert_owned_distinct_basetemps(commands, output, source, expected_count):
    """Check actual constructed argv; synthetic runners never execute pytest."""
    bases = []
    for name, command, _ in commands:
        arguments = [arg.split("=", 1)[1] for arg in command if arg.startswith("--basetemp=")]
        execution = any(arg.startswith("--junitxml=") for arg in command)
        if not execution:
            assert arguments == [], f"non-execution stage {name} must not reuse or clear a test base"
            continue
        assert len(arguments) == 1, name
        base = Path(arguments[0])
        assert base.is_absolute() and base.parent == output / "work"
        assert base.parent.is_dir()
        assert base.resolve().is_relative_to(output.resolve())
        assert not base.resolve().is_relative_to(source.resolve())
        bases.append(base)
    assert len(bases) == len(set(bases)) == expected_count


def test_standalone_each_tool_group_owns_unique_retained_basetemp(tmp_path, monkeypatch):
    # Reuse orchestration fixtures only; do not run another test suite.
    from test.python.test_driver_fault_contract import _standalone_fixture
    code, report, calls, output = _standalone_fixture(tmp_path, monkeypatch)
    assert code == 0 and report["status"] == "passed"
    _assert_owned_distinct_basetemps(calls, output, validation.ROOT, 12)


def test_fresh_runner_correctness_and_nine_groups_keep_separate_retained_basetemps(tmp_path, monkeypatch):
    from test.python.test_driver_fault_contract import _fresh_wheel_scope_fixture
    output, report, calls, code = _fresh_wheel_scope_fixture(tmp_path, monkeypatch)
    assert code == 0 and report["status"] == "passed"
    _assert_owned_distinct_basetemps(calls, output, output / "source", 10)
