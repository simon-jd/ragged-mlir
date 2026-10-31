"""Independent host contracts for real Driver faults and sanitizer isolation.

Synthetic spies below exercise Python gates only. They neither load CUDA images
nor provide device correctness/sanitizer evidence. Frozen offline images are
read without rebuilding or executing them.
"""
from __future__ import annotations

from contextlib import nullcontext
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest

from scripts import gpu_validation_scope as scope
from scripts import reproduce_gpu as reproduction
from scripts import run_gpu_validation as standalone

ROOT = Path(__file__).resolve().parents[2]
OFFLINE = ROOT / "evidence/driver-faults/offline-images"
SUPPRESSION = ROOT / "test/gpu/driver-load-api-suppressions.xml"
_spec = importlib.util.spec_from_file_location("independent_driver_fault_probe", ROOT / "test/gpu/driver_fault_probe.py")
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)
MAIN = "test/gpu/test_correctness.py::test_synthetic_host_selection_fixture"


def _json(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, allow_nan=False) + "\n")


@pytest.mark.parametrize("mode,name,code", [
    ("corrupt_elf", "CUDA_ERROR_INVALID_IMAGE", 200),
    ("corrupt_elf", "CUDA_ERROR_UNKNOWN", 999),  # driver 595.99
    ("wrong_target", "CUDA_ERROR_NO_BINARY_FOR_GPU", 209),
    ("runtime_wrong_target", "CUDA_ERROR_NO_BINARY_FOR_GPU", 209),
])
def test_exact_driver_rejection_contract(mode, name, code):
    parsed = probe.validate_driver_rejection(mode, f"cuModuleLoadData: {name} ({code}): synthetic host diagnostic")
    assert parsed == {"operation": "cuModuleLoadData", "name": name, "code": code,
                      "description": "synthetic host diagnostic"}


@pytest.mark.parametrize("message", [
    "cuModuleLoadData: CUDA_ERROR_OUT_OF_MEMORY (2): synthetic host diagnostic",
    "cuModuleLoadData: CUDA_ERROR_UNKNOWN (2): synthetic host diagnostic",
    "cuLaunchKernel: CUDA_ERROR_INVALID_IMAGE (200): synthetic host diagnostic",
    "cuModuleLoadDataEx: CUDA_ERROR_INVALID_IMAGE (200): synthetic host diagnostic",
    "cuModuleLoadData: CUDA_ERROR_INVALID_IMAGE (209): synthetic host diagnostic",
    "cuModuleLoadData: CUDA_ERROR_NO_BINARY_FOR_GPU (200): synthetic host diagnostic",
    "cuModuleLoadData: CUDA_ERROR_NO_BINARY_FOR_GPU (209): synthetic host diagnostic",
    "prefix cuModuleLoadData: CUDA_ERROR_INVALID_IMAGE (200): synthetic host diagnostic",
    "cuModuleLoadData: CUDA_ERROR_INVALID_IMAGE (200): synthetic host diagnostic\nextra",
    "cuModuleLoadData: CUDA_ERROR_INVALID_IMAGE (200): ",
])
def test_unrelated_or_malformed_diagnostic_is_not_expected_rejection(message):
    with pytest.raises(AssertionError):
        probe.validate_driver_rejection("corrupt_elf", message)


@pytest.mark.parametrize("mode", ["valid", "bad_magic", "declared_target", "unknown"])
def test_header_and_host_manifest_gates_cannot_claim_driver_rejection(mode):
    with pytest.raises(AssertionError):
        probe.validate_driver_rejection(mode, "cuModuleLoadData: CUDA_ERROR_INVALID_IMAGE (200): synthetic host diagnostic")


class _SyntheticHostBuffer:
    def __init__(self):
        self.values = [-17.0] * 33
    def __getitem__(self, key):
        return SimpleNamespace(data_ptr=lambda: 1)
    def cpu(self):
        return self
    def tolist(self):
        return self.values.copy()
    def fill_(self, value):
        self.values[:] = [value] * 33


def _host_probe_spies(tmp_path, monkeypatch, *, defect=None, mode="corrupt_elf"):
    if not (OFFLINE / "sm_89.cubin").exists():
        pytest.skip("offline driver-probe images are generated locally, not stored in git")
    """Explicit synthetic CPU spy; no native or CUDA call is executed."""
    package = tmp_path / "synthetic-host-package/__init__.py"
    native_file = package.with_name("synthetic-native-placeholder.txt")
    package.parent.mkdir()
    package.write_text("synthetic host package placeholder; never imported\n")
    native_file.write_text("synthetic host native placeholder; never loaded\n")
    expected = {"package_root": str(package.parent.parent), "package": probe.fingerprint(package),
                "native": probe.fingerprint(native_file),
                "device": {"ordinal": 0, "uuid": "synthetic-host-uuid", "capability": [8, 9], "name": "synthetic host device metadata"}}
    if defect == "package_origin":
        expected["package"]["sha256"] = "0" * 64
    if defect == "native_origin":
        expected["native"]["sha256"] = "0" * 64
    if defect == "device_uuid":
        expected["device"]["uuid"] = "other-synthetic-host-uuid"
    _json(tmp_path / "expected.json", expected)
    buffer = _SyntheticHostBuffer()
    calls = []
    def constructor(blob):
        calls.append(blob)
        if len(calls) == 1:
            def launch(*args):
                buffer.values[1] = 123.0
                if defect == "positive_guard":
                    buffer.values[-1] = 123.0
            return SimpleNamespace(device=1 if defect == "native_ordinal" else 0, launch=launch, close=lambda: None)
        if defect == "fault_sentinel":
            buffer.values[-1] = 123.0
        if defect == "fault_succeeds":
            return SimpleNamespace(launch=lambda *a, **kw: None)
        raise RuntimeError("cuModuleLoadData: CUDA_ERROR_INVALID_IMAGE (200): synthetic host diagnostic")
    native = SimpleNamespace(__file__=str(native_file), DriverModule=constructor)
    stream = SimpleNamespace(cuda_stream=7, synchronize=lambda: None)
    cuda = SimpleNamespace(is_available=lambda: defect != "no_device", set_device=lambda _: None,
        current_device=lambda: 0, get_device_properties=lambda _: SimpleNamespace(uuid="synthetic-host-uuid", major=8,
        minor=9 if defect != "architecture" else 0, name="synthetic host device metadata"),
        Stream=lambda **kwargs: stream, stream=lambda _: nullcontext())
    modules = {"ragged": SimpleNamespace(__file__=str(package)), "ragged._native": native,
               "torch": SimpleNamespace(cuda=cuda, full=lambda *a, **kw: buffer, float32="synthetic-host-f32")}
    monkeypatch.setattr(probe.importlib, "import_module", lambda name: modules[name])
    monkeypatch.setattr(sys, "path", sys.path.copy())
    args = SimpleNamespace(expected=tmp_path / "expected.json", device=0, images_dir=OFFLINE, mode=mode)
    return args, {"synthetic_host_fixture": True, "scope": "CPU-only forwarding spy; no CUDA execution"}, calls, native, constructor


def test_corrupt_image_spy_preserves_genuine_full_backing_buffer_and_only_changes_elfclass(tmp_path, monkeypatch):
    args, record, calls, native, constructor = _host_probe_spies(tmp_path, monkeypatch)
    probe.probe(args, record)
    original = (OFFLINE / "sm_89.cubin").read_bytes()
    assert len(calls) == 2 and calls[0] == original
    assert len(calls[1]) == len(original) and calls[1][:4] == b"\x7fELF"
    assert [i for i, pair in enumerate(zip(original, calls[1])) if pair[0] != pair[1]] == [4]
    assert calls[1][4] == 0
    assert record["mutation"]["sha256"] == hashlib.sha256(calls[1]).hexdigest()
    assert record["fault_calls"] == {"python_constructor_calls": 1, "successful_constructors": 0, "python_launch_calls": 0}
    assert native.DriverModule is constructor
    assert record["synthetic_host_fixture"] is True


@pytest.mark.parametrize("defect,count", [
    ("package_origin", 0), ("native_origin", 0), ("no_device", 0), ("device_uuid", 0),
    ("architecture", 0), ("native_ordinal", 1), ("positive_guard", 1), ("fault_sentinel", 2), ("fault_succeeds", 2),
])
def test_origin_device_positive_control_and_sentinel_must_pass_before_fault_is_accepted(tmp_path, monkeypatch, defect, count):
    args, record, calls, native, constructor = _host_probe_spies(tmp_path, monkeypatch, defect=defect)
    with pytest.raises((AssertionError, RuntimeError)):
        probe.probe(args, record)
    assert len(calls) == count
    assert record.get("status") != "passed"
    assert native.DriverModule is constructor


@pytest.mark.parametrize("failure", ["exit", "timeout"])
def test_offline_tool_failure_is_retained_and_never_reclassified_as_driver_rejection(tmp_path, monkeypatch, failure):
    def runner(command, **kwargs):
        assert kwargs["timeout"] == 13 and kwargs["check"] is False
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, 13, output=b"synthetic partial stdout", stderr=b"synthetic partial stderr")
        return SimpleNamespace(returncode=9, stdout="synthetic stdout", stderr="synthetic stderr")
    monkeypatch.setattr(probe.subprocess, "run", runner)
    with pytest.raises(RuntimeError):
        probe._run_logged(["synthetic-host-tool"], tmp_path, "fixture", 13)
    record = json.loads((tmp_path / "fixture.exit.json").read_text())
    assert record["timed_out"] is (failure == "timeout")
    assert "synthetic" in (tmp_path / "fixture.stdout.txt").read_text()
    assert "synthetic" in (tmp_path / "fixture.stderr.txt").read_text()


def test_existing_offline_image_evidence_is_never_overwritten(tmp_path, monkeypatch):
    marker = tmp_path / "retained.txt"
    marker.write_text("retained evidence")
    monkeypatch.setattr(probe.shutil, "which", lambda _: "/synthetic/tool")
    monkeypatch.setattr(probe, "_run_logged", lambda *a, **kw: pytest.fail("must not start assembler"))
    with pytest.raises(ValueError):
        probe.build_images(tmp_path)
    assert marker.read_text() == "retained evidence"


def test_actual_collection_partition_keeps_order_union_and_exact_three_fault_groups():
    nodes = [scope.API_NODEIDS[1], MAIN, *scope.SAFE_NODEIDS, scope.API_NODEIDS[0], scope.API_NODEIDS[2]]
    groups = scope.partition_sanitizer_nodeids(nodes)
    assert groups["main"] == [MAIN]
    assert groups["native_safe"] == list(scope.SAFE_NODEIDS)
    assert groups["loader_api"] == [scope.API_NODEIDS[1], scope.API_NODEIDS[0], scope.API_NODEIDS[2]]
    sets = [set(group) for group in groups.values()]
    assert set.union(*sets) == set(nodes)
    assert all(not sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3))


def test_explicit_k_subset_may_partition_but_full_collection_cannot_omit_faults():
    nodes = [MAIN, scope.API_NODEIDS[0]]
    assert scope.partition_sanitizer_nodeids(nodes, require_all_faults=False) == {
        "main": [MAIN], "native_safe": [], "loader_api": [scope.API_NODEIDS[0]]}
    with pytest.raises(ValueError):
        scope.partition_sanitizer_nodeids(nodes)


@pytest.mark.parametrize("nodes", [
    [], [MAIN, MAIN], "test/gpu/test_correctness.py::test_x", ["test/python/test_x.py::test_x"],
    ["/test/gpu/test_x.py::test_x"], ["test/gpu/../test_x.py::test_x"], ["test/gpu/./test_x.py::test_x"],
    ["test/gpu/test_x.py::"], ["test/gpu/test_x.py::::test_x"], ["test/gpu/test_x.py::test_x\n"],
    [scope.FAULT_FUNCTION + "[unknown]"], [scope.FAULT_FILE + "::other_test"],
])
def test_partition_rejects_duplicate_noncanonical_unknown_or_non_gpu_nodes(nodes):
    with pytest.raises(ValueError):
        scope.partition_sanitizer_nodeids(nodes, require_all_faults=False)


def test_real_suppression_contains_only_exact_api_name_error_and_result_pairs():
    record = scope.validate_suppression_file(SUPPRESSION)
    assert record["sha256"] == reproduction.digest(SUPPRESSION)
    assert record["rules"] == [
        {"api": "^cuModuleLoadData$", "error": "^CUDA_ERROR_INVALID_IMAGE$", "result": "200"},
        {"api": "^cuModuleLoadData$", "error": "^CUDA_ERROR_NO_BINARY_FOR_GPU$", "result": "209"}]


@pytest.mark.parametrize("change", ["api", "name", "code", "kind", "extra", "duplicate", "stack", "root"])
def test_broadened_or_unrelated_sanitizer_suppression_is_rejected(tmp_path, change):
    tree = ET.parse(SUPPRESSION)
    root = tree.getroot()
    first = root[0]
    if change == "api": first.find("what/api").text = "cu.*"
    elif change == "name": first.find("what/error").text = "^CUDA_ERROR_.*$"
    elif change == "code": first.find("what/result").text = "999"
    elif change == "kind": first.find("kind").text = "Memory access error"
    elif change == "extra": root.append(ET.Element("record"))
    elif change == "duplicate": root[1].find("what/error").text = first.find("what/error").text; root[1].find("what/result").text = "200"
    elif change == "stack": ET.SubElement(first.find("what"), "stack").text = ".*"
    else: root.set("synthetic", "extra attribute")
    path = tmp_path / "synthetic-suppression.xml"
    tree.write(path)
    with pytest.raises(ValueError):
        scope.validate_suppression_file(path)


def _synthetic_junit(tmp_path, nodes):
    root = ET.Element("testsuites")
    suite = ET.SubElement(root, "testsuite", name="synthetic-host-fixture-no-GPU")
    for node in nodes:
        case = ET.SubElement(suite, "testcase", name=node.split("::")[-1])
        props = ET.SubElement(case, "properties")
        ET.SubElement(props, "property", name="ragged_nodeid", value=node)
    xml = tmp_path / "synthetic-host.xml"
    ET.ElementTree(root).write(xml)
    identity = tmp_path / "synthetic-host-selection.json"
    _json(identity, {"synthetic_host_fixture": True, "pytest_selection": {
        "collected": len(nodes), "nodeids": list(nodes), "deselected": []}})
    return xml, identity


def test_junit_has_to_match_the_explicit_requested_scope_not_just_self_consistent_children(tmp_path):
    xml, identity = _synthetic_junit(tmp_path, scope.API_NODEIDS)
    assert reproduction.validate_pytest_selection(xml, identity, expected_nodeids=scope.API_NODEIDS)["tests"] == 3
    with pytest.raises(reproduction.Failed):
        reproduction.validate_pytest_selection(xml, identity, expected_nodeids=scope.SAFE_NODEIDS)


@pytest.mark.parametrize("change", ["missing", "extra", "skip", "failure", "error", "duplicate", "missing_property", "duplicate_property", "count_bool", "deselected"])
def test_scope_selection_cannot_drop_duplicate_skip_or_substitute_executed_nodes(tmp_path, change):
    nodes = list(scope.SAFE_NODEIDS)
    xml, identity = _synthetic_junit(tmp_path, nodes)
    record = json.loads(identity.read_text())
    tree = ET.parse(xml)
    suite = tree.getroot()[0]
    if change == "missing":
        suite.remove(suite[-1]); record["pytest_selection"]["nodeids"].pop(); record["pytest_selection"]["collected"] = 2
    elif change == "extra":
        extra = MAIN; case = ET.SubElement(suite, "testcase", name="extra"); props = ET.SubElement(case, "properties")
        ET.SubElement(props, "property", name="ragged_nodeid", value=extra)
        record["pytest_selection"]["nodeids"].append(extra); record["pytest_selection"]["collected"] = 4
    elif change in ("skip", "failure", "error"):
        ET.SubElement(suite[0], "skipped" if change == "skip" else change)
    elif change == "duplicate": suite[1].find("properties/property").set("value", nodes[0])
    elif change == "missing_property": suite[0].remove(suite[0].find("properties"))
    elif change == "duplicate_property": ET.SubElement(suite[0].find("properties"), "property", name="ragged_nodeid", value=nodes[0])
    elif change == "count_bool": record["pytest_selection"]["collected"] = True
    else: record["pytest_selection"]["deselected"] = [MAIN]
    tree.write(xml); _json(identity, record)
    with pytest.raises(reproduction.Failed):
        reproduction.validate_pytest_selection(xml, identity, expected_nodeids=nodes)


def _standalone_fixture(tmp_path, monkeypatch, *, tools=None, selection=None, defect=None, fail_stage="collection"):
    """No children or GPUs run: only the real parent validators read synthetic files."""
    args = standalone.parser().parse_args(["--output", str(tmp_path / "synthetic-run"), "--stage-timeout-seconds", "13"])
    args.tools = tools or ["plain", "memcheck", "racecheck", "synccheck"]
    args.k = selection
    fake_torch_file = tmp_path / "synthetic-torch.txt"
    fake_torch_file.write_text("synthetic CPU host metadata; no Torch/CUDA execution\n")
    fake_torch = SimpleNamespace(__version__="synthetic-host-fixture", __file__=str(fake_torch_file),
        cuda=SimpleNamespace(is_available=lambda: True, get_device_capability=lambda: (8, 9)))
    monkeypatch.setattr(standalone.importlib, "import_module", lambda name: fake_torch)
    sanitizer = tmp_path / "synthetic-compute-sanitizer.txt"
    sanitizer.write_text("synthetic non-executable host placeholder\n")
    monkeypatch.setattr(standalone.shutil, "which", lambda _: str(sanitizer))
    nodes = [MAIN, *scope.FAULT_NODEIDS] if selection is None else [MAIN, scope.API_NODEIDS[0]]
    calls = []
    def runner(command, **kwargs):
        name = Path(kwargs["stdout"].name).stem
        calls.append((name, command, kwargs))
        kwargs["stdout"].write("synthetic host child fixture; no subprocess or GPU execution\n")
        if name == fail_stage and defect in ("exit", "timeout", "interrupt"):
            if defect == "timeout": raise subprocess.TimeoutExpired(command, kwargs["timeout"])
            if defect == "interrupt": raise KeyboardInterrupt("synthetic host interrupt")
            return SimpleNamespace(returncode=86)
        if name == "compute-sanitizer-version":
            return SimpleNamespace(returncode=0)
        record_nodes = nodes if name == "collection" else [node for node in command if node.startswith("test/gpu/")]
        record = {"synthetic_host_fixture": True,
            "pytest_selection": {"collected": len(record_nodes), "nodeids": record_nodes.copy(),
                "deselected": [scope.SAFE_NODEIDS[0]] if name == "collection" and selection else []},
            "collector": {"file": str(ROOT / "scripts/gpu_validation_scope.py"),
                "sha256": reproduction.digest(ROOT / "scripts/gpu_validation_scope.py")},
            "python": {"executable": str(Path(sys.executable).absolute()), "version": sys.version},
            "rootdir": str(ROOT), "cwd": str(ROOT)}
        identity = Path(kwargs["env"][scope.SELECTION_ENV])
        if name == fail_stage and defect == "producer": record["collector"]["sha256"] = "0" * 64
        if name == fail_stage and defect == "substitute":
            record["pytest_selection"]["nodeids"] = [MAIN]
            record["pytest_selection"]["collected"] = 1
        if name == fail_stage and defect == "duplicate":
            record["pytest_selection"]["nodeids"] *= 2
            record["pytest_selection"]["collected"] *= 2
        if not (name == fail_stage and defect == "missing_identity"):
            _json(identity, record)
        if name != "collection":
            xml = Path(next(arg.split("=", 1)[1] for arg in command if arg.startswith("--junitxml=")))
            fixture_xml, _ = _synthetic_junit(tmp_path, record["pytest_selection"]["nodeids"])
            tree = ET.parse(fixture_xml)
            if name == fail_stage and defect == "skip": ET.SubElement(tree.getroot()[0][0], "skipped")
            if not (name == fail_stage and defect == "missing_xml"): tree.write(xml)
        return SimpleNamespace(returncode=0)
    preflight = lambda **kw: {"status": "ready", "device": {"capability": [8, 9], "uuid": "synthetic-host-uuid"},
                              "synthetic_host_fixture": True}
    code = standalone.run(args, preflight=preflight, runner=runner)
    return code, json.loads((args.output / "results.json").read_text()), calls, args.output


def test_standalone_no_device_never_imports_torch_collects_or_launches_sanitizer(tmp_path, monkeypatch):
    args = standalone.parser().parse_args(["--output", str(tmp_path / "blocked")])
    monkeypatch.setattr(standalone.importlib, "import_module", lambda _: pytest.fail("preflight must precede Torch import"))
    code = standalone.run(args, preflight=lambda **kw: {"status": "blocked", "reason": "synthetic host NO_DEVICE fixture"},
        runner=lambda *a, **kw: pytest.fail("must not launch a child"))
    report = json.loads((args.output / "results.json").read_text())
    assert code == 2 and report["status"] == "blocked"
    assert report["runs"] == report["stages"] == []


def test_standalone_synthetic_plan_covers_union_once_per_tool_and_suppresses_only_loader_api(tmp_path, monkeypatch):
    code, report, calls, output = _standalone_fixture(tmp_path, monkeypatch)
    assert code == 0 and report["status"] == "passed"
    assert len(report["runs"]) == 12
    temps = []
    for name, command, _ in calls:
        selected = [arg.split("=", 1)[1] for arg in command if arg.startswith("--basetemp=")]
        if name in ("collection", "compute-sanitizer-version"):
            assert selected == []
        else:
            assert selected == [str(output / "work" / name)]
            assert Path(selected[0]).resolve().is_relative_to(output.resolve())
            temps += selected
    assert len(temps) == len(set(temps)) == 12
    for tool in ("plain", "memcheck", "racecheck", "synccheck"):
        execution = [entry for entry in calls if entry[0].startswith(tool + "-")]
        seen = []
        for name, command, options in execution:
            assert options["timeout"] == 13 and options["check"] is False
            seen.extend(node for node in command if node.startswith("test/gpu/"))
            assert "-k" not in command and "--collect-only" not in command
            if tool != "plain":
                assert command[command.index("--target-processes") + 1] == "all"
                assert command[command.index("--error-exitcode") + 1] == "86"
            if tool != "plain" and name.endswith("loader_api"):
                assert command.count("--suppressions") == 1
                assert command[command.index("--suppressions") + 1] == str(SUPPRESSION)
            else:
                assert not any("suppress" in arg for arg in command)
        assert len(seen) == len(set(seen)) == 7
        assert set(seen) == {MAIN, *scope.FAULT_NODEIDS}


def test_standalone_k_applies_only_to_collection_before_partition(tmp_path, monkeypatch):
    code, report, calls, _ = _standalone_fixture(tmp_path, monkeypatch, tools=["memcheck"], selection="synthetic-selection-expression")
    assert code == 0 and report["groups"]["native_safe"] == []
    collection = next(command for name, command, _ in calls if name == "collection")
    assert collection[collection.index("-k") + 1] == "synthetic-selection-expression"
    assert "--collect-only" in collection
    assert [run["group"] for run in report["runs"]] == ["main", "loader_api"]
    assert all("-k" not in command for name, command, _ in calls if name != "collection")


@pytest.mark.parametrize("stage", ["collection", "memcheck-native_safe", "memcheck-loader_api"])
@pytest.mark.parametrize("defect", ["exit", "timeout", "interrupt", "producer", "missing_identity", "duplicate"])
def test_standalone_stage_failure_stops_without_retry_or_pass_label(tmp_path, monkeypatch, stage, defect):
    code, report, calls, output = _standalone_fixture(tmp_path, monkeypatch, tools=["memcheck"], defect=defect, fail_stage=stage)
    assert code == 1 and report["status"] == "failed"
    assert [name for name, _, _ in calls].count(stage) == 1
    assert calls[-1][0] == stage
    failed = report["stages"][-1]
    assert failed["name"] == stage and failed["status"] == "failed"
    assert failed["log_sha256"] == reproduction.digest(output / failed["log"])
    assert bool(failed.get("timed_out")) is (defect == "timeout")
    assert bool(report.get("interrupted")) is (defect == "interrupt")


@pytest.mark.parametrize("defect", ["substitute", "skip", "missing_xml"])
def test_standalone_api_group_requires_exact_requested_nodes_with_zero_skips(tmp_path, monkeypatch, defect):
    code, report, calls, _ = _standalone_fixture(tmp_path, monkeypatch, tools=["memcheck"], defect=defect, fail_stage="memcheck-loader_api")
    assert code == 1 and report["status"] == "failed" and calls[-1][0] == "memcheck-loader_api"


def test_standalone_existing_output_is_rejected_before_preflight(tmp_path):
    output = tmp_path / "retained"
    output.mkdir()
    marker = output / "retained.txt"
    marker.write_text("retained evidence")
    args = standalone.parser().parse_args(["--output", str(output)])
    assert standalone.run(args, preflight=lambda **kw: pytest.fail("existing output must stop before preflight")) == 1
    assert marker.read_text() == "retained evidence"


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_standalone_invalid_timeout_rejects_without_creating_output(tmp_path, timeout):
    args = standalone.parser().parse_args(["--output", str(tmp_path / "out")])
    args.stage_timeout_seconds = timeout
    with pytest.raises(ValueError): standalone.run(args)
    assert not args.output.exists()


def _fresh_wheel_scope_fixture(tmp_path, monkeypatch, *, defect=None):
    """Reuse only synthetic artifact setup, never execute the old test suite."""
    spec = importlib.util.spec_from_file_location("independent_reproduction_host_fixtures", ROOT / "test/python/test_reproduce_gpu_contract.py")
    fixtures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixtures)
    original_source = fixtures._minimal_source
    def source(path):
        checkout = original_source(path)
        suppression = checkout / "test/gpu/driver-load-api-suppressions.xml"
        suppression.parent.mkdir()
        suppression.write_bytes(SUPPRESSION.read_bytes())
        return checkout
    monkeypatch.setattr(fixtures, "_minimal_source", source)
    original_run = reproduction.run
    calls = []
    def run(args, *, preflight, runner):
        def scoped_runner(command, **kwargs):
            name = Path(kwargs["stdout"].name).name.split("-", 1)[1].removesuffix(".log")
            calls.append((name, list(command), kwargs))
            if not name.startswith("sanitizer-"):
                return runner(command, **kwargs)
            kwargs["stdout"].write("synthetic host scope fixture; no installed wheel/GPU execution\n")
            if name == "sanitizer-collection":
                nodes = [MAIN, *scope.FAULT_NODEIDS]
                if defect == "incomplete_collection": nodes.pop()
            else:
                nodes = [arg for arg in command if arg.startswith("test/gpu/")]
            identity_path = Path(command[command.index("--identity-output") + 1])
            uuid = command[command.index("--expected-uuid") + 1]
            identity = fixtures._synthetic_identity(args.output, args.output / "source", uuid)
            identity["pytest_selection"] = {"collected": len(nodes), "nodeids": nodes, "deselected": []}
            if name == "sanitizer-memcheck-loader_api" and defect == "native_origin":
                identity["modules"]["ragged._native"]["sha256"] = "0" * 64
            if name != "sanitizer-collection":
                fixture_xml, _ = _synthetic_junit(tmp_path, nodes)
                tree = ET.parse(fixture_xml)
                if name == "sanitizer-memcheck-loader_api" and defect == "skip":
                    ET.SubElement(tree.getroot()[0][0], "skipped")
                xml = Path(next(arg.split("=", 1)[1] for arg in command if arg.startswith("--junitxml=")))
                tree.write(xml)
            _json(identity_path, identity)
            return SimpleNamespace(returncode=0)
        return original_run(args, preflight=preflight, runner=scoped_runner)
    monkeypatch.setattr(reproduction, "run", run)
    output, report, _, code = fixtures._pipeline(tmp_path, monkeypatch, sanitizers=True)
    return output, report, calls, code


def test_fresh_wheel_sanitizer_scopes_keep_identity_worker_and_explicit_optional_exclusion(tmp_path, monkeypatch):
    output, report, calls, code = _fresh_wheel_scope_fixture(tmp_path, monkeypatch)
    assert code == 0 and report["status"] == "passed"
    collection = next(command for name, command, _ in calls if name == "sanitizer-collection")
    assert "--collect-only" in collection and "--ignore=test/gpu/test_fbgemm_baseline.py" in collection
    assert not any(arg.startswith("--basetemp=") for arg in collection)
    correctness = next(command for name, command, _ in calls if name == "gpu-correctness")
    assert f"--basetemp={output / 'work/gpu-correctness'}" in correctness
    temps = [str(output / "work/gpu-correctness")]
    for tool in ("memcheck", "racecheck", "synccheck"):
        seen = []
        for name, command, options in calls:
            if not name.startswith("sanitizer-" + tool + "-"): continue
            seen.extend(arg for arg in command if arg.startswith("test/gpu/"))
            assert str(output / "venv/bin/python") in command and "-I" in command
            assert str(output / "source/scripts/reproduce_gpu_worker.py") in command
            assert "--identity-output" in command and "--expected-uuid" in command
            assert "PYTHONPATH" not in options["env"]
            assert command[command.index("--error-exitcode") + 1] == "86"
            selected = [arg.split("=", 1)[1] for arg in command if arg.startswith("--basetemp=")]
            assert selected == [str(output / "work" / name.removeprefix("sanitizer-"))]
            assert Path(selected[0]).resolve().is_relative_to(output.resolve())
            temps += selected
            if name.endswith("loader_api"):
                assert command.count("--suppressions") == 1
                assert command[command.index("--suppressions") + 1] == str(output / "source/test/gpu/driver-load-api-suppressions.xml")
            else:
                assert not any("suppress" in arg for arg in command)
        assert len(seen) == len(set(seen)) == 7 and set(seen) == {MAIN, *scope.FAULT_NODEIDS}
    assert len(temps) == len(set(temps)) == 10


@pytest.mark.parametrize("defect", ["incomplete_collection", "native_origin", "skip"])
def test_fresh_wheel_new_scopes_cannot_weaken_collection_native_origin_or_no_skip_gates(tmp_path, monkeypatch, defect):
    _, report, calls, code = _fresh_wheel_scope_fixture(tmp_path, monkeypatch, defect=defect)
    assert code == 1 and report["status"] == "failed" and report["device_execution_passed"] is False
    expected = "sanitizer-collection" if defect == "incomplete_collection" else "sanitizer-memcheck-loader_api"
    assert report["stages"][-1]["name"] == calls[-1][0] == expected
    assert report["stages"][-1]["status"] == "failed"
