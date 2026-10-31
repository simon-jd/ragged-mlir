"""Real sm_89 Driver load faults; missing CUDA is skip, never a pass.

Every fault has a successful real load/launch control in its own bounded child
process. Crashes, timeouts, missing artifacts and unexpected exception stages
fail. The declared-target case is explicitly a host manifest gate, whereas
damaged ELF and genuine sm_90 cubin cases must reach cuModuleLoadData.
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from driver_fault_probe import MODES, build_images, fingerprint, validate_driver_rejection
from test_correctness import gpu_environment


@pytest.fixture(scope="session")
def driver_fault_setup(gpu_environment, tmp_path_factory):
    torch, compile_spec, _, _ = gpu_environment
    import ragged
    from ragged import Hints, ProgramSpec
    native = importlib.import_module("ragged._native")
    root = tmp_path_factory.mktemp("real-driver-faults")
    images = root / "images"
    build_images(images)
    ordinal = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(ordinal)
    expected = {"package_root": str(Path(ragged.__file__).resolve().parent.parent),
                "package": fingerprint(ragged.__file__), "native": fingerprint(native.__file__),
                "device": {"ordinal": ordinal, "uuid": str(props.uuid), "name": props.name,
                           "capability": [props.major, props.minor]}}
    expected_path = root / "expected-environment.json"
    expected_path.write_text(json.dumps(expected, indent=2) + "\n")
    spec = ProgramSpec("sum", feature_dim=1, hints=Hints(max_len=32, offsets_checked=True))
    program = compile_spec(spec)
    spec_path = root / "spec.json"
    spec_path.write_text(json.dumps(spec.to_dict(), indent=2) + "\n")
    return images, ordinal, expected_path, expected, program.artifact.path, spec_path


@pytest.mark.parametrize("mode", [
    pytest.param(mode, marks=pytest.mark.cuda_loader_api_fault)
    if mode in ("corrupt_elf", "wrong_target", "runtime_wrong_target") else mode
    for mode in MODES
])
def test_real_driver_faults_in_isolated_process(driver_fault_setup, tmp_path, record_property, mode):
    images, ordinal, expected_path, expected, artifact_path, spec_path = driver_fault_setup
    probe = Path(__file__).with_name("driver_fault_probe.py").resolve()
    output = tmp_path / "probe.json"
    command = [sys.executable, "-I", str(probe), "--mode", mode, "--device", str(ordinal),
               "--expected", str(expected_path), "--images-dir", str(images),
               "--runtime-artifact", str(artifact_path), "--spec", str(spec_path), "--result", str(output)]
    (tmp_path / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=90, check=False)
    except subprocess.TimeoutExpired as exc:
        (tmp_path / "stdout.txt").write_bytes(exc.stdout or b"")
        (tmp_path / "stderr.txt").write_bytes(exc.stderr or b"")
        pytest.fail(f"Driver fault probe timed out; this is a test failure, not rejection evidence: {tmp_path}")
    (tmp_path / "stdout.txt").write_text(result.stdout)
    (tmp_path / "stderr.txt").write_text(result.stderr)
    (tmp_path / "exit-code.txt").write_text(str(result.returncode) + "\n")
    record_property("driver_fault_evidence", str(tmp_path))
    assert result.returncode == 0, f"Driver fault probe failed/crashed ({result.returncode}): {result.stderr}\n{result.stdout}\n{tmp_path}"
    assert output.is_file(), "successful child must publish its real probe result"
    evidence = json.loads(output.read_text())
    assert evidence["status"] == "passed" and evidence["mode"] == mode
    assert evidence["origins"] == {name: expected[name] for name in ("package", "native")}
    assert evidence["device"] == expected["device"]
    assert evidence["positive_control"] == {"python_constructor_calls": 1, "python_launch_calls": 1,
                                             "written_value": 123.0, "guard_intact": True}
    if mode == "valid":
        assert evidence["branch"] == "native_positive_control"
    else:
        assert evidence["sentinel_unchanged"] is True
        assert evidence["fault_calls"]["python_launch_calls"] == 0
        assert evidence["fault_calls"]["successful_constructors"] == 0
        if mode == "declared_target":
            assert evidence["branch"] == "host_manifest_gate_not_driver"
        elif mode == "bad_magic":
            assert evidence["branch"] == "native_header_gate_not_module_load"
        else:
            assert evidence["branch"] == "driver_module_load"
            assert evidence["fault_calls"]["python_constructor_calls"] == 1
            assert evidence["driver_error"] == validate_driver_rejection(mode, evidence["exception"]["message"])
