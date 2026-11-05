"""Isolated real Driver API fault probe; --build-dir is CPU-only ptxas setup.

This support module has no simulated CUDA path. Python call counters record
calls made here, not hooks into cuLaunchKernel. A malformed-image crash or
timeout is a failed test in the parent, never an expected rejection.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import importlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import traceback


MODES = ("valid", "bad_magic", "corrupt_elf", "wrong_target", "declared_target", "runtime_wrong_target")
EXPECTED_DRIVER_ERRORS = {
    # Driver 595.99 reports an ELF whose class byte is ELFCLASSNONE as
    # CUDA_ERROR_UNKNOWN rather than CUDA_ERROR_INVALID_IMAGE.
    "corrupt_elf": {("CUDA_ERROR_INVALID_IMAGE", 200), ("CUDA_ERROR_UNKNOWN", 999)},
    "wrong_target": {("CUDA_ERROR_NO_BINARY_FOR_GPU", 209)},
    "runtime_wrong_target": {("CUDA_ERROR_NO_BINARY_FOR_GPU", 209)},
}


def validate_driver_rejection(mode: str, message: str) -> dict:
    """Require the exact image/target error, not an unrelated driver failure.

    Names and values follow the CUDA header's CUresult definitions. Actual
    devices must confirm these expected branches; OOM, a different operation,
    or an unexpected code always fails this probe.
    """
    match = re.fullmatch(r"(cu[A-Za-z0-9_]+): (CUDA_ERROR_[A-Z_]+) \(([0-9]+)\): ([^\r\n]+)", message)
    if match is None:
        raise AssertionError(f"unexpected native driver diagnostic: {message!r}")
    operation, name, code, description = match.groups()
    parsed = {"operation": operation, "name": name, "code": int(code), "description": description}
    expected = EXPECTED_DRIVER_ERRORS.get(mode)
    if expected is None or operation != "cuModuleLoadData" or (name, int(code)) not in expected:
        raise AssertionError(f"unexpected driver rejection for {mode}: {parsed}; expected {expected}")
    return parsed


def _run_logged(command, directory: Path, stem: str, timeout: int):
    """Bound offline tools and preserve diagnostics even when they time out."""
    (directory / f"{stem}.command.json").write_text(json.dumps(command, indent=2) + "\n")
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        for suffix, value in (("stdout", exc.stdout), ("stderr", exc.stderr)):
            payload = value.encode() if isinstance(value, str) else value or b""
            (directory / f"{stem}.{suffix}.txt").write_bytes(payload)
        (directory / f"{stem}.exit.json").write_text(json.dumps({"timed_out": True, "timeout_seconds": timeout}) + "\n")
        raise RuntimeError(f"ptxas {stem} timed out after {timeout}s; diagnostics: {directory}") from exc
    (directory / f"{stem}.stdout.txt").write_text(result.stdout)
    (directory / f"{stem}.stderr.txt").write_text(result.stderr)
    (directory / f"{stem}.exit.json").write_text(json.dumps({"exit_code": result.returncode, "timed_out": False}) + "\n")
    if result.returncode:
        raise RuntimeError(f"ptxas {stem} failed ({result.returncode}): {result.stderr}")
    return result


def fingerprint(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def build_images(directory: Path):
    """Assemble genuine target-specific ELF images; does not initialize CUDA."""
    executable = shutil.which("ptxas")
    if executable is None:
        raise RuntimeError("Driver fault tests require the actual ptxas assembler on PATH")
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("image directory must be absent or empty; do not overwrite evidence")
    directory.mkdir(parents=True, exist_ok=True)
    version = _run_logged([executable, "--version"], directory, "ptxas-version", timeout=15)
    report = {"scope": "CPU-only ptxas fixture assembly; no module load or GPU execution",
              "ptxas": fingerprint(executable), "ptxas_version": version.stdout.strip(), "images": {}}
    for target in ("sm_89", "sm_90"):
        source, binary = directory / f"{target}.ptx", directory / f"{target}.cubin"
        source.write_text(f""".version 8.0
.target {target}
.address_size 64
.visible .entry ragged_driver_probe(.param .u64 output) {{
  .reg .b64 %pointer;
  ld.param.u64 %pointer, [output];
  st.global.u32 [%pointer], 0x42f60000;
  ret;
}}
""")
        command = [executable, str(source), f"-arch={target}", "-v", "-o", str(binary)]
        result = _run_logged(command, directory, target, timeout=60)
        if not binary.read_bytes().startswith(b"\x7fELF"):
            raise RuntimeError("ptxas fixture did not produce ELF cubin")
        report["images"][target] = {"command": command, "exit_code": result.returncode,
                                    "source": fingerprint(source), "binary": fingerprint(binary)}
    (directory / "images.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def probe(args, report):
    # -I removes PYTHONPATH and cwd imports. Explicitly choose the parent's
    # actual package root, including an installed wheel, then verify both files.
    expected = json.loads(args.expected.read_text())
    sys.path.insert(0, expected["package_root"])
    ragged = importlib.import_module("ragged")
    native = importlib.import_module("ragged._native")
    origins = {"package": fingerprint(ragged.__file__), "native": fingerprint(native.__file__)}
    if origins != {name: expected[name] for name in ("package", "native")}:
        raise AssertionError("child package/native origin differs from the parent test environment")
    report["origins"] = origins
    torch = importlib.import_module("torch")
    if not torch.cuda.is_available():
        raise RuntimeError("parent preflight passed but child has no CUDA device")
    torch.cuda.set_device(args.device)
    props = torch.cuda.get_device_properties(args.device)
    device = {"ordinal": torch.cuda.current_device(), "uuid": str(props.uuid),
              "capability": [props.major, props.minor], "name": props.name}
    report["device"] = device
    if device != expected["device"] or device["capability"] != [8, 9]:
        raise AssertionError("child CUDA device differs from parent preflight")
    manifest = json.loads((args.images_dir / "images.json").read_text())
    blobs = {}
    for target in ("sm_89", "sm_90"):
        path = args.images_dir / f"{target}.cubin"
        if fingerprint(path) != manifest["images"][target]["binary"]:
            raise AssertionError(f"{target} cubin changed after fixture assembly")
        blobs[target] = path.read_bytes()
    report["images"] = manifest
    stream = torch.cuda.Stream(device=args.device)
    with torch.cuda.stream(stream):
        guard = torch.full((33,), -17.0, device=args.device, dtype=torch.float32)
    # Positive control in every isolated process proves context, loading and
    # real launch work before a fault is interpreted as a successful rejection.
    real_module = native.DriverModule
    control = real_module(blobs["sm_89"])
    if control.device != args.device:
        raise AssertionError("native module reports a different CUDA device ordinal")
    control.launch("ragged_driver_probe", [1, 1, 1], [1, 1, 1], 0,
                   stream.cuda_stream, [guard[1:].data_ptr()])
    stream.synchronize()
    actual = guard.cpu().tolist()
    if actual != [-17.0, 123.0, *([-17.0] * 31)]:
        raise AssertionError("sm_89 positive control output or guard differs")
    control.close()
    report["positive_control"] = {"python_constructor_calls": 1, "python_launch_calls": 1,
                                   "written_value": actual[1], "guard_intact": True}
    if args.mode == "valid":
        report.update(status="passed", branch="native_positive_control")
        return
    with torch.cuda.stream(stream):
        guard.fill_(-17.0)
    stream.synchronize()
    calls = {"python_constructor_calls": 0, "successful_constructors": 0, "python_launch_calls": 0}
    report["fault_calls"] = calls

    def observed_module(blob, *args, **kwargs):
        calls["python_constructor_calls"] += 1
        module = real_module(blob, *args, **kwargs)
        calls["successful_constructors"] += 1
        class Observed:
            def launch(self, *launch_args, **launch_kwargs):
                calls["python_launch_calls"] += 1
                return module.launch(*launch_args, **launch_kwargs)
        return Observed()

    # Only count Python calls while forwarding all actual loading and launches
    # unchanged to the real extension. No synthetic GPU result is supplied.
    native.DriverModule = observed_module
    try:
        if args.mode in ("declared_target", "runtime_wrong_target"):
            from ragged import ArtifactCache, CacheKey, CompiledProgram, ProgramSpec
            cache_manifest = json.loads((args.runtime_artifact / "manifest.json").read_text())
            key = CacheKey(**cache_manifest["key"])
            artifact = ArtifactCache(args.runtime_artifact.parent.parent).load(key)
            spec = ProgramSpec.from_dict(json.loads(args.spec.read_text()))
            report["runtime_artifact"] = {"cache_key": key.digest,
                "original_cubin_sha256": hashlib.sha256(artifact.cubin).hexdigest(),
                "metadata_declared_chip": artifact.metadata["chip"]}
            if args.mode == "declared_target":
                metadata = dict(artifact.metadata, chip="sm_90")
                CompiledProgram(spec, replace(artifact, metadata=metadata))
            else:
                # Inject after verified cache load: this is an in-memory target
                # mismatch, not a test that edits a checksum-protected cache.
                program = CompiledProgram(spec, replace(artifact, cubin=blobs["sm_90"]))
                with torch.cuda.stream(stream):
                    values = torch.tensor([[1.0], [2.0]], device=args.device)
                    offsets = torch.tensor([0, 2], device=args.device, dtype=torch.int64)
                program.run(values, offsets, guard[1:2].view(1, 1), stream=stream)
        else:
            blob = blobs["sm_90"] if args.mode == "wrong_target" else b"not-an-ELF"
            if args.mode == "corrupt_elf":
                damaged = bytearray(blobs["sm_89"])
                damaged[4] = 0  # ELFCLASSNONE; preserve magic and full backing length.
                blob = bytes(damaged)
                report["mutation"] = {"byte_offset": 4, "old": blobs["sm_89"][4], "new": 0,
                                      "size_preserved": len(blob), "sha256": hashlib.sha256(blob).hexdigest()}
            observed_module(blob)
    except RuntimeError as exc:
        message = str(exc)
        report["exception"] = {"type": type(exc).__name__, "message": message}
    else:
        raise AssertionError("fault unexpectedly succeeded; no rejection was observed")
    finally:
        native.DriverModule = real_module
    stream.synchronize()
    if guard.cpu().tolist() != [-17.0] * 33:
        raise AssertionError("fault path changed the output sentinel")
    report["sentinel_unchanged"] = True
    if calls["successful_constructors"] != 0 or calls["python_launch_calls"] != 0:
        raise AssertionError("fault returned a module or submitted a Python launch")
    message = report["exception"]["message"]
    if args.mode == "declared_target":
        if "chip differs" not in message or calls["python_constructor_calls"] != 0:
            raise AssertionError("declared target did not reject at the host manifest gate")
        report["branch"] = "host_manifest_gate_not_driver"
    elif args.mode == "bad_magic":
        if "expected a compiled ELF cubin" not in message or calls["python_constructor_calls"] != 1:
            raise AssertionError("non-ELF did not reject at the native header gate")
        report["branch"] = "native_header_gate_not_module_load"
    else:
        if calls["python_constructor_calls"] != 1:
            raise AssertionError("fault did not reach a real cuModuleLoadData rejection")
        report["driver_error"] = validate_driver_rejection(args.mode, message)
        report["branch"] = "driver_module_load"
    report["status"] = "passed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--device", type=int)
    parser.add_argument("--expected", type=Path)
    parser.add_argument("--images-dir", type=Path)
    parser.add_argument("--runtime-artifact", type=Path)
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    if args.build_dir:
        report = build_images(args.build_dir.resolve())
        print(json.dumps({"status": "assembled", "scope": report["scope"], "images": report["images"]}, indent=2))
        return 0
    if any(value is None for value in (args.mode, args.device, args.expected, args.images_dir, args.result)):
        parser.error("worker mode requires --mode --device --expected --images-dir --result")
    report = {"scope": "real isolated CUDA Driver fault probe", "status": "running",
              "mode": args.mode, "command": sys.argv, "python": sys.executable}
    try:
        probe(args, report)
        code = 0
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        code = 1
    args.result.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "mode": args.mode, "result": str(args.result)}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
