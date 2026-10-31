#!/usr/bin/env python3
"""Run real sm_89 tests with exact sanitizer scopes and retained evidence.

No device is blocked (exit 2), never sanitizer success. A -k selection is applied
only to actual collection. Each selected node then runs once per requested tool:
main and native_safe have no suppression; loader_api alone uses two exact API
rules. All device checks and unrelated CUDA API diagnostics remain enabled.
"""
from __future__ import annotations

import argparse
import importlib
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

try:
    from scripts.gpu_validation_scope import (SELECTION_ENV, collected_nodeids,
                                              partition_sanitizer_nodeids, validate_suppression_file)
    from scripts.reproduce_gpu import (Failed, bounded_run, digest, driver_preflight,
                                       read_json, validate_pytest_selection, write_json)
except ModuleNotFoundError as error:
    if error.name != "scripts":
        raise
    from gpu_validation_scope import (SELECTION_ENV, collected_nodeids,
                                      partition_sanitizer_nodeids, validate_suppression_file)
    from reproduce_gpu import (Failed, bounded_run, digest, driver_preflight,
                               read_json, validate_pytest_selection, write_json)

ROOT = Path(__file__).resolve().parents[1]
LOGGED_ENV = ("PATH", "PYTHONPATH", "PYTEST_DISABLE_PLUGIN_AUTOLOAD", SELECTION_ENV,
              "CUDA_VISIBLE_DEVICES", "CUDA_ROOT", "CUDA_HOME", "CUDA_PATH", "RAGGED_COMPILE")


def junit_counts(path: Path) -> dict[str, int]:
    """Return raw JUnit statistics; execution gates use validate_pytest_selection."""
    tree = ET.parse(path)
    cases = tree.findall(".//testcase")
    return {"tests": len(cases), "skipped": sum(c.find("skipped") is not None for c in cases),
            "failures": sum(c.find("failure") is not None for c in cases),
            "errors": sum(c.find("error") is not None for c in cases)}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--tools", nargs="+", choices=("plain", "memcheck", "racecheck", "synccheck"),
                        default=["plain", "memcheck", "racecheck", "synccheck"])
    result.add_argument("-k", default=None, help="pytest expression applied only during actual collection")
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--stage-timeout-seconds", type=float, default=14400)
    return result


def run(args, *, preflight=driver_preflight, runner=bounded_run):
    if (not args.tools or len(set(args.tools)) != len(args.tools)
            or any(tool not in ("plain", "memcheck", "racecheck", "synccheck") for tool in args.tools)
            or not math.isfinite(args.stage_timeout_seconds) or args.stage_timeout_seconds <= 0):
        raise ValueError("tools must be nonempty/unique and timeout finite/positive")
    requested = args.output.absolute()
    output = requested.resolve()
    if output.exists() or requested.is_symlink():
        print("failed: output must not exist", file=sys.stderr)
        return 1
    output.mkdir(parents=True)
    report = {"schema": 2, "timestamp_unix_ns": time.time_ns(), "status": "running",
              "selection": args.k, "rootdir": str(ROOT), "runs": [], "stages": [],
              "scope": "real selected sm_89 tests; each node exactly once per tool; API suppression only in loader_api"}

    def save():
        write_json(output / "results.json", report)

    save()
    try:
        # A stdlib Driver gate runs before Torch import or any child process.
        device = preflight(device=0)
        report["driver"] = device
        if device.get("status") != "ready":
            report.update(status="blocked", exit_code=2, reason=device.get("reason", "CUDA Driver is not ready"))
            save()
            return 2
        try:
            torch = importlib.import_module("torch")
            if not torch.cuda.is_available() or tuple(torch.cuda.get_device_capability()) != (8, 9):
                raise RuntimeError("the invoking environment must provide CUDA PyTorch on sm_89")
        except (ImportError, RuntimeError) as error:
            report.update(status="blocked", exit_code=2, reason=str(error))
            save()
            return 2
        report["torch"] = {"version": torch.__version__, "file": str(Path(torch.__file__).resolve()),
                            "sha256": digest(Path(torch.__file__))}
        report["python"] = {"executable": str(Path(sys.executable).absolute()), "version": sys.version}
        sanitizer = shutil.which("compute-sanitizer")
        if any(tool != "plain" for tool in args.tools) and not sanitizer:
            report.update(status="blocked", exit_code=2, reason="compute-sanitizer is not on PATH")
            save()
            return 2
        suppression = validate_suppression_file(ROOT / "test/gpu/driver-load-api-suppressions.xml")
        report["suppression"] = suppression
        source_files = [*sorted((ROOT / "test/gpu").glob("*.py")), ROOT / "pyproject.toml",
                        ROOT / "scripts/run_gpu_validation.py", ROOT / "scripts/gpu_validation_scope.py",
                        ROOT / "scripts/reproduce_gpu.py", Path(suppression["path"])]
        report["source_sha256"] = {str(path.relative_to(ROOT)): digest(path) for path in source_files}
        env = os.environ.copy()
        for key in tuple(env):
            if key.startswith("PYTEST_") or key in ("PYTHONPATH", "PYTHONHOME", SELECTION_ENV):
                del env[key]
        env.update(PYTHONPATH=os.pathsep.join((str(ROOT / "python"), str(ROOT))), PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")

        def stage(name, command, *, selection=None, validator=None):
            local_env = env.copy()
            if selection is not None:
                local_env[SELECTION_ENV] = str(selection)
            log = output / (name + ".log")
            item = {"name": name, "argv": list(map(str, command)), "cwd": str(ROOT),
                    "env_overrides": {key: local_env[key] for key in LOGGED_ENV if key in local_env},
                    "timeout_seconds": args.stage_timeout_seconds, "log": log.name, "status": "running"}
            report["stages"].append(item)
            save()
            started = time.monotonic()
            try:
                with log.open("w") as stream:
                    result = runner(item["argv"], cwd=str(ROOT), env=local_env, stdout=stream,
                                    stderr=subprocess.STDOUT, check=False, timeout=args.stage_timeout_seconds)
                item["exit_code"] = result.returncode
                if result.returncode:
                    raise Failed(f"{name} exited {result.returncode}; see {log}")
                if validator:
                    item["validation"] = validator()
                item["status"] = "passed"
                return item
            except (Exception, KeyboardInterrupt) as error:
                item.update(status="failed", error=f"{type(error).__name__}: {error}")
                item["exit_code"] = getattr(error, "returncode", item.get("exit_code"))
                if isinstance(error, subprocess.TimeoutExpired):
                    item["timed_out"] = True
                if isinstance(error, KeyboardInterrupt):
                    item["interrupted"] = True
                raise
            finally:
                item["wall_seconds"] = time.monotonic() - started
                if log.is_file():
                    item["log_sha256"] = digest(log)
                if selection is not None and selection.is_file():
                    item["selection"] = {"path": selection.name, "sha256": digest(selection)}
                save()

        if sanitizer and any(tool != "plain" for tool in args.tools):
            sanitizer = str(Path(sanitizer).resolve())
            report["compute_sanitizer"] = {"file": sanitizer, "sha256": digest(Path(sanitizer))}
            stage("compute-sanitizer-version", [sanitizer, "--version"])
            report["compute_sanitizer"]["version"] = (output / "compute-sanitizer-version.log").read_text().strip()

        def check_collection(path, *, allow_deselected=False):
            record = read_json(path)
            expected_collector = {"file": str(ROOT / "scripts/gpu_validation_scope.py"),
                                  "sha256": report["source_sha256"]["scripts/gpu_validation_scope.py"]}
            if (record.get("collector") != expected_collector or record.get("rootdir") != str(ROOT)
                    or record.get("cwd") != str(ROOT) or record.get("python") != report["python"]):
                raise Failed("pytest collection producer/root/Python identity differs from this invocation")
            return collected_nodeids(record, allow_deselected=allow_deselected)

        pytest_prefix = [sys.executable, "-m", "pytest", "-p", "scripts.gpu_validation_scope",
                         "-c", str(ROOT / "pyproject.toml"), "--rootdir", str(ROOT)]
        collection = output / "collection.json"
        command = [*pytest_prefix, "--collect-only", "-q", "test/gpu"]
        if args.k:
            command += ["-k", args.k]
        def validate_collection():
            nodes = check_collection(collection, allow_deselected=bool(args.k))
            return {"groups": partition_sanitizer_nodeids(nodes, require_all_faults=not bool(args.k))}
        collected = stage("collection", command, selection=collection, validator=validate_collection)
        groups = collected["validation"]["groups"]
        report["groups"] = groups
        # pytest's default retention deletes older sessions. Keep every group's
        # actual probe/compiler/fixture files inside this invocation instead.
        (output / "work").mkdir()
        for tool in args.tools:
            for group, nodes in groups.items():
                if not nodes:
                    continue
                label = f"{tool}-{group}"
                xml = output / (label + ".xml")
                identity = output / (label + "-selection.json")
                command = [*pytest_prefix, "-x", "-ra", f"--junitxml={xml}",
                           f"--basetemp={output / 'work' / label}", *nodes]
                if tool != "plain":
                    prefix = [sanitizer, "--tool", tool, "--target-processes", "all", "--error-exitcode", "86"]
                    if group == "loader_api":
                        prefix += ["--suppressions", suppression["path"]]
                    command = [*prefix, *command]
                def validate_execution():
                    check_collection(identity)
                    return {"junit": validate_pytest_selection(xml, identity, expected_nodeids=nodes),
                            "junit_sha256": digest(xml)}
                item = stage(label, command, selection=identity, validator=validate_execution)
                report["runs"].append({"tool": tool, "group": group, "nodeids": nodes,
                                       "status": item["status"], "junit": item["validation"]["junit"],
                                       "suppression": suppression if tool != "plain" and group == "loader_api" else None})
        for relative, expected_hash in report["source_sha256"].items():
            if digest(ROOT / relative) != expected_hash:
                raise Failed("GPU validation source changed during execution: " + relative)
        report.update(status="passed", exit_code=0)
    except (Exception, KeyboardInterrupt) as error:
        report.update(status="failed", exit_code=1, reason=f"{type(error).__name__}: {error}")
        if isinstance(error, KeyboardInterrupt):
            report["interrupted"] = True
    report["finished_unix_ns"] = time.time_ns()
    save()
    print(f"{report['status']}: {output / 'results.json'}")
    return report["exit_code"]


def main(argv=None):
    options = parser()
    args = options.parse_args(argv)
    try:
        return run(args)
    except ValueError as error:
        options.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
