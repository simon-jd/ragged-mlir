"""Exact, stdlib-only sanitizer scopes and an actual pytest collection recorder.

Only loader_api may use the separately validated API suppression file. Main and
native_safe retain every sanitizer diagnostic. This module also acts as a pytest
plugin for the existing-environment runner; the fresh-wheel runner uses its own
installed-wheel identity plugin and shares the same partition function.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

FAULT_FILE = "test/gpu/test_driver_faults.py"
FAULT_FUNCTION = FAULT_FILE + "::test_real_driver_faults_in_isolated_process"
SAFE_MODES = ("valid", "bad_magic", "declared_target")
API_MODES = ("corrupt_elf", "wrong_target", "runtime_wrong_target")
SAFE_NODEIDS = tuple(f"{FAULT_FUNCTION}[{mode}]" for mode in SAFE_MODES)
API_NODEIDS = tuple(f"{FAULT_FUNCTION}[{mode}]" for mode in API_MODES)
FAULT_NODEIDS = SAFE_NODEIDS + API_NODEIDS
GROUPS = ("main", "native_safe", "loader_api")
SELECTION_ENV = "RAGGED_GPU_SELECTION_PATH"
MARKER = "cuda_loader_api_fault: isolated real cuModuleLoadData rejection requiring narrowly scoped API suppression under compute-sanitizer"


def checked_nodeids(nodes, *, nonempty=True):
    if not isinstance(nodes, (list, tuple)) or (nonempty and not nodes):
        raise ValueError("actual nodeids must be a nonempty list or tuple")
    for node in nodes:
        if (not isinstance(node, str) or not node or "::" not in node
                or any(ord(c) < 32 or ord(c) == 127 for c in node)
                or any(not component for component in node.split("::"))):
            raise ValueError("invalid pytest node identity")
        filename = node.split("::", 1)[0]
        if (not filename.startswith("test/gpu/") or "\\" in filename
                or any(part in ("", ".", "..") for part in filename.split("/"))
                or not filename.endswith(".py")):
            raise ValueError("node identity must be relative to the fixed GPU source root")
    if len(set(nodes)) != len(nodes):
        raise ValueError("duplicate actual pytest node identity")
    return list(nodes)


def partition_sanitizer_nodeids(actual_collected_nodes, require_all_faults=True):
    """Partition the actual selection without dropping, duplicating or guessing cases."""
    if type(require_all_faults) is not bool:
        raise ValueError("require_all_faults must be a bool")
    nodes = checked_nodeids(actual_collected_nodes)
    groups = {name: [] for name in GROUPS}
    for node in nodes:
        if node in SAFE_NODEIDS:
            groups["native_safe"].append(node)
        elif node in API_NODEIDS:
            groups["loader_api"].append(node)
        elif Path(node.split("::", 1)[0]).name == "test_driver_faults.py":
            raise ValueError("unknown Driver fault test or fault mode: " + node)
        else:
            groups["main"].append(node)
    if require_all_faults and not set(FAULT_NODEIDS).issubset(nodes):
        raise ValueError("full sanitizer collection must include all six Driver fault cases")
    combined = [node for group in groups.values() for node in group]
    if len(combined) != len(nodes) or set(combined) != set(nodes):
        raise ValueError("sanitizer scopes do not partition the actual selection")
    return groups


def collected_nodeids(record, *, allow_deselected=False):
    selection = record.get("pytest_selection", {})
    nodes = checked_nodeids(selection.get("nodeids"))
    deselected = checked_nodeids(selection.get("deselected"), nonempty=False)
    if (type(selection.get("collected")) is not int or selection["collected"] != len(nodes)
            or set(nodes).intersection(deselected) or (deselected and not allow_deselected)):
        raise ValueError("invalid actual pytest collection evidence")
    return nodes


def validate_suppression_file(path):
    """Reject broadening: exactly two API/error/result rules, no device/stack rules."""
    path = Path(path)
    root = ET.parse(path).getroot()
    rules = []
    if root.tag != "ComputeSanitizerOutput" or root.attrib or len(root) != 2:
        raise ValueError("Driver suppression must contain exactly two API rules")
    for record in root:
        if record.tag != "record" or record.attrib or [child.tag for child in record] != ["kind", "what"]:
            raise ValueError("invalid Driver suppression record")
        kind, what = record
        if kind.text != "Api" or kind.attrib or len(kind) or what.attrib or [child.tag for child in what] != ["api", "error", "result"]:
            raise ValueError("only exact API/error/result suppressions are permitted")
        if any(child.attrib or len(child) for child in what):
            raise ValueError("unexpected suppression fields")
        rules.append(tuple(child.text for child in what))
    expected = {("^cuModuleLoadData$", "^CUDA_ERROR_INVALID_IMAGE$", "200"),
                ("^cuModuleLoadData$", "^CUDA_ERROR_NO_BINARY_FOR_GPU$", "209")}
    if len(set(rules)) != 2 or set(rules) != expected:
        raise ValueError("Driver suppression API/error/result scope differs from the two permitted faults")
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "rules": [dict(zip(("api", "error", "result"), rule)) for rule in rules],
            "scope": "loader_api group only; all device checks and unrelated API errors remain enabled"}


def pytest_configure(config):
    config.addinivalue_line("markers", MARKER)
    config._ragged_scope_deselected = []


def pytest_deselected(items):
    if items:
        items[0].config._ragged_scope_deselected.extend(item.nodeid for item in items)


def pytest_collection_finish(session):
    for item in session.items:
        if any(name == "ragged_nodeid" for name, _ in item.user_properties):
            raise ValueError("reserved ragged_nodeid property was already present")
        item.user_properties.append(("ragged_nodeid", item.nodeid))
    destination = os.environ.get(SELECTION_ENV)
    if not destination:
        raise ValueError("collection recorder requires " + SELECTION_ENV)
    record = {"pytest_selection": {"collected": len(session.items),
              "nodeids": [item.nodeid for item in session.items],
              "deselected": session.config._ragged_scope_deselected},
              "collector": {"file": str(Path(__file__).resolve()),
                            "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
              "python": {"executable": str(Path(sys.executable).absolute()), "version": sys.version},
              "rootdir": str(session.config.rootpath.resolve()), "cwd": str(Path.cwd().resolve())}
    path = Path(destination)
    if path.exists():
        raise ValueError("refusing to overwrite prior collection evidence")
    path.write_text(json.dumps(record, allow_nan=False, indent=2) + "\n")
