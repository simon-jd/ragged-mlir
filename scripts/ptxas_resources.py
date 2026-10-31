"""Independent evidence parser for pinned ptxas verbose entry reports.

This is verification tooling, not the compiler or runtime's resource parser.
"""
from __future__ import annotations

import re


def parse_resource_log(text: str, expected_names=None) -> dict[str, dict[str, int]]:
    chunks = re.split(r"Compiling entry function '([^']+)' for '([^']+)'", text)
    if len(chunks) == 1:
        raise ValueError("no ptxas entry resource reports")
    entries = {}
    for index in range(1, len(chunks), 3):
        name, chip, properties = chunks[index:index + 3]
        if chip != "sm_89":
            raise ValueError(f"unexpected ptxas target for {name}: {chip}")
        if name in entries:
            raise ValueError(f"duplicate ptxas entry: {name}")
        headers = re.findall(r"Function properties for (\S+)", properties)
        if headers != [name]:
            raise ValueError(f"missing or mismatched function properties for {name}")

        def one(pattern, *, optional=False):
            values = re.findall(pattern, properties)
            if optional and not values:
                return 0
            if len(values) != 1:
                raise ValueError(f"incomplete or duplicate resource field for {name}: {pattern}")
            return int(values[0])

        entries[name] = {
            "registers_per_thread": one(r"Used (\d+) registers"),
            "static_shared_bytes": one(r"(\d+) bytes smem", optional=True),
            "stack_frame_bytes": one(r"(\d+) bytes stack frame"),
            "spill_store_bytes": one(r"(\d+) bytes spill stores"),
            "spill_load_bytes": one(r"(\d+) bytes spill loads"),
        }
    if expected_names is not None:
        names = list(expected_names)
        if len(names) != len(set(names)) or set(entries) != set(names):
            raise ValueError("ptxas entries differ from the expected main/validation entries")
    return entries
