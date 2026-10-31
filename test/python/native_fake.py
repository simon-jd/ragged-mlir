"""A host stand-in for ragged._native.DriverModule: no CUDA, no kernels.

Launch plans run as the native module runs them: the arguments are the inputs
followed by the scratch descriptors with their pointer fields relocated to the
scratch base; a side lane waits for a fork event recorded on the caller stream
before its first step, and the caller waits for each side lane's join event
after the last step. Every launch goes through ``launch``.
"""
from __future__ import annotations

from typing import Any, Callable


class PlanModule:
    def __init__(self, launch: Callable[..., Any], events: list | None = None):
        self.launch, self.events, self.plans = launch, events if events is not None else [], []

    @staticmethod
    def resident_blocks(name: str, threads: int, shared: int) -> int:
        # An RTX 4090: 128 multiprocessors, 24 resident 32-thread or 12 128-thread blocks each.
        return 128 * (24 if threads == 32 else 12)

    @staticmethod
    def side_stream(caller: int, lane: int) -> int:
        return caller * 1000 + lane

    def add_plan(self, steps, scratch, relocations) -> int:
        self.plans.append((list(steps), list(scratch), list(relocations)))
        return len(self.plans) - 1

    def clear_plans(self) -> None:
        self.plans.clear()

    def run_plan(self, plan: int, stream: int, inputs: list[int], scratch_base: int) -> None:
        steps, scratch, relocations = self.plans[plan]
        arguments = list(inputs) + scratch
        for index in relocations:
            arguments[len(inputs) + index] += scratch_base
        forked: list[int] = []
        for name, grid, block, shared, lane in steps:
            target = stream
            if lane:
                target = self.side_stream(stream, lane)
                if lane not in forked:
                    if not forked:
                        self.events.append(("record", stream, "fork"))
                    self.events.append(("wait", target, "fork"))
                    forked.append(lane)
            self.events.append(("launch", target, name))
            self.launch(name, grid, list(block), shared, target, arguments)
        for lane in sorted(forked):
            self.events.append(("record", self.side_stream(stream, lane), ("join", lane)))
            self.events.append(("wait", stream, ("join", lane)))
