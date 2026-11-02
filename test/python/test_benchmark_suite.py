"""Host tests for the benchmark suite runner, L2 flush timing and summary generation."""
from contextlib import nullcontext
import json
from types import SimpleNamespace

import pytest

from bench import run as benchmark
from bench.suite import ALL_METHODS, cases
from bench.summarize import summarize


def test_suite_covers_every_case_with_each_comparison():
    configurations = cases()
    assert len({(group, name) for group, name, _ in configurations}) == len(configurations) == 77
    main = [flags for group, _, flags in configurations if group == "main"]
    assert len(main) == 19
    assert all(flags[flags.index("--methods") + 1:] == list(ALL_METHODS) for flags in main)
    groups = {group for group, _, _ in configurations}
    assert groups == {"main", "hints", "ablation/serial-classes", "ablation/no-row-pipelining", "ablation/three-pass",
                      "ablation/no-vectorize"}


class FakeTorch:
    """Records the order of flushes, synchronizations and timed calls."""

    def __init__(self, log):
        self.log = log
        outer = self

        class Event:
            def __init__(self, enable_timing):
                pass

            def record(self):
                outer.log.append("record")

            def synchronize(self):
                pass

            def elapsed_time(self, other):
                return 0.5

        self.cuda = SimpleNamespace(Event=Event, synchronize=lambda: log.append("sync"),
                                    nvtx=SimpleNamespace(range=lambda name: nullcontext()))


class Flush:
    def __init__(self, log):
        self.log = log

    def zero_(self):
        self.log.append("flush")

    def numel(self):
        return 1024

    def element_size(self):
        return 1


@pytest.mark.parametrize("l2", ["warm", "cold", "both"])
def test_cold_timings_flush_and_synchronize_before_every_call(l2):
    log = []
    torch = FakeTorch(log)
    torch.cuda.memory_allocated = torch.cuda.memory_reserved = torch.cuda.max_memory_allocated = \
        torch.cuda.max_memory_reserved = lambda: 0
    torch.cuda.reset_peak_memory_stats = lambda: None
    call = lambda: log.append("call") or object()
    result = benchmark.measure(call, torch, method="m", warmup=1, repeats=3, l2=l2, flush=Flush(log))
    assert ("complete_call_cuda_event_ms" in result) == (l2 != "cold")
    assert ("cold_l2" in result) == (l2 != "warm")
    # Warm-up, then the memory call, each followed by a synchronization.
    assert log[:4] == ["call", "sync", "call", "sync"]
    warm = ["record", "call", "record"] * 3 if l2 != "cold" else []
    cold = ["flush", "sync", "record", "call", "record"] * 3 if l2 != "warm" else []
    assert log[4:] == warm + cold
    if l2 != "warm":
        assert result["cold_l2"]["complete_call_cuda_event_ms"]["count"] == 3
        assert result["cold_l2"]["flush_bytes"] == 1024


def _write(root, group, case, seed, rep, cells, commit="abc"):
    path = root / group / case / f"seed-{seed}" / f"rep-{rep}"
    path.mkdir(parents=True)
    report = {"status": "complete", "source": {"git_commit": commit},
              "case": {"op": "sum", "dtype": "f32", "D": 1, "distribution": "uniform", "elements": 64,
                       "segments": 2, "min_length": 32, "max_length": 32},
              "gpu_state": {"before": {"host_loadavg": [1.0, 1, 1], "gpus": [{"clocks.sm": "2520"}]},
                            "after": {"host_loadavg": [3.0, 1, 1], "gpus": [{"clocks.sm": "2760"}]}},
              "methods": cells}
    (path / "results.json").write_text(json.dumps(report))


def _cell(method, warm_ms, cold_ms=None, status="measured"):
    cell = {"method": method, "status": status}
    if status == "measured":
        cell["complete_call_cuda_event_ms"] = {"median": warm_ms}
        if cold_ms is not None:
            cell["cold_l2"] = {"complete_call_cuda_event_ms": {"median": cold_ms}}
    else:
        cell["reason"] = "synthetic"
    return cell


def test_summary_is_the_median_and_quartiles_of_per_run_medians(tmp_path):
    for rep, (warm, cold) in enumerate([(0.010, 0.020), (0.012, 0.024), (0.014, 0.028), (0.016, 0.032)]):
        _write(tmp_path, "main", "sum-f32-d1-uniform", 711 + rep % 2, rep,
               [_cell("ragged-auto", warm, cold), _cell("cub", 0.005, status="measured" if rep else "failed")])
    _write(tmp_path, "ablation/serial-classes", "sum-f32-d1-uniform", 711, 0, [_cell("ragged-auto", 0.02)])
    summary = summarize(tmp_path)
    assert summary["runs"] == 5 and summary["source_commits"] == ["abc"]
    main, ablation = sorted(summary["cases"], key=lambda entry: entry["group"], reverse=True)
    assert (main["group"], ablation["group"]) == ("main", "ablation/serial-classes")
    ragged = main["methods"]["ragged-auto"]
    assert ragged["warm"] == {"median_us": 13.0, "p25_us": 11.5, "p75_us": 14.5, "runs": 4,
                              "by_seed_median_us": {"711": 12.0, "712": 14.0}}
    assert ragged["cold"]["median_us"] == 26.0
    cub = main["methods"]["cub"]
    assert cub["statuses"] == {"failed": 1, "measured": 3} and cub["warm"]["runs"] == 3 and "cold" not in cub
    assert main["lengths_by_seed"] == {"711": [32, 32], "712": [32, 32]}
    assert summary["gpu_state"]["clocks.sm"] == [2520.0, 2760.0] and summary["host_loadavg_1min"] == [1.0, 3.0]
