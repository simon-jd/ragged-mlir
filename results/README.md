# Results

Every committed result, the commit it was measured at, when, and the claim it
backs. All of it but `a10/` ran on one RTX 4090 (driver 595.99, CUDA 13.2)
that also drives a desktop and is shared with other projects; clocks were not
locked. `a10/` holds the runs on one NVIDIA A10-24Q vGPU (sm_86) on a
Snowflake trial compute pool and lists its own files
([a10/README.md](a10/README.md)).
Dates are local (UTC-8). The detailed tables are in
[bench/README.md](bench/README.md).

## Benchmark runs

Each `bench/runs/<group>/<case>/seed-<seed>/rep-<n>/results.json` is one
`python -m bench.run` process: one input, every method of that group, warm and
cold L2. It records the commit it ran at (`source`), the `nvidia-smi` clocks,
temperature, power and utilization before the first method and after each
one, and the host load average. A case is `<op>-<dtype>-d<D>-<lengths>`, with
`-b<segments>` for the held-out shapes.

| Path | Runs | Measured at | Date | What it measured | Backs |
|---|---:|---|---|---|---|
| `bench/runs/main/` | 285 | `f0bd690` | 2026-11-01 | 19 workloads, seeds 711 to 713, 5 repetitions: ragged with cost model version 1, both forced schedules, the comparisons of that commit, eager NJT | the main-run tables in bench/README.md; half of version 2's training data |
| `bench/runs/hints/` | 90 | `f0bd690` | 2026-11-01 | 6 workloads with exact length hints | the length-hint table |
| `bench/runs/ablation/` | 780 | `f0bd690` | 2026-11-01 | 52 workload and ablation pairs: three passes, no row pipelining, serial classes, no vector access | the ablation and vector-access tables |
| `bench/runs/compare/` | 285 | `2d69970` (210 runs), `6b0b842` (75) | 2026-11-02, 11-03 | the 19 main workloads against every comparison, with CUDA-graph replays | the 19-workload table in the top README (in-sample for the cost model) |
| `bench/runs/heldout/` | 567 | `a7a1428` | 2026-11-02 | first held-out set: 7 shapes, 63 workloads, 3 seeds, 3 repetitions, both schedules | version 1's held-out result (115 of 189); the other half of version 2's training data |
| `bench/runs/heldout2/` | 432 | `6b0b842` | 2026-11-03 | second held-out set: 8 shapes, 48 workloads, 3 seeds, 3 repetitions, version 2 installed | the held-out result in the top README |
| `bench/calibration/main/` | 190 | `686d9ce` | 2026-11-01 | the 19 main workloads, seeds 711 and 712, 5 repetitions, both forced schedules only | version 1's fit (`fit-v1.txt`) |

Notes on the runs:

- The `ragged` (adaptive) method ran cost model version 1 in every group but
  `heldout2` and the 75 re-run `compare` runs, which ran version 2. Those 75
  are the five workloads whose schedule version 2 changes (the bursty and
  long-tail D=128 sum and max, the long-tail D=1 softmax); they replace the
  earlier runs of those workloads. The commits between `f0bd690` and
  `6b0b842` do not change the generated kernels.
- The `heldout` runs record one uncommitted edit, to `docs/upstream.md`.
- The two held-out sets are fixed in `bench/suite.py` (`HELDOUT_SHAPES`,
  `HELDOUT2_SHAPES`). Each was committed before the model it evaluates ran on
  it: the second in `5551923`, before the refit in `6b0b842`.
- The suites ran as `python -m bench.suite --lock /var/tmp/rtx4090.lock`
  with `--lock-batch 600` (main, hints, ablation), `540` (`compare`,
  `heldout`) and `480` (`heldout2` and the 75 re-runs). The GPU was shared, so
  a suite held the lock for up to 10 minutes at a time and waited between
  batches.
- `bench/runs/suite.log` (main, hints and ablation, 1,155 processes) and the
  `suite.log` in `compare/`, `heldout/` and `heldout2/` list every process's
  exit code and duration. Exit 1 means a method failed; here that is always
  eager NJT, whose 120 tracebacks (`njt-eager.stderr.txt`) are beside the
  results in `main/`.
- No run tree duplicates another. `main` and `compare` measure the same 19
  workloads at different commits with different method sets; `main`'s
  comparison columns are superseded by `compare`, but its forced schedules,
  eager NJT and version 1 choices are not measured anywhere else, and it is
  training data for version 2.

## Derived files

| Path | Measured at | Recorded | What it holds |
|---|---|---|---|
| `bench/summary.json` | the runs above | `1dd9504`, 2026-11-03 | per method and L2 state: the median over the 15 runs of each run's median call time, the 25th and 75th percentiles, each seed's median, and for `ragged-auto` each seed's schedule. `python -m bench.summarize results/bench/runs --output results/bench/summary.json` |
| `bench/calibration/cost_model-v1.json`, `fit-v1.txt` | fitted at `6e587a4` | `6b0b842`, 2026-11-03 (shipped from `6e587a4` until then) | version 1 of the cost model and its fit on `calibration/main` |
| `bench/calibration/evaluation-v1.txt` | `runs/main` | `1d2a666`, 2026-11-02 | version 1's choices on `runs/main`: seed 713 (not in its fit), then all seeds |
| `bench/calibration/heldout-shapes.txt` | `runs/heldout` | `68237eb`, 2026-11-02 | version 1 on `runs/heldout` |
| `bench/calibration/fit.txt` | `runs/main`, `runs/heldout` | `6b0b842`; cold-L2 cross-validation `e5bd70a`, 2026-11-03 | version 2 (the shipped `python/ragged/cost_model.json`), fitted on `runs/main` and `runs/heldout` (`python -m bench.calibrate fit results/bench/runs --groups main heldout`), with leave-one-shape-out cross-validation |
| `bench/calibration/refit-training.txt` | `runs/main`, `runs/heldout` | `6b0b842`, 2026-11-03 | both versions' choices on version 2's training runs; version 2's numbers there are in-sample |
| `bench/calibration/heldout2.txt` | `runs/heldout2` | `054f524`, 2026-11-03 | version 2 on `runs/heldout2`, and version 1's choices replayed on the same runs |
| `bench/profile.txt` | `f0bd690`; last part `6b0b842` | `1d2a666`, 2026-11-02; `593d0ef`, 2026-11-03 | Nsight Systems kernel traces of both schedules on seed 711's inputs, warm and cold L2; then both schedules on four first-held-out inputs where version 1 chose the slower one |
| `bench/host.txt` | `f0bd690` | `1d2a666`, 2026-11-02 | host time per call of both schedules |
| `bench/cold-l2.txt` | `f0bd690` | `1d2a666`, 2026-11-02 | a cold-L2 call after the benchmark's flush and after a flush that only reads, which separates the write-back of the flush's dirty lines |
| `bench/burst-gap/` | not recorded | `df60c47`, 2026-11-02 | both ragged programs interleaved in one process on the bursty D=1 sum of each seed; `run.py` wrote `results.json` |
| `tests/summary.txt` | see the file | | GPU, host and lit/FileCheck test runs |
| `sanitizer/summary.txt` | see the file | | compute-sanitizer runs |
