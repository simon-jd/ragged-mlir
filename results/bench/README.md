# Benchmark results

Measured on an RTX 4090 (driver 595.99, CUDA 13.2) at commit
`f0bd6903feb5b75c52cd1857e666887a14ee9db7`, which every `results.json` under
`runs/` records under `source`.

- `runs/<group>/<case>/seed-<seed>/rep-<n>/results.json`: one `bench.run`
  process each, from `python -m bench.suite --lock /var/tmp/rtx4090.lock
  --lock-batch 600` (19 workloads with every method, 6 with length hints, 52
  ablations; seeds 711 to 713; 5 repetitions; warm and cold L2). The GPU was
  shared with other projects, so the suite held the lock for up to 10 minutes
  at a time and waited between batches. `runs/suite.log` lists every
  process's exit code and duration; exit 1 means a method failed (the
  tracebacks are beside the results), which here is always eager NJT.
- `summary.json`: `python -m bench.summarize results/bench/runs --output
  results/bench/summary.json`. For each method and L2 state it gives the
  median over the 15 runs of each run's median call time, the 25th and 75th
  percentiles, each seed's median, and for `ragged-auto` the schedule each
  seed's runs chose.
- `calibration/`: the 190 runs (seeds 711 and 712, `ragged-classified` and
  `ragged-naive`, at commit `686d9ce`) that `python/ragged/cost_model.json`
  is fitted to, the fit (`fit.txt`, from `python -m bench.calibrate fit`),
  and `evaluation.txt` (`python -m bench.calibrate evaluate results/bench/runs
  --seeds 713`, then all three seeds): both schedules, the chosen one and the
  adaptive call per workload and seed.
- `profile.txt`: Nsight Systems kernel traces of both schedules on seed 711's
  inputs, warm and cold L2.
- `host.txt`: host time per call of both schedules.
- `cold-l2.txt`: a cold-L2 call after the benchmark's flush and after a flush
  that only reads, which separates the write-back of the flush's dirty lines.

Clocks were not locked; each run records `nvidia-smi` clocks, temperature,
power and utilization before its first method and after each one, and the
host load average (SM clock 2,520 to 2,805 MHz, memory clock 10,251 to
10,501 MHz, 37 to 64 °C over the suite). Paths in the records are relative to
the checkout.
