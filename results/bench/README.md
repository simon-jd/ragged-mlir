# Benchmark results

Measured on an RTX 4090 (driver 595.99, CUDA 13.2) at commit
`8ed1380cfef61320b7b818df571bdfb823912b46`, which every `results.json` records
under `source`.

- `runs/<group>/<case>/seed-<seed>/rep-<n>/results.json`: one `bench.run`
  process each, from `python -m bench.suite --lock /var/tmp/rtx4090.lock`
  (19 workloads with every method, 6 with length hints, 43 ablations; seeds
  711 to 713; 5 repetitions; warm and cold L2). `runs/suite.log` lists every
  process's exit code and duration; exit 1 means a method failed (the
  tracebacks are beside the results), which here is always eager NJT.
- `summary.json`: `python -m bench.summarize results/bench/runs --output
  results/bench/summary.json`. For each method and L2 state it gives the
  median over the 15 runs of each run's median call time, the 25th and 75th
  percentiles, and each seed's median.
- `profile.txt`: Nsight Systems kernel traces of ragged on seed 711's inputs.
- `host.txt`: host time per ragged call.

Clocks were not locked; each run records `nvidia-smi` clocks, temperature,
power and utilization before its first method and after each one, and the
host load average (SM clock 2,520 to 2,805 MHz, memory clock 10,251 to
10,501 MHz, 36 to 65 °C over the suite). Paths in the records are relative to
the checkout.
