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
- `runs/heldout/` and `calibration/heldout-shapes.txt`: the cost model on
  seven shapes it was not fitted on, fixed in `bench/suite.py`
  (`HELDOUT_SHAPES`) before they were measured (below).
- `burst-gap/`: both ragged programs interleaved in one process on the D=1
  bursty sum of each seed (below); `run.py` produced `results.json`.
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

## The cost model on held-out shapes

`python -m bench.suite --groups heldout --repetitions 3` at commit
`a7a1428` (567 runs: 7 shapes × sum, max, softmax × uniform, long-tail,
bursty × seeds 711 to 713 × 3 repetitions), then `python -m bench.calibrate
evaluate results/bench/runs --group heldout --seeds 711 712 713`
(`calibration/heldout-shapes.txt`). The fitted shapes are D=1 with 16,384
segments in 1,048,576 rows and D=128 with 4,096 segments in 262,144 rows. Per
shape, over its 27 (workload, seed) inputs, the adaptive call against the
faster of the two schedules measured in the same runs, warm L2:

| D | Segments | Rows | Within 3% | Median excess | Largest excess |
|---:|---:|---:|---:|---:|---|
| 1 | 4,096 | 1,048,576 | 14 of 27 | +2.1% | +62% (softmax, uniform) |
| 1 | 65,536 | 1,048,576 | 0 of 27 | +24.9% | +115% (softmax, long tail) |
| 32 | 8,192 | 524,288 | 18 of 27 | +0.2% | +41% (max, uniform) |
| 64 | 4,096 | 262,144 | 17 of 27 | +0.3% | +80% (max, long tail) |
| 128 | 1,024 | 262,144 | 26 of 27 | +0.0% | +3.3% (max, uniform) |
| 128 | 16,384 | 262,144 | 19 of 27 | +0.2% | +14% (softmax, long tail) |
| 256 | 2,048 | 131,072 | 21 of 27 | +0.4% | +8.0% (sum, bursty) |

In all, 115 of 189 inputs are within 3% warm and 101 of 189 cold. The
model's two systematic errors: with 65,536 segments at D=1 it always runs
block per segment, which is 12 to 115% slower than classification; at D=32
and D=64 with uniform lengths it classifies the sum and max, 35 to 41% slower
than block per segment (and runs block per segment for the D=32 uniform
softmax, 19 to 20% slower). The D=1 shapes are also the noisiest: their calls
take 24 to 167 µs, much of it host time, and in 7 of their 162 (workload,
seed, method) cells one run is 1.5 to 2.5 times the cell's median, which
three repetitions per seed do not always outvote.

## Seed 711's bursty D=1 sum

`calibration/evaluation.txt` shows the adaptive call 6.7% slower than
`ragged-naive` on seed 711's bursty D=1 sum (32.8 against 30.7 µs), although
both run the same kernel there. The seed does not change this workload's
lengths, only where the burst of long segments starts, and four of the five
`ragged-naive` runs of seed 711 measured 30.7 µs against 32.5 to 32.8 µs on
the other seeds. Interleaving the two programs in one process (`burst-gap/`:
10 rounds of 30 calls each, alternating the order) does not reproduce the
gap:

| Seed | ragged-naive | ragged (adaptive) | host enqueue, naive / adaptive | kernel, both |
|---:|---:|---:|---:|---:|
| 711 | 31.7 | 31.9 | 27.5 / 28.4 | 14.4 |
| 712 | 36.4 | 36.0 | 36.6 / 35.4 | 14.4 |
| 713 | 34.6 | 35.3 | 33.5 / 34.4 | 14.4 |

The kernel takes 14.4 µs for every seed (torch.profiler); the rest of the
call is host time, which the GPU waits for at D=1. The adaptive call's
schedule lookup adds about 1 µs of host work at most, and the 2 µs of the
suite is run-to-run variation in that host part, which the first rounds of
each seed also show (73 to 130 µs before the process settles).
