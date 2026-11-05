# Results on an A10 vGPU (sm_86)

One NVIDIA A10-24Q vGPU (sm_86, 72 multiprocessors, 24 GB; PyTorch reports a
6 MiB L2 against the RTX 4090's 72 MiB) on a Snowflake trial compute pool,
driver 570.211.01 with the CUDA 13.2 forward-compatibility driver (libcuda
595.112.07). Four Snowflake ML jobs on 2026-11-05 (UTC), listed in
`jobs/`. The node has no internet, no CUDA toolkit and an older glibc
(2.39) than the build machine (2.43), so each job received:

- a wheelhouse of the repository's pinned environment (PyTorch 2.14.1+cu132,
  Triton 3.8.0, FBGEMM_GPU 1.9.0+cu130, NumPy, pytest), installed with
  `pip --no-index --no-deps`;
- a `git archive` of the commit, with the native runtime module and the
  CUDA/CUB comparison library (`-arch=sm_86`) built inside `ubuntu:24.04`
  (glibc 2.39);
- every ragged program the runs use, compiled for sm_86 on the build machine
  into an artifact cache with its toolchain record (`bench.precompile` and a
  4090 run of the GPU tests with `RAGGED_TEST_CHIP=sm_86`). No
  `ragged-compile`, `ptxas` or `nvcc` was on the node's `PATH`, and
  `RAGGED_COMPILE` named a missing file: every ragged program was loaded
  from the cache (`Compiler(toolchain_record=...)`);
- `ptxas` 13.2.86 and libdevice for the Triton comparison only
  (`TRITON_PTXAS_PATH`, as on the 4090).

Not available on the node: compute-sanitizer (GPU debugging is disabled on
the vGPU), CUPTI, clock locking. `nvidia-smi` reports SM and memory clocks
of 1,695 and 6,251 MHz before and after every step and no temperature or
power. The runs' `results.json` record `source.git_commit` as null because
the archive is not a git checkout; the commit is in each job's setup
`RESULT` line (`jobs/*/instance0.log`) and in the table below.

| Path | Job | Commit | What it holds |
|---|---|---|---|
| `tests/gpu-tests.log`, `gpu-tests.xml` | ragged-c | `0e4f674` | `python -m pytest test/gpu` with `RAGGED_TEST_CHIP=sm_86` and the precompiled cache: 2,324 passed, 6 skipped (the Driver fault probes, which assemble sm_89 and sm_90 cubins with `ptxas` and expect an sm_89 device), 30 strict xfails (FBGEMM, as on the 4090), 0 failed |
| `tests/before-0e4f674/` | ragged-a | `c49c5c0` | the same run before the Triton and FBGEMM adapters accepted sm_86 (and without a `ptxas` for Triton): every ragged test passed; 146 comparison tests failed and 132 errored on those two refusals |
| `runs/heldout2/` | ragged-a | `c49c5c0` | the second held-out set: 8 shapes, 48 workloads, seeds 711 to 713, 3 repetitions, both schedules and the adaptive call (432 runs) |
| `runs/heldout/` | ragged-b | `0e4f674` | the first held-out set: 7 shapes, 63 workloads, 3 seeds, 3 repetitions (567 runs) |
| `runs/compare/` | ragged-c, ragged-d | `0e4f674` | the 19 benchmark workloads against every comparison, with CUDA-graph replays and `ragged-classified` added, 3 seeds, 5 repetitions (285 runs) |
| `transfer.txt` | | `0e4f674` | `python -m bench.calibrate transfer results/a10/runs --output results/a10/cost_model-a10.json`: the held-out evaluation of the three models |
| `cost_model-a10.json` | | | the model refitted on `runs/compare` and `runs/heldout` |
| `summary.json` | | | `python -m bench.summarize results/a10/runs --output results/a10/summary.json` |
| `jobs/<run>/` | | | each job's `report.json` (status trace, seconds, credit estimate) and container log with the `RESULT` lines; `runs/<group>/*-suite-stdout.txt` is each step's suite output with `nvidia-smi` before and after |

The transfer protocol (which runs train the refit, which are held out) was
committed in `c49c5c0` before any A10 run (`bench/suite.py`,
`TRANSFER_FIT_GROUPS`). The two commits differ only in the comparison
adapters, which the held-out groups do not run.

Job ragged-a ran the GPU tests and `heldout2`, then was terminated by the
platform 69 minutes in, during `heldout` (status `TERMINATING`, no error
from the payload); those runs were lost and `heldout` was run again, by
seed, in ragged-b. The `compare` group ran by seed in ragged-c (711, 712)
and ragged-d (713). `runs/<group>/suite.log` lists every process's exit code
and duration; all 1,284 exited 0. The four jobs took 4,192, 2,354, 3,556 and
1,903 s, estimated at 1.98, 1.11, 1.68 and 0.90 credits (5.67 in all).

## The cost model on the A10

The same 144 inputs as the 4090's second held-out set (README rows 1 and
2). Each model's choice is replayed on the A10's medians of both forced
schedules; "as run" is the adaptive call measured, which ran the shipped
model with the device's L2. Inputs within 3% of the faster schedule, and the
mean excess over it (`transfer.txt`):

| Model | warm | cold |
|---|---:|---:|
| fitted on the RTX 4090, as shipped (72 MiB L2) | 120 of 144, +2.41% | 113 of 144, +2.29% |
| the same with the A10's 6 MiB L2, replayed | 126, +1.31% | 122, +1.49% |
| the same, as run | 126, +1.81% | 123, +2.00% |
| refitted on the A10 (`runs/compare`, `runs/heldout`: 246 inputs) | 138, +0.49% | 134, +0.75% |

On the 4090 the shipped model was within 3% for 112 warm and 104 cold as run
(119 and 107 replayed).

- As shipped it misses the D=2 bimodal sum and max (block per segment, 22%
  slower warm), the D=16 and D=2 softmax (classification, 11 to 18%) and the
  D=1 softmax with 200,000 segments (classification, 12 to 18%).
- With the A10's L2 it chooses right on the D=16 softmax and the D=2
  bimodal sum and max, but classifies the D=48 sum and max with 6,000
  segments, 7 to 12% slower than one block per segment, runs the D=2
  bimodal softmax as one block per segment (7 to 8% slower warm) and still
  classifies the D=1 softmax with 200,000 segments.
- Refitted, it still classifies the D=1 softmax with 200,000 segments (10
  to 18% slower) and runs the D=2 bimodal softmax as one block per segment
  (7 to 8% slower warm). Its fixed costs are about twice the 4090 model's:
  74 µs for a classified sum or max and 46 µs for block per segment, against
  36 and 21 µs (`cost_model-a10.json`).

## The 19 benchmark workloads

Medians over 15 processes (seeds 711 to 713, 5 repetitions), complete-call
µs with a warm L2, as in the top README's table; the result columns compare
ragged's adaptive call with the fastest comparison in each state, and the
last column is the 4090's warm result. *chose* is the adaptive program's
schedule per seed (C classification, N one block per segment). The SM clock
read 1,695 MHz before and after every step; it cannot be locked on the vGPU.

| Op | D | Lengths | chose | ragged | fastest comparison | warm | cold | CUDA graph replay | RTX 4090, warm |
|---|---:|---|---|---:|---|---|---|---|---|
| sum | 1 | uniform | N/N/N | 81 | torch.segment_reduce 35 | loss 2.32× | loss 2.18× vs torch.segment_reduce | loss 3.00× vs NJT compiled | loss 2.19× |
| sum | 1 | long tail | C/N/N | 81 | cuSPARSE 72 | loss 1.13× | loss 1.03× vs cuSPARSE | loss 1.09× vs Triton | loss 1.56× |
| sum | 1 | bursty | N/N/N | 80 | FBGEMM 62 | loss 1.28× | loss 1.14× vs FBGEMM | loss 1.41× vs Triton | loss 1.47× |
| sum | 128 | uniform | C/C/C | 336 | torch.segment_reduce 285 | loss 1.18× | loss 1.18× vs torch.segment_reduce | loss 1.08× vs NJT compiled | win 0.87× |
| sum | 128 | long tail | C/C/C | 369 | cuSPARSE 344 | loss 1.07× | loss 1.09× vs cuSPARSE | tie 1.01× vs Triton | loss 1.13× |
| sum | 128 | bursty | C/C/C | 354 | Triton 325 | loss 1.09× | loss 1.08× vs cuSPARSE | loss 1.10× vs Triton | loss 1.07× |
| max | 1 | uniform | N/N/N | 81 | torch.segment_reduce 35 | loss 2.32× | loss 2.19× vs torch.segment_reduce | loss 3.00× vs NJT compiled | loss 2.19× |
| max | 1 | long tail | C/N/N | 81 | Triton 76 | loss 1.07× | tie 1.00× vs Triton | loss 1.34× vs Triton | loss 1.11× |
| max | 1 | bursty | N/N/N | 80 | CUB 66 | loss 1.22× | loss 1.10× vs Triton | loss 1.65× vs Triton | loss 1.26× |
| max | 128 | uniform | C/C/C | 333 | torch.segment_reduce 296 | loss 1.13× | loss 1.12× vs torch.segment_reduce | loss 1.08× vs NJT compiled | win 0.86× |
| max | 128 | long tail | C/C/C | 356 | Triton 353 | tie 1.01× | tie 1.00× vs Triton | tie 0.99× vs Triton | tie 1.00× |
| max | 128 | bursty | C/C/C | 355 | Triton 325 | loss 1.09× | loss 1.08× vs Triton | loss 1.10× vs Triton | loss 1.07× |
| softmax | 1 | uniform | N/N/N | 91 | NJT compiled 115 | win 0.79× | win 0.78× vs NJT compiled | win 0.79× vs NJT compiled | win 0.80× |
| softmax | 1 | long tail | C/N/C | 110 | Triton 135 | win 0.81× | win 0.88× vs Triton | win 0.76× vs Triton | win 0.86× |
| softmax | 1 | bursty | N/N/N | 87 | Triton 94 | win 0.92× | win 0.92× vs Triton | win 0.96× vs Triton | tie 1.00× |
| softmax | 128 | uniform | N/N/N | 847 | Triton 669 | loss 1.27× | loss 1.26× vs Triton | loss 1.29× vs Triton | tie 0.99× |
| softmax | 128 | long tail | C/C/C | 975 | Triton 868 | loss 1.12× | loss 1.13× vs Triton | loss 1.14× vs Triton | tie 1.03× |
| softmax | 128 | bursty | C/C/C | 955 | Triton 948 | tie 1.01× | tie 1.00× vs Triton | tie 1.00× vs Triton | loss 1.05× |
| softmax bf16 | 128 | long tail | C/C/C | 679 | Triton 508 | loss 1.34× | loss 1.35× vs Triton | loss 1.42× vs Triton | loss 1.38× |

Ragged wins 3, ties 2 and loses 14 warm (the 4090: 4, 4 and 11), wins 3,
ties 3 and loses 13 cold (1, 3, 15) and on CUDA-graph replay (5, 4, 10). It
wins only the D=1 softmax. A D=1 sum or max takes about 80 µs (34 on the
4090), most of it the fixed cost the refit puts at 46 to 74 µs. The uniform
D=128 sum and max, wins on the 4090, lose to `torch.segment_reduce` by 1.18
and 1.13 times, and the uniform D=128 softmax, a tie on the 4090, loses to
tuned Triton by 1.27 times (both of ragged's schedules take about 845 µs;
with the A10's L2 the model runs block per segment there).
