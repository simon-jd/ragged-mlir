# ragged-mlir: an MLIR compiler for segmented reductions on ragged tensors

- An out-of-tree MLIR compiler that generates CUDA kernels for segmented sum,
  max and softmax over ragged tensors. It picks a warp, block or multi-block
  mapping for each segment on the device.
- On an RTX 4090, with a warm L2, it beats every comparison in 4 of 19
  workloads, is within 10% of the fastest in 6 and is more than 10% slower in
  9 (cold L2: 1, 7 and 11). The comparisons are autotuned Triton, CUB, `torch.segment_reduce`,
  NJT and a CUDA block-per-segment kernel. Its weakest case is D=1, where
  `torch.segment_reduce` or CUB is 1.4 to 3.2 times faster.
- Each time is the median of 15 processes (3 seeds of the length distribution
  × 5 repetitions). Every method passes a float64 oracle before it is timed.

![ragged's speedup over the fastest comparison method per workload, warm and cold L2](docs/speedup.svg)

Ragged (jagged) batches, a packed `[N, D]` tensor plus `B+1` segment offsets,
are how variable-length sequences, graph neighborhoods and recommendation
features reach a GPU. Reducing each segment (sum, max, softmax) is simple until
the lengths are skewed: one block per segment leaves most of the GPU idle on a
long segment, one warp per segment wastes it on short ones.

`ragged-mlir` is an out-of-tree compiler on MLIR 23.1.3 that generates CUDA
kernels (sm_89) for segmented sum, max and softmax and picks a mapping per
segment at run time. It has its own dialect and passes, lowers through the GPU
and NVVM dialects to PTX and `ptxas`, caches compiled artifacts, and launches
them through the CUDA Driver API on PyTorch streams.

```python
from ragged import ArtifactCache, Compiler, ProgramSpec

program = Compiler("build/bin/ragged-compile").compile(
    ProgramSpec("softmax", dtype="bf16", feature_dim=128, offsets_dtype="i64"),
    ArtifactCache("~/.cache/ragged"))
program.run(values, offsets, out)   # CUDA tensors; runs on the current stream
```

## How it works

- **Dialect.** `ragged.segment_reduce` and `ragged.segment_softmax` over a
  `[N, D]` memref and i32/i64 offsets, with a semantics contract
  ([docs/semantics.md](docs/semantics.md)): f32 accumulation for f32 and bf16,
  NaN propagation and `-inf` identities for max, empty segments, exact output
  shapes. A verifier rejects ill-formed programs.
- **Mapping.** `ragged-legalize` makes accumulation, identity and NaN handling
  explicit. `ragged-plan` assigns segments to three classes by length: up to
  32 rows one warp (a shuffle tree, no shared memory), up to 4,096 rows one
  128-thread block, longer segments 8 blocks with partial results and a combine
  kernel. Caller-supplied `min_len`/`max_len` hints prune classes at
  compile time; otherwise a classify kernel bins segments into per-class
  worklists on the device, and persistent workers for each class drain them.
  Each warp of the classify kernel ranks its lanes per class with one shuffle
  scan and reserves its queue positions with at most three atomics.
- **Scheduling.** Each persistent worker grid is one resident wave of that
  kernel, measured with the occupancy API when the module loads (on an RTX
  4090, 1,536 128-thread blocks or 3,072 warps, divided among feature tiles and
  split parts). The three classes touch disjoint segments, so their worker
  chains run on separate streams between a fork after classification and a
  join on the caller's stream, the CTA chain submitted first
  ([docs/classified-abi.md](docs/classified-abi.md)). Why each of these
  choices was made, and what it costs, is in
  [docs/design-tradeoffs.md](docs/design-tradeoffs.md).
- **Softmax.** `ragged-fuse` chooses the statistics schedule per segment
  class. A warp or CTA class computes the maximum and exponential sum in one
  online pass and normalizes in a second pass, in one kernel. Each of a split
  segment's eight blocks computes the same online statistics for its part, a
  combine kernel merges the eight, and a normalization kernel writes the
  output: two reads of the rows instead of three. The online statistics are
  compensated sums against a reference that trails the maximum by at most
  one, so their accuracy does not depend on segment length and they need no
  length hints. In-place softmax uses three passes in every class, the
  schedule that separates reads from writes. Optional f32 scale and additive
  mask.
- **Lowering.** `ragged-to-gpu` emits the kernels with feature-dimension
  tiling (`d_tile`), then the standard GPU → NVVM → PTX pipeline and `ptxas`
  produce a cubin. Compensated f32 summation keeps bf16 and f32 sums within the
  float64 oracle's tolerance. Each lane's row loop issues eight loads before
  folding them in row order, so eight loads are in flight per lane and the
  arithmetic, and therefore every result bit, is that of the one-row loop
  ([docs/optimizations.md](docs/optimizations.md)).
- **Artifacts and runtime.** The compiler writes a cubin and a manifest (kernel
  order, grids, workspace shapes, ptxas resource usage, toolchain provenance).
  The Python runtime validates tensors against the manifest, allocates one
  scratch buffer for all workspace on the consumer stream, and launches through
  a small C++ Driver API module (`runtime/`): one `dlopen`/`cuInit` per process,
  a launch plan per problem size that issues every kernel and the stream fork
  and join in one native call, and modules unloaded only after the work
  launched from them has finished (an event per stream that used them). A
  checked mode validates the offsets on the device before any main kernel runs.

## Results

**Correctness.** Numerical results in the GPU tests and the benchmark are
checked against a float64 NumPy oracle with dtype-specific tolerances
(`oracle/reference.py`). On an RTX 4090 (driver 595.99, CUDA 13.2); the GPU
and sanitizer runs are at the commit the benchmark measured
([results/tests/summary.txt](results/tests/summary.txt),
[results/sanitizer/summary.txt](results/sanitizer/summary.txt)):

| Suite | Result |
|---|---|
| GPU tests (`test/gpu`): every op, dtype, D in {1, 7, 16, 128, 1024}, i32/i64 offsets, four length distributions, empty segments, masks, in-place, checked offsets, streams, CUDA graph capture, Driver API faults; row pipelining and concurrent classes compared bitwise against the rolled, serial schedule; online and three-pass softmax in every class, including 65,536-row segments; class queues read back for counts, membership and per-warp order; both Triton mappings and the CUDA and CUB comparisons | 2,044 passed; 164 FBGEMM cases skip (optional package not installed) |
| compute-sanitizer memcheck, racecheck, initcheck and synccheck on the classified, mask, checked-offsets, scheduling and per-class softmax suites (595 tests, one process per suite and tool; racecheck skips eight stream-ordering tests that queue large matmuls) | 0 errors, 0 hazards |
| lit/FileCheck tests of the dialect, passes and lowering (`test/`) | 40 passed |
| host tests (`test/python`): cache, manifest validation, runtime contracts and stream schedule, benchmark harness, suite and summary | 1,706 passed |

**Speed.** Complete-call GPU time in µs: CUDA events around the call, so host
launch gaps count. Each process times 30 calls after 5 warm-up calls; a cell
is the median over 15 processes (seeds 711 to 713 of the length distribution,
5 repetitions each) of each process's median, with the 25th to 75th
percentile below it (`bench/suite.py`, `bench/summarize.py`,
[results/bench/](results/bench/README.md)). f32 unless noted; 1M rows in
16,384 segments at D=1, 262,144 rows in 4,096 segments at D=128; uniform
(64 rows), long-tail (Pareto) and bursty lengths. **Warm** L2 is back-to-back
calls; **cold** L2 overwrites a 288 MiB buffer and synchronizes before every
call. Clocks are not locked: each run records `nvidia-smi` state (SM clock
2,520 to 2,805 MHz over the suite), and the host ran other work at the same
time. Each method passes the oracle before it is timed. The comparisons
([baselines/README.md](baselines/README.md)):

- `ragged-naive`: this compiler with one block per segment (`strategy="naive"`).
- `torch.segment_reduce` from PyTorch 2.14 (sum and max).
- Triton 3.8: one program per segment, or segments cut into chunks of a
  tuned row count with a combine kernel; online softmax. The first call on
  each input times 8 (D=1) or 16 (D=128) configurations on that input and
  keeps the fastest, which favors Triton.
- CUB 3.5 `DeviceSegmentedReduce` at D=1; its softmax is CUB's max, an
  element-parallel exp, CUB's sum and an element-parallel division.
- A CUDA kernel with one 256-thread block per segment, threads over rows and
  up to 32 feature columns.
- NJT (`torch.nested`), compiled with `torch.compile(fullgraph=True)` and
  eager, built with the exact minimum and maximum sequence lengths.

Warm L2:

| Op | D | Lengths | ragged | ragged-naive | torch.segment_reduce | Triton (tuned) | CUB | CUDA, block per segment | NJT compiled | NJT eager |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sum | 1 | uniform | 49 <sub>48–50</sub> | 34 <sub>33–36</sub> | 15 <sub>15–15</sub> | 35 <sub>33–38</sub> | 31 <sub>31–31</sub> | 42 <sub>42–43</sub> | 30 <sub>29–30</sub> | 152 <sub>151–155</sub> |
| sum | 1 | long tail | 47 <sub>44–51</sub> | 35 <sub>34–84</sub> | 638 <sub>514–5,215</sub> | 40 <sub>35–74</sub> | 30 <sub>29–47</sub> | 45 <sub>42–167</sub> | 285 <sub>246–2,180</sub> | 10,209 <sub>9,470–10,883</sub> (10 runs) |
| sum | 1 | bursty | 39 <sub>39–40</sub> | 33 <sub>32–34</sub> | 235 <sub>229–240</sub> | 31 <sub>30–35</sub> | 26 <sub>26–26</sub> | 38 <sub>38–38</sub> | 76 <sub>76–77</sub> | 2,244 <sub>2,241–2,262</sub> |
| sum | 128 | uniform | 135 <sub>131–136</sub> | 167 <sub>166–168</sub> | 156 <sub>156–156</sub> | 169 <sub>168–169</sub> | – | 162 <sub>161–162</sub> | 165 <sub>164–166</sub> | 569 <sub>566–577</sub> |
| sum | 128 | long tail | 194 <sub>192–196</sub> | 238 <sub>224–246</sub> | 665 <sub>576–721</sub> | 186 <sub>186–187</sub> | – | 1,174 <sub>1,008–1,269</sub> | 1,115 <sub>1,015–1,122</sub> | fails |
| sum | 128 | bursty | 181 <sub>180–181</sub> | 172 <sub>171–172</sub> | 481 <sub>478–485</sub> | 173 <sub>172–174</sub> | – | 914 <sub>913–916</sub> | 758 <sub>757–818</sub> | fails |
| max | 1 | uniform | 49 <sub>48–49</sub> | 34 <sub>34–34</sub> | 15 <sub>15–15</sub> | 31 <sub>31–32</sub> | 33 <sub>32–33</sub> | 42 <sub>42–42</sub> | 30 <sub>30–31</sub> | 120 <sub>118–122</sub> |
| max | 1 | long tail | 44 <sub>44–44</sub> | 37 <sub>34–53</sub> | 460 <sub>356–3,801</sub> | 37 <sub>31–53</sub> | 32 <sub>31–54</sub> | 43 <sub>41–163</sub> | 313 <sub>267–2,308</sub> | 9,849 <sub>8,847–10,850</sub> (10 runs) |
| max | 1 | bursty | 39 <sub>39–40</sub> | 33 <sub>33–35</sub> | 159 <sub>121–159</sub> | 30 <sub>30–31</sub> | 27 <sub>26–28</sub> | 39 <sub>38–41</sub> | 83 <sub>83–89</sub> | 2,215 <sub>2,211–2,218</sub> |
| max | 128 | uniform | 133 <sub>132–135</sub> | 168 <sub>167–168</sub> | 156 <sub>156–156</sub> | 168 <sub>168–172</sub> | – | 161 <sub>161–162</sub> | 164 <sub>164–184</sub> | 541 <sub>537–545</sub> |
| max | 128 | long tail | 186 <sub>183–189</sub> | 199 <sub>196–200</sub> | 325 <sub>292–342</sub> | 183 <sub>182–184</sub> | – | 1,146 <sub>1,005–1,236</sub> | 1,143 <sub>1,137–1,187</sub> | fails |
| max | 128 | bursty | 180 <sub>180–181</sub> | 168 <sub>167–168</sub> | 266 <sub>222–267</sub> | 172 <sub>170–176</sub> | – | 903 <sub>902–905</sub> | 914 <sub>911–924</sub> | fails |
| softmax | 1 | uniform | 55 <sub>55–56</sub> | 36 <sub>36–36</sub> | – | 47 <sub>47–48</sub> | 76 <sub>75–76</sub> | 60 <sub>60–60</sub> | 48 <sub>47–49</sub> | 172 <sub>169–175</sub> |
| softmax | 1 | long tail | 57 <sub>55–83</sub> | 57 <sub>51–263</sub> | – | 66 <sub>63–94</sub> | 79 <sub>72–110</sub> | 84 <sub>75–497</sub> | 681 <sub>564–5,539</sub> | 12,599 <sub>11,107–13,554</sub> (10 runs) |
| softmax | 1 | bursty | 47 <sub>46–48</sub> | 37 <sub>36–38</sub> | – | 40 <sub>37–40</sub> | 62 <sub>62–63</sub> | 55 <sub>54–55</sub> | 169 <sub>168–171</sub> | 2,652 <sub>2,652–2,661</sub> |
| softmax | 128 | uniform | 312 <sub>310–314</sub> | 336 <sub>336–337</sub> | – | 315 <sub>314–316</sub> | – | 314 <sub>314–315</sub> | 460 <sub>458–461</sub> | 1,192 <sub>1,188–1,200</sub> |
| softmax | 128 | long tail | 437 <sub>426–444</sub> | 575 <sub>528–601</sub> | – | 394 <sub>385–396</sub> | – | 2,456 <sub>2,071–2,703</sub> | 4,442 <sub>3,732–4,691</sub> | fails |
| softmax | 128 | bursty | 509 <sub>505–510</sub> | 533 <sub>526–545</sub> | – | 480 <sub>478–484</sub> | – | 1,918 <sub>1,914–1,949</sub> | 3,435 <sub>3,432–3,443</sub> | fails |
| softmax bf16 | 128 | long tail | 265 <sub>261–275</sub> | 444 <sub>397–471</sub> | – | 224 <sub>223–229</sub> | – | 2,867 <sub>2,525–3,140</sub> | 4,283 <sub>3,797–4,584</sub> | fails |

Cold L2:

| Op | D | Lengths | ragged | ragged-naive | torch.segment_reduce | Triton (tuned) | CUB | CUDA, block per segment | NJT compiled | NJT eager |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sum | 1 | uniform | 54 <sub>53–55</sub> | 37 <sub>36–38</sub> | 22 <sub>22–22</sub> | 39 <sub>36–43</sub> | 36 <sub>36–38</sub> | 47 <sub>46–47</sub> | 35 <sub>33–37</sub> | 155 <sub>155–162</sub> |
| sum | 1 | long tail | 57 <sub>54–65</sub> | 45 <sub>44–134</sub> | 1,213 <sub>981–9,557</sub> | 62 <sub>45–120</sub> | 38 <sub>36–76</sub> | 63 <sub>61–320</sub> | 290 <sub>249–2,194</sub> | 9,873 <sub>8,847–10,868</sub> (10 runs) |
| sum | 1 | bursty | 51 <sub>50–52</sub> | 39 <sub>38–40</sub> | 526 <sub>524–531</sub> | 39 <sub>35–40</sub> | 33 <sub>32–34</sub> | 43 <sub>42–43</sub> | 81 <sub>80–82</sub> | 2,255 <sub>2,249–2,274</sub> |
| sum | 128 | uniform | 228 <sub>227–230</sub> | 218 <sub>217–219</sub> | 207 <sub>207–207</sub> | 222 <sub>220–224</sub> | – | 214 <sub>213–215</sub> | 192 <sub>191–198</sub> | 625 <sub>619–636</sub> |
| sum | 128 | long tail | 251 <sub>249–254</sub> | 291 <sub>278–300</sub> | 719 <sub>624–779</sub> | 238 <sub>237–240</sub> | – | 1,227 <sub>1,060–1,321</sub> | 1,120 <sub>1,032–1,131</sub> | fails |
| sum | 128 | bursty | 231 <sub>231–232</sub> | 229 <sub>227–230</sub> | 536 <sub>533–539</sub> | 230 <sub>227–232</sub> | – | 963 <sub>962–964</sub> | 763 <sub>760–829</sub> | fails |
| max | 1 | uniform | 54 <sub>53–56</sub> | 37 <sub>36–38</sub> | 18 <sub>18–19</sub> | 36 <sub>34–41</sub> | 37 <sub>37–38</sub> | 47 <sub>45–48</sub> | 37 <sub>33–39</sub> | 124 <sub>121–127</sub> |
| max | 1 | long tail | 54 <sub>52–58</sub> | 43 <sub>40–81</sub> | 785 <sub>621–6,291</sub> | 57 <sub>45–86</sub> | 42 <sub>38–82</sub> | 65 <sub>58–316</sub> | 314 <sub>273–2,323</sub> | 9,832 <sub>8,838–10,845</sub> (10 runs) |
| max | 1 | bursty | 51 <sub>50–52</sub> | 39 <sub>38–39</sub> | 228 <sub>222–230</sub> | 37 <sub>36–40</sub> | 34 <sub>33–37</sub> | 44 <sub>42–46</sub> | 89 <sub>86–94</sub> | 2,222 <sub>2,218–2,227</sub> |
| max | 128 | uniform | 228 <sub>227–229</sub> | 218 <sub>216–218</sub> | 206 <sub>206–206</sub> | 221 <sub>220–223</sub> | – | 213 <sub>213–214</sub> | 191 <sub>190–207</sub> | 593 <sub>591–600</sub> |
| max | 128 | long tail | 242 <sub>239–244</sub> | 249 <sub>245–252</sub> | 377 <sub>343–394</sub> | 235 <sub>233–240</sub> | – | 1,198 <sub>1,053–1,285</sub> | 1,164 <sub>1,161–1,223</sub> | fails |
| max | 128 | bursty | 231 <sub>231–232</sub> | 223 <sub>223–225</sub> | 289 <sub>266–290</sub> | 226 <sub>225–228</sub> | – | 952 <sub>951–954</sub> | 921 <sub>914–934</sub> | fails |
| softmax | 1 | uniform | 67 <sub>66–68</sub> | 41 <sub>41–42</sub> | – | 49 <sub>48–51</sub> | 78 <sub>78–79</sub> | 61 <sub>61–63</sub> | 52 <sub>49–52</sub> | 172 <sub>170–178</sub> |
| softmax | 1 | long tail | 71 <sub>69–103</sub> | 80 <sub>72–356</sub> | – | 71 <sub>67–98</sub> | 82 <sub>74–112</sub> | 84 <sub>76–498</sub> | 684 <sub>565–5,268</sub> | 12,290 <sub>11,022–13,565</sub> (10 runs) |
| softmax | 1 | bursty | 64 <sub>62–66</sub> | 46 <sub>44–47</sub> | – | 41 <sub>40–43</sub> | 65 <sub>64–67</sub> | 56 <sub>55–57</sub> | 171 <sub>170–173</sub> | 2,658 <sub>2,541–2,663</sub> |
| softmax | 128 | uniform | 349 <sub>348–351</sub> | 344 <sub>343–346</sub> | – | 325 <sub>324–326</sub> | – | 332 <sub>331–332</sub> | 479 <sub>479–483</sub> | 1,222 <sub>1,219–1,241</sub> |
| softmax | 128 | long tail | 458 <sub>454–470</sub> | 597 <sub>552–623</sub> | – | 415 <sub>406–416</sub> | – | 2,464 <sub>2,088–2,746</sub> | 4,466 <sub>3,729–4,479</sub> | fails |
| softmax | 128 | bursty | 516 <sub>515–517</sub> | 542 <sub>536–595</sub> | – | 494 <sub>492–495</sub> | – | 1,941 <sub>1,937–1,971</sub> | 3,426 <sub>3,158–3,432</sub> | fails |
| softmax bf16 | 128 | long tail | 304 <sub>301–319</sub> | 472 <sub>425–500</sub> | – | 251 <sub>250–253</sub> | – | 2,867 <sub>2,495–3,151</sub> | 3,906 <sub>3,813–4,366</sub> | fails |

On the long tail the seed sets the longest segment: 300,412 rows for seed
711 at D=1 and 29,049 and 35,765 for the others, so the upper quartile of a
D=1 long-tail cell is mostly seed 711. `summary.json` lists every seed's
median.

What the numbers say:

- **ragged is ahead at D=128 with uniform lengths and on the D=1 long-tail
  softmax.** Warm, sum and max take 135 and 133 µs, against 156 for
  `torch.segment_reduce` and 162 for the CUDA kernel; softmax 312 against 314
  for the CUDA kernel and 315 for Triton. The D=1 long-tail softmax takes
  57 µs against 66 for Triton and 79 for CUB. With seed 711's 300,412-row
  segment ragged is the fastest D=1 long-tail softmax (83 µs, Triton 95, CUB
  111) and max (45 µs, CUB and Triton 53).
- **Tuned Triton wins the skewed D=128 cases.** It takes 2 to 16% less time
  than ragged on the long-tail and bursty sum, max and softmax (bursty sum 173
  against 181 µs, long-tail softmax 394 against 437, bf16 224 against 265).
- **At D=1 the libraries are faster.** `torch.segment_reduce` takes 15 µs on
  uniform sum and max, where ragged takes 49; CUB takes 26 to 32 µs on the
  long-tail and bursty sum and max, ragged 39 to 47; Triton's softmax takes 40
  and 47 µs on bursty and uniform lengths, ragged 47 and 55. ragged pays for
  device classification on every call: an init and a classify kernel, three
  class chains with a stream fork and join, and about 24 µs of host time
  (`results/bench/host.txt`).
- **ragged's own block-per-segment mode beats its default in 11 of 19 warm
  workloads** (13 cold): every D=1 workload (the long-tail softmax by 0.2 µs)
  and the bursty D=128 sum and max (172 against 181 µs, 168 against 180).
  Its D=1 long-tail times depend on the longest segment: with seed 711 the
  naive sum takes 85 µs against 53 for the default, with the other seeds 34
  against 45. Nothing chooses between the two automatically.
- **Length hints help at D=1 and hurt at D=128.** Exact bounds make a uniform
  workload compile to one block-per-segment kernel: 34 µs instead of 49 for
  the D=1 sum and 36 instead of 55 for the D=1 softmax, but 168 instead of 135
  for the D=128 sum and 336 instead of 312 for the D=128 softmax. On the long
  tail the bounds prune no class.
- **Cold L2 removes ragged's D=128 uniform lead.** Reading the 128 MiB input
  of the D=128 uniform sum takes 133 µs at the 4090's 1.01 TB/s. Warm, the
  call takes 135 µs and its CTA kernel 105 µs (`results/bench/profile.txt`),
  less than that bound, so part of the input comes from L2. Cold, the call
  takes 228 µs (0.59 TB/s), slower than compiled NJT (192),
  `torch.segment_reduce` (207), the CUDA kernel (214), ragged-naive (218) and
  Triton (222).
- **NJT** with exact sequence lengths compiles to fast code for uniform
  lengths (30 µs for the D=1 sum, the fastest cold D=128 uniform sum and max)
  and is 2 to 16 times slower than ragged on skewed ones. Eager NJT pads to the longest
  segment: it fails on an int32 index limit for skewed D=128 inputs and for
  seed 711 at D=1, and takes 0.1 to 12.6 ms elsewhere.

**Ablations**, ragged only, warm, µs (`results/bench/runs/ablation/`), each
turning off one optimization in the same build:
`--disable-optimization online_fusion` (three passes in every class),
`--disable-optimization row_pipelining` and `--serial-classes`:

| Workload | ragged | three passes in every class | rows not pipelined | classes serial |
|---|---:|---:|---:|---:|
| sum, D=1, uniform | 49 <sub>48–50</sub> | – | 50 <sub>49–51</sub> | 56 <sub>55–57</sub> |
| sum, D=1, long tail | 47 <sub>44–51</sub> | – | 45 <sub>44–51</sub> | 50 <sub>49–52</sub> |
| sum, D=1, bursty | 39 <sub>39–40</sub> | – | 40 <sub>39–40</sub> | 43 <sub>43–43</sub> |
| sum, D=128, uniform | 135 <sub>131–136</sub> | – | 133 <sub>130–133</sub> | 139 <sub>138–140</sub> |
| sum, D=128, long tail | 194 <sub>192–196</sub> | – | 216 <sub>210–221</sub> | 211 <sub>211–214</sub> |
| sum, D=128, bursty | 181 <sub>180–181</sub> | – | 181 <sub>181–182</sub> | 183 <sub>182–184</sub> |
| max, D=1, uniform | 49 <sub>48–49</sub> | – | 50 <sub>49–50</sub> | 56 <sub>56–57</sub> |
| max, D=1, long tail | 44 <sub>44–44</sub> | – | 45 <sub>44–49</sub> | 50 <sub>49–50</sub> |
| max, D=1, bursty | 39 <sub>39–40</sub> | – | 40 <sub>39–41</sub> | 43 <sub>43–44</sub> |
| max, D=128, uniform | 133 <sub>132–135</sub> | – | 132 <sub>131–135</sub> | 142 <sub>141–145</sub> |
| max, D=128, long tail | 186 <sub>183–189</sub> | – | 212 <sub>207–217</sub> | 200 <sub>197–201</sub> |
| max, D=128, bursty | 180 <sub>180–181</sub> | – | 182 <sub>181–183</sub> | 183 <sub>182–183</sub> |
| softmax, D=1, uniform | 55 <sub>55–56</sub> | 80 <sub>80–81</sub> | 53 <sub>53–54</sub> | 61 <sub>61–62</sub> |
| softmax, D=1, long tail | 57 <sub>55–83</sub> | 69 <sub>67–89</sub> | 57 <sub>54–107</sub> | 64 <sub>63–93</sub> |
| softmax, D=1, bursty | 47 <sub>46–48</sub> | 59 <sub>59–60</sub> | 48 <sub>47–48</sub> | 53 <sub>53–53</sub> |
| softmax, D=128, uniform | 312 <sub>310–314</sub> | 636 <sub>635–638</sub> | 311 <sub>310–314</sub> | 325 <sub>323–326</sub> |
| softmax, D=128, long tail | 437 <sub>426–444</sub> | 653 <sub>644–666</sub> | 602 <sub>593–632</sub> | 489 <sub>474–497</sub> |
| softmax, D=128, bursty | 509 <sub>505–510</sub> | 665 <sub>662–666</sub> | 496 <sub>495–497</sub> | 511 <sub>510–512</sub> |
| softmax bf16, D=128, long tail | 265 <sub>261–275</sub> | 351 <sub>342–365</sub> | – | – |

Online statistics are faster in every softmax workload: three passes take 19
to 104% longer. Concurrent class chains are faster in all 18 f32 workloads:
serial chains take 0.5 to 15% longer. Row pipelining helps the long-tail D=128 workloads
most (sum 216 against 194 µs, max 212 against 186, softmax 602 against 437)
and is 0.4 to 3.6% slower in seven others: the uniform D=1 softmax, D=1
long-tail sum and softmax, uniform D=128 sum, max and softmax, and bursty
D=128 softmax.

**Length hints** (`--length-hints`, exact bounds), warm, µs:

| Workload | ragged | ragged with exact length hints | classes with hints | ragged-naive |
|---|---:|---:|---|---:|
| sum, D=1, uniform | 49 <sub>48–50</sub> | 34 <sub>33–35</sub> | cta | 34 <sub>33–36</sub> |
| sum, D=1, long tail | 47 <sub>44–51</sub> | 45 <sub>44–48</sub> | warp+cta+split | 35 <sub>34–84</sub> |
| sum, D=128, uniform | 135 <sub>131–136</sub> | 168 <sub>167–169</sub> | cta | 167 <sub>166–168</sub> |
| sum, D=128, long tail | 194 <sub>192–196</sub> | 196 <sub>193–196</sub> | warp+cta+split | 238 <sub>224–246</sub> |
| softmax, D=1, uniform | 55 <sub>55–56</sub> | 36 <sub>35–36</sub> | cta | 36 <sub>36–36</sub> |
| softmax, D=128, uniform | 312 <sub>310–314</sub> | 336 <sub>335–337</sub> | cta | 336 <sub>336–337</sub> |

**Kernel traces** (`results/bench/profile.txt`, seed 711, warm). With uniform
lengths at D=128 every 64-row segment is in the CTA class, whose online
softmax kernel takes 285 µs to read and write 128 MiB each: 0.94 TB/s, which
is possible only because most of its second read of each 32 KiB segment
hits in cache. On the D=128 long-tail softmax the CTA, warp and split
classes take 317, 64 and 62 µs run one after another; concurrently the call's
GPU span is 420 µs instead of 471, although every kernel runs longer while
they share the GPU. On the D=1 long-tail sum the classify kernel takes 7 µs
and the GPU span is 27 µs of a 53 µs call; the host time of a call is 22 to
28 µs (`results/bench/host.txt`).

## Build and run

LLVM/MLIR 23.1.3 (`scripts/bootstrap_llvm.sh` builds it), CUDA 13.2 with
`ptxas`, CMake, Ninja, Python 3.12 with PyTorch 2.14 (CUDA 13.2 build) and
nanobind.

```sh
scripts/bootstrap_llvm.sh          # builds LLVM/MLIR in .deps/llvm-build, writes .deps/ragged-source-sdk.cmake
cmake -G Ninja -C .deps/ragged-source-sdk.cmake -S . -B build
ninja -C build && ninja -C build check-ragged          # compiler and lit tests
cmake -S runtime -B build-runtime && cmake --build build-runtime && cmake --install build-runtime --prefix python
python -m pytest                                       # host tests (test/python); a CPU PyTorch is enough
PYTHONPATH=python RAGGED_COMPILE=build/bin/ragged-compile python -m pytest test/gpu
PYTHONPATH=python:. python -m bench.run --op softmax --dtype bf16 --d 128 \
  --distribution long_tail --segments 4096 --total-elements 262144 \
  --output build/bench/softmax-bf16-d128-long_tail
```

With the official LLVM 23.1.3 Linux SDK instead of a source build (how the
published results were built), configure with
`-DMLIR_DIR=<sdk>/lib/cmake/mlir` and point
`-DRAGGED_LLVM_TEST_TOOLS_DIR` at a directory holding `FileCheck` and `not`,
which the SDK does not ship ([docs/provenance.md](docs/provenance.md)).

`bench.run` refuses an output directory that is not empty, so the command
above writes a new one under `build/`; compare it with
`results/bench/runs/main/softmax-bf16-d128-long_tail/seed-711/`. The
`cuda-naive` and `cub` methods need the comparison library that
`baselines/README.md` shows how to build. The whole suite (about 1,020
processes) and its summary and chart:

```sh
python -m bench.suite --output build/runs --lock /var/tmp/gpu.lock
python -m bench.summarize build/runs --output build/summary.json
python -m bench.chart build/summary.json --output build/speedup.svg
```

## Limitations

- sm_89 only; one device per program; forward only (no autograd).
- Validated mode trusts the offsets; checked mode validates them on the device
  at the cost of a stream synchronization before the main kernels.
- Segment-class bounds (32 and 4,096 rows) and the 8-way split are fixed in the
  compiler and the classified ABI, not tuned per GPU or per D.
- At small sizes the host still costs 22 to 28 µs per call
  (`results/bench/host.txt`): tensor validation, descriptor construction and
  the scratch allocation in Python, then one native call.
- The online fold is one dependency chain per lane. When few segments are
  long, as for the split class of the long-tail inputs, its kernel is bound by
  latency rather than bandwidth. Recentering once per batch of eight pipelined
  rows would shorten the chain, but would change results against the rolled
  loop, which row pipelining now reproduces bit for bit.
- The default path does not choose between device classification and one
  block per segment; `ragged-naive` is faster in 11 of the 19 workloads with a
  warm L2, and exact length hints that select it make D=128 slower.
- The fixed cost of classification makes ragged slower than
  `torch.segment_reduce`, CUB and Triton at D=1 except on the softmax of very
  long segments, and tuned Triton is 2 to 16% faster on skewed D=128 inputs.
- With a cold L2 the D=128 uniform sum reads its input at 0.59 TB/s, well
  below the 1.01 TB/s DRAM bandwidth; the warm-L2 times near that bound come
  partly from L2.
- Persistent workers take queue positions with a fixed stride, so their speed
  depends on the order classification leaves in the queues and on which SM
  each worker lands. In separate A/B runs (not part of `results/`), the
  warp-ordered queues made the long-tail D=128 sum and max about 6% slower
  than per-segment atomics, and submitting the split chain before the CTA
  chain made the uniform D=128 sum take 176 µs instead of 133
  ([docs/design-tradeoffs.md](docs/design-tradeoffs.md)).
- One GPU (an RTX 4090 that also drives a desktop), unlocked clocks, and a
  host running other work; synthetic length distributions plus three seeds.
  The Triton reference is tuned on each benchmark input, but FBGEMM and
  `torch_scatter` are not measured.

## Layout

| Path | What it is |
|---|---|
| `include/Ragged/`, `lib/IR/` | the dialect, ops and verifiers |
| `lib/Transforms/` | legalization and mapping selection |
| `lib/Conversion/ToGPU.cpp` | lowering to GPU kernels |
| `tools/` | `ragged-opt` and `ragged-compile` |
| `python/ragged/` | frontend, artifact cache, manifest validation, runtime, PyTorch op registration |
| `runtime/` | CUDA Driver API module loader and launcher (nanobind) |
| `oracle/` | float64 reference and test-case minimizer |
| `baselines/`, `bench/` | comparison implementations and the benchmark harness |
| `test/` | lit tests (`IR`, `Transforms`, `Conversion`), host tests, GPU tests |
| `docs/` | semantics, ABIs, optimizations, error model, design trade-offs |
| `results/` | benchmark runs and summary, test and sanitizer summaries |
