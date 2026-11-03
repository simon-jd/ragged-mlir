# Benchmark results

Measured on an RTX 4090 (driver 595.99, CUDA 13.2). Every `results.json`
under `runs/` records the commit it ran at under `source`: `f0bd690` for the
main, hints and ablation groups, `a7a1428` for `heldout`, `2d69970` for
`compare` (five of its workloads re-run at `6b0b842`, below) and `6b0b842`
for `heldout2` (the commits between them do not change the generated
kernels). The `ragged` (adaptive) method ran cost model version 1 in every
group but `heldout2` and those five `compare` workloads, which ran version 2.

- `runs/<group>/<case>/seed-<seed>/rep-<n>/results.json`: one `bench.run`
  process each, from `python -m bench.suite --lock /var/tmp/rtx4090.lock
  --lock-batch 600` (main: 19 workloads with every method, 6 with length
  hints, 52 ablations; seeds 711 to 713; 5 repetitions; warm and cold L2),
  and the `compare` and `heldout` groups (`--lock-batch 540`) and the
  `heldout2` group and the five re-run `compare` workloads (`--lock-batch
  480`, below). The GPU was shared with other projects, so the suite held
  the lock for up to 10 minutes at a time and waited between batches.
  `runs/suite.log` (and `runs/compare/suite.log`, `runs/heldout/suite.log`,
  `runs/heldout2/suite.log`) lists every
  process's exit code and duration; exit 1 means a method failed (the
  tracebacks are beside the results), which here is always eager NJT.
- `summary.json`: `python -m bench.summarize results/bench/runs --output
  results/bench/summary.json`. For each method and L2 state it gives the
  median over the 15 runs of each run's median call time, the 25th and 75th
  percentiles, each seed's median, and for `ragged-auto` the schedule each
  seed's runs chose.
- `calibration/`: the cost model's fits and evaluations. Version 1: the 190
  runs it was fitted to (`main/`, seeds 711 and 712, `ragged-classified` and
  `ragged-naive`, at commit `686d9ce`), the model (`cost_model-v1.json`), its
  fit (`fit-v1.txt`), `evaluation.txt` (its choices on the main runs) and
  `heldout-shapes.txt` (on the first held-out shapes). Version 2, the shipped
  `python/ragged/cost_model.json`: its fit on `runs/main` and `runs/heldout`
  (`fit.txt`, `python -m bench.calibrate fit results/bench/runs --groups main
  heldout`), both versions' choices on those runs (`refit-training.txt`) and
  the evaluation on the second held-out set (`heldout2.txt`).
- `runs/heldout/` and `runs/heldout2/`: the two held-out sets, fixed in
  `bench/suite.py` (`HELDOUT_SHAPES`, `HELDOUT2_SHAPES`) before the model
  they evaluate was run on them (below).
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

## Every comparison, with CUDA-graph replays

`python -m bench.suite --groups compare` at commit `2d69970`, with the
bursty and long-tail D=128 sum and max and the long-tail D=1 softmax re-run
at `6b0b842` (`--only`, 75 runs that replace the earlier ones), whose
refitted cost model chooses differently there: the 19 main
workloads, 285 runs (seeds 711 to 713, 5 repetitions), every comparison
including CUB at D=128, cuSPARSE and FBGEMM, and a CUDA-graph replay of each
method (`--cuda-graph`). µs, median over runs with the 25th to 75th
percentile; – is not applicable. The README's summary table and chart come
from these runs.

Warm L2:

| Op | D | Lengths | ragged | ragged-naive | torch.segment_reduce | NJT compiled | CUDA, block per segment | CUB | cuSPARSE | Triton (tuned) | FBGEMM |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sum | 1 | uniform | 34 <sub>33–34</sub> | 33 <sub>33–34</sub> | 15 <sub>15–15</sub> | 27 <sub>26–28</sub> | 42 <sub>42–42</sub> | 30 <sub>30–30</sub> | 23 <sub>22–23</sub> | 33 <sub>32–33</sub> | 27 <sub>27–28</sub> |
| sum | 1 | long tail | 35 <sub>34–47</sub> | 34 <sub>33–83</sub> | 638 <sub>514–5,024</sub> | 310 <sub>259–2,151</sub> | 46 <sub>43–164</sub> | 30 <sub>30–47</sub> | 22 <sub>21–23</sub> | 34 <sub>33–71</sub> | 55 <sub>52–330</sub> |
| sum | 1 | bursty | 33 <sub>33–34</sub> | 32 <sub>32–33</sub> | 235 <sub>230–240</sub> | 80 <sub>79–80</sub> | 41 <sub>41–41</sub> | 26 <sub>26–26</sub> | 23 <sub>23–23</sub> | 31 <sub>30–31</sub> | 26 <sub>25–27</sub> |
| sum | 128 | uniform | 136 <sub>131–137</sub> | 167 <sub>167–167</sub> | 155 <sub>155–155</sub> | 166 <sub>165–166</sub> | 161 <sub>161–161</sub> | 1,417 <sub>1,415–1,418</sub> | 170 <sub>169–170</sub> | 167 <sub>167–168</sub> | – |
| sum | 128 | long tail | 191 <sub>188–194</sub> | 195 <sub>189–199</sub> | 661 <sub>570–720</sub> | 1,054 <sub>1,009–1,114</sub> | 1,188 <sub>1,008–1,280</sub> | 1,351 <sub>1,328–1,429</sub> | 169 <sub>169–170</sub> | 185 <sub>183–186</sub> | – |
| sum | 128 | bursty | 181 <sub>181–182</sub> | 168 <sub>168–169</sub> | 480 <sub>476–483</sub> | 756 <sub>754–816</sub> | 913 <sub>911–914</sub> | 1,250 <sub>1,247–1,260</sub> | 169 <sub>169–169</sub> | 170 <sub>169–172</sub> | – |
| max | 1 | uniform | 34 <sub>33–34</sub> | 33 <sub>32–33</sub> | 15 <sub>15–15</sub> | 28 <sub>27–29</sub> | 42 <sub>41–42</sub> | 32 <sub>31–32</sub> | – | 31 <sub>30–31</sub> | – |
| max | 1 | long tail | 34 <sub>34–45</sub> | 33 <sub>33–52</sub> | 459 <sub>353–3,944</sub> | 347 <sub>289–2,275</sub> | 46 <sub>43–160</sub> | 33 <sub>32–51</sub> | – | 31 <sub>31–49</sub> | – |
| max | 1 | bursty | 34 <sub>33–34</sub> | 33 <sub>32–33</sub> | 158 <sub>113–159</sub> | 86 <sub>85–87</sub> | 41 <sub>41–41</sub> | 27 <sub>27–28</sub> | – | 30 <sub>29–30</sub> | – |
| max | 128 | uniform | 134 <sub>131–139</sub> | 167 <sub>167–168</sub> | 156 <sub>156–156</sub> | 166 <sub>166–166</sub> | 161 <sub>161–161</sub> | 1,507 <sub>1,507–1,507</sub> | – | 168 <sub>167–168</sub> | – |
| max | 128 | long tail | 181 <sub>178–182</sub> | 192 <sub>187–196</sub> | 322 <sub>292–341</sub> | 1,138 <sub>1,137–1,342</sub> | 1,133 <sub>983–1,226</sub> | 1,405 <sub>1,404–1,405</sub> | – | 180 <sub>179–181</sub> | – |
| max | 128 | bursty | 181 <sub>181–182</sub> | 168 <sub>167–169</sub> | 266 <sub>222–267</sub> | 910 <sub>907–927</sub> | 899 <sub>849–902</sub> | 1,201 <sub>1,198–1,225</sub> | – | 169 <sub>169–170</sub> | – |
| softmax | 1 | uniform | 36 <sub>36–37</sub> | 35 <sub>35–35</sub> | – | 45 <sub>44–46</sub> | 60 <sub>60–60</sub> | 75 <sub>75–76</sub> | – | 46 <sub>46–47</sub> | 74 <sub>74–74</sub> |
| softmax | 1 | long tail | 55 <sub>55–82</sub> | 56 <sub>50–262</sub> | – | 758 <sub>632–5,666</sub> | 92 <sub>81–493</sub> | 77 <sub>76–109</sub> | – | 65 <sub>62–93</sub> | 386 <sub>326–2,697</sub> |
| softmax | 1 | bursty | 37 <sub>36–39</sub> | 36 <sub>35–37</sub> | – | 173 <sub>173–174</sub> | 59 <sub>59–59</sub> | 67 <sub>67–67</sub> | – | 37 <sub>37–41</sub> | 105 <sub>104–177</sub> |
| softmax | 128 | uniform | 311 <sub>310–312</sub> | 321 <sub>321–321</sub> | – | 456 <sub>456–457</sub> | 314 <sub>314–314</sub> | 4,010 <sub>4,008–4,022</sub> | – | 314 <sub>314–314</sub> | 1,951 <sub>1,950–1,951</sub> |
| softmax | 128 | long tail | 403 <sub>402–411</sub> | 485 <sub>461–506</sub> | – | 4,455 <sub>3,721–4,705</sub> | 2,438 <sub>2,123–2,705</sub> | 3,767 <sub>3,762–4,038</sub> | – | 392 <sub>390–394</sub> | 2,523 <sub>2,513–2,527</sub> |
| softmax | 128 | bursty | 504 <sub>504–505</sub> | 504 <sub>502–506</sub> | – | 3,447 <sub>3,442–3,461</sub> | 1,935 <sub>1,929–1,942</sub> | 3,637 <sub>3,620–3,644</sub> | – | 479 <sub>477–481</sub> | 2,737 <sub>2,701–2,742</sub> |
| softmax bf16 | 128 | long tail | 304 <sub>299–306</sub> | 458 <sub>426–469</sub> | – | 4,267 <sub>3,785–4,790</sub> | 2,876 <sub>2,482–3,152</sub> | 4,160 <sub>4,159–4,164</sub> | – | 221 <sub>220–222</sub> | 2,941 <sub>2,930–2,946</sub> |

Cold L2:

| Op | D | Lengths | ragged | ragged-naive | torch.segment_reduce | NJT compiled | CUDA, block per segment | CUB | cuSPARSE | Triton (tuned) | FBGEMM |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sum | 1 | uniform | 37 <sub>36–37</sub> | 36 <sub>36–38</sub> | 22 <sub>22–22</sub> | 30 <sub>30–31</sub> | 45 <sub>45–46</sub> | 35 <sub>34–35</sub> | 44 <sub>44–45</sub> | 35 <sub>34–35</sub> | 32 <sub>31–32</sub> |
| sum | 1 | long tail | 44 <sub>43–66</sub> | 44 <sub>42–131</sub> | 1,209 <sub>979–9,547</sub> | 312 <sub>262–2,161</sub> | 64 <sub>59–315</sub> | 37 <sub>36–75</sub> | 44 <sub>44–45</sub> | 44 <sub>42–79</sub> | 82 <sub>77–520</sub> |
| sum | 1 | bursty | 38 <sub>37–39</sub> | 37 <sub>37–38</sub> | 521 <sub>518–525</sub> | 82 <sub>82–84</sub> | 45 <sub>44–46</sub> | 33 <sub>33–34</sub> | 45 <sub>44–46</sub> | 36 <sub>35–37</sub> | 35 <sub>32–36</sub> |
| sum | 128 | uniform | 230 <sub>229–230</sub> | 218 <sub>217–218</sub> | 206 <sub>206–207</sub> | 187 <sub>187–188</sub> | 212 <sub>212–213</sub> | 1,446 <sub>1,444–1,447</sub> | 219 <sub>219–219</sub> | 218 <sub>218–219</sub> | – |
| sum | 128 | long tail | 251 <sub>247–253</sub> | 247 <sub>241–250</sub> | 719 <sub>622–780</sub> | 1,087 <sub>1,016–1,121</sub> | 1,240 <sub>1,063–1,333</sub> | 1,380 <sub>1,360–1,458</sub> | 219 <sub>219–220</sub> | 235 <sub>233–236</sub> | – |
| sum | 128 | bursty | 234 <sub>233–234</sub> | 223 <sub>223–224</sub> | 535 <sub>533–538</sub> | 757 <sub>755–826</sub> | 962 <sub>959–963</sub> | 1,280 <sub>1,276–1,289</sub> | 219 <sub>219–219</sub> | 226 <sub>225–226</sub> | – |
| max | 1 | uniform | 37 <sub>37–37</sub> | 36 <sub>36–37</sub> | 18 <sub>18–19</sub> | 31 <sub>30–31</sub> | 45 <sub>45–45</sub> | 37 <sub>36–37</sub> | – | 34 <sub>33–34</sub> | – |
| max | 1 | long tail | 40 <sub>39–60</sub> | 39 <sub>39–79</sub> | 782 <sub>618–6,240</sub> | 349 <sub>293–2,293</sub> | 63 <sub>58–312</sub> | 39 <sub>38–78</sub> | – | 43 <sub>42–77</sub> | – |
| max | 1 | bursty | 39 <sub>38–39</sub> | 38 <sub>38–39</sub> | 226 <sub>213–226</sub> | 89 <sub>88–91</sub> | 45 <sub>44–46</sub> | 33 <sub>33–34</sub> | – | 35 <sub>33–36</sub> | – |
| max | 128 | uniform | 230 <sub>229–230</sub> | 218 <sub>218–219</sub> | 205 <sub>205–206</sub> | 188 <sub>187–191</sub> | 212 <sub>212–213</sub> | 1,537 <sub>1,536–1,537</sub> | – | 218 <sub>218–219</sub> | – |
| max | 128 | long tail | 237 <sub>236–240</sub> | 244 <sub>239–248</sub> | 375 <sub>343–394</sub> | 1,160 <sub>1,159–1,346</sub> | 1,176 <sub>1,030–1,279</sub> | 1,437 <sub>1,436–1,438</sub> | – | 231 <sub>230–232</sub> | – |
| max | 128 | bursty | 233 <sub>233–234</sub> | 223 <sub>222–223</sub> | 288 <sub>265–289</sub> | 918 <sub>909–929</sub> | 949 <sub>906–950</sub> | 1,232 <sub>1,231–1,255</sub> | – | 224 <sub>224–224</sub> | – |
| softmax | 1 | uniform | 41 <sub>41–42</sub> | 41 <sub>40–41</sub> | – | 46 <sub>46–48</sub> | 61 <sub>61–62</sub> | 77 <sub>77–78</sub> | – | 48 <sub>47–48</sub> | 75 <sub>75–76</sub> |
| softmax | 1 | long tail | 71 <sub>70–101</sub> | 77 <sub>70–352</sub> | – | 759 <sub>633–5,382</sub> | 92 <sub>82–498</sub> | 79 <sub>78–111</sub> | – | 66 <sub>63–95</sub> | 386 <sub>327–2,529</sub> |
| softmax | 1 | bursty | 45 <sub>44–47</sub> | 45 <sub>43–47</sub> | – | 174 <sub>174–176</sub> | 60 <sub>60–61</sub> | 69 <sub>68–69</sub> | – | 39 <sub>38–42</sub> | 106 <sub>104–178</sub> |
| softmax | 128 | uniform | 348 <sub>348–350</sub> | 331 <sub>331–331</sub> | – | 479 <sub>478–479</sub> | 332 <sub>332–332</sub> | 3,776 <sub>3,755–4,021</sub> | – | 324 <sub>324–325</sub> | 1,952 <sub>1,950–1,954</sub> |
| softmax | 128 | long tail | 422 <sub>418–436</sub> | 507 <sub>482–526</sub> | – | 4,109 <sub>3,702–4,502</sub> | 2,457 <sub>2,151–2,732</sub> | 3,791 <sub>3,785–4,049</sub> | – | 414 <sub>410–415</sub> | 2,510 <sub>2,505–2,519</sub> |
| softmax | 128 | bursty | 520 <sub>519–520</sub> | 519 <sub>516–521</sub> | – | 3,159 <sub>3,154–3,367</sub> | 1,953 <sub>1,950–1,964</sub> | 3,437 <sub>3,415–3,644</sub> | – | 491 <sub>489–493</sub> | 2,736 <sub>2,699–2,742</sub> |
| softmax bf16 | 128 | long tail | 321 <sub>318–331</sub> | 471 <sub>442–485</sub> | – | 3,896 <sub>3,804–4,365</sub> | 2,889 <sub>2,497–3,166</sub> | 4,181 <sub>4,176–4,187</sub> | – | 246 <sub>245–248</sub> | 2,949 <sub>2,845–2,958</sub> |

CUDA-graph replay (warm L2), the same kernels without the host's share of the
call:

| Op | D | Lengths | ragged | ragged-naive | torch.segment_reduce | NJT compiled | CUDA, block per segment | CUB | cuSPARSE | Triton (tuned) | FBGEMM |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sum | 1 | uniform | 16 <sub>16–16</sub> | 16 <sub>16–16</sub> | 11 <sub>11–11</sub> | 5.4 <sub>5.3–5.4</sub> | 33 <sub>33–33</sub> | 18 <sub>18–18</sub> | 22 <sub>22–22</sub> | 15 <sub>14–15</sub> | 18 <sub>18–18</sub> |
| sum | 1 | long tail | 17 <sub>17–27</sub> | 17 <sub>16–66</sub> | 634 <sub>508–5,019</sub> | 292 <sub>240–2,132</sub> | 38 <sub>35–154</sub> | 19 <sub>18–35</sub> | 22 <sub>21–22</sub> | 16 <sub>15–31</sub> | 47 <sub>42–322</sub> |
| sum | 1 | bursty | 16 <sub>16–16</sub> | 16 <sub>16–16</sub> | 230 <sub>225–236</sub> | 62 <sub>61–62</sub> | 33 <sub>33–33</sub> | 14 <sub>14–14</sub> | 22 <sub>22–22</sub> | 13 <sub>12–13</sub> | 16 <sub>16–18</sub> |
| sum | 128 | uniform | 102 <sub>100–102</sub> | 150 <sub>150–150</sub> | 151 <sub>151–151</sub> | 143 <sub>143–143</sub> | 153 <sub>153–153</sub> | 1,302 <sub>1,300–1,320</sub> | 160 <sub>160–161</sub> | 149 <sub>149–149</sub> | – |
| sum | 128 | long tail | 173 <sub>170–177</sub> | 177 <sub>172–181</sub> | 657 <sub>566–715</sub> | 1,005 <sub>991–1,058</sub> | 1,126 <sub>998–1,230</sub> | 1,314 <sub>1,313–1,315</sub> | 160 <sub>160–160</sub> | 167 <sub>164–167</sub> | – |
| sum | 128 | bursty | 159 <sub>158–159</sub> | 151 <sub>151–152</sub> | 476 <sub>471–478</sub> | 737 <sub>736–795</sub> | 905 <sub>902–906</sub> | 1,149 <sub>1,148–1,160</sub> | 160 <sub>160–160</sub> | 151 <sub>151–153</sub> | – |
| max | 1 | uniform | 16 <sub>16–16</sub> | 16 <sub>16–16</sub> | 11 <sub>11–11</sub> | 5.4 <sub>5.2–5.8</sub> | 33 <sub>33–33</sub> | 20 <sub>20–20</sub> | – | 12 <sub>12–12</sub> | – |
| max | 1 | long tail | 16 <sub>16–24</sub> | 16 <sub>16–36</sub> | 455 <sub>349–3,547</sub> | 329 <sub>269–2,262</sub> | 37 <sub>35–151</sub> | 21 <sub>20–40</sub> | – | 13 <sub>12–33</sub> | – |
| max | 1 | bursty | 16 <sub>16–16</sub> | 16 <sub>16–16</sub> | 154 <sub>108–154</sub> | 68 <sub>68–68</sub> | 33 <sub>33–33</sub> | 15 <sub>15–15</sub> | – | 12 <sub>11–12</sub> | – |
| max | 128 | uniform | 102 <sub>101–102</sub> | 150 <sub>150–150</sub> | 151 <sub>151–152</sub> | 143 <sub>143–143</sub> | 153 <sub>153–153</sub> | 1,382 <sub>1,381–1,490</sub> | – | 149 <sub>149–149</sub> | – |
| max | 128 | long tail | 163 <sub>162–165</sub> | 175 <sub>169–179</sub> | 317 <sub>287–337</sub> | 1,114 <sub>1,026–1,188</sub> | 1,108 <sub>956–1,217</sub> | 1,393 <sub>1,391–1,394</sub> | – | 162 <sub>160–163</sub> | – |
| max | 128 | bursty | 158 <sub>158–159</sub> | 151 <sub>150–151</sub> | 262 <sub>217–262</sub> | 887 <sub>886–908</sub> | 863 <sub>840–890</sub> | 1,186 <sub>1,182–1,188</sub> | – | 151 <sub>150–151</sub> | – |
| softmax | 1 | uniform | 18 <sub>18–18</sub> | 18 <sub>18–19</sub> | – | 26 <sub>25–26</sub> | 51 <sub>51–51</sub> | 56 <sub>56–56</sub> | – | 28 <sub>28–28</sub> | 64 <sub>64–64</sub> |
| softmax | 1 | long tail | 34 <sub>33–55</sub> | 40 <sub>34–246</sub> | – | 737 <sub>607–5,347</sub> | 83 <sub>72–484</sub> | 58 <sub>58–92</sub> | – | 46 <sub>43–60</sub> | 376 <sub>315–2,402</sub> |
| softmax | 1 | bursty | 18 <sub>18–20</sub> | 18 <sub>18–20</sub> | – | 153 <sub>153–153</sub> | 51 <sub>51–51</sub> | 48 <sub>48–49</sub> | – | 19 <sub>19–24</sub> | 96 <sub>93–168</sub> |
| softmax | 128 | uniform | 288 <sub>287–289</sub> | 303 <sub>303–304</sub> | – | 436 <sub>436–436</sub> | 305 <sub>305–306</sub> | 3,715 <sub>3,714–3,716</sub> | – | 296 <sub>296–296</sub> | 1,744 <sub>1,744–1,744</sub> |
| softmax | 128 | long tail | 391 <sub>388–397</sub> | 468 <sub>443–488</sub> | – | 4,041 <sub>3,362–4,466</sub> | 2,431 <sub>2,041–2,695</sub> | 3,747 <sub>3,745–3,759</sub> | – | 383 <sub>373–384</sub> | 2,261 <sub>2,253–2,267</sub> |
| softmax | 128 | bursty | 474 <sub>474–474</sub> | 487 <sub>485–487</sub> | – | 3,121 <sub>3,118–3,125</sub> | 1,917 <sub>1,915–1,930</sub> | 3,377 <sub>3,362–3,390</sub> | – | 463 <sub>462–464</sub> | 2,468 <sub>2,437–2,477</sub> |
| softmax bf16 | 128 | long tail | 278 <sub>276–282</sub> | 439 <sub>408–451</sub> | – | 3,853 <sub>3,416–4,326</sub> | 2,863 <sub>2,469–3,142</sub> | 4,144 <sub>4,144–4,145</sub> | – | 182 <sub>181–184</sub> | 2,681 <sub>2,672–2,687</sub> |

## Main run: ragged's schedules and the original comparisons

The main group at commit `f0bd690`, before the cuSPARSE, FBGEMM and D=128 CUB
comparisons were added; its comparison columns are superseded by the run
above, and it adds eager NJT and the forced classified schedule (`ragged,
classified`). Its `ragged` column and *chose* ran cost model version 1. Methods as in [docs/benchmarking.md](../../docs/benchmarking.md).

Warm L2:

| Op | D | Lengths | ragged | chose | ragged, classified | ragged-naive | torch.segment_reduce | Triton (tuned) | CUB | CUDA, block per segment | NJT compiled | NJT eager |
|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sum | 1 | uniform | 33 <sub>33–34</sub> | N/N/N | 48 <sub>48–49</sub> | 34 <sub>33–34</sub> | 15 <sub>15–15</sub> | 33 <sub>32–33</sub> | 30 <sub>30–30</sub> | 42 <sub>42–42</sub> | 28 <sub>28–29</sub> | 147 <sub>145–150</sub> |
| sum | 1 | long tail | 35 <sub>34–46</sub> | C/N/N | 44 <sub>42–45</sub> | 34 <sub>34–83</sub> | 638 <sub>512–5,356</sub> | 33 <sub>32–69</sub> | 29 <sub>28–47</sub> | 42 <sub>40–165</sub> | 284 <sub>238–2,176</sub> | 10,178 <sub>9,032–11,273</sub> (10 runs) |
| sum | 1 | bursty | 33 <sub>33–33</sub> | N/N/N | 39 <sub>39–39</sub> | 32 <sub>31–33</sub> | 235 <sub>229–240</sub> | 30 <sub>30–30</sub> | 25 <sub>25–25</sub> | 38 <sub>38–39</sub> | 75 <sub>75–77</sub> | 2,232 <sub>2,230–2,235</sub> |
| sum | 128 | uniform | 130 <sub>129–132</sub> | C/C/C | 132 <sub>130–136</sub> | 167 <sub>167–168</sub> | 155 <sub>155–156</sub> | 167 <sub>167–168</sub> | – | 161 <sub>161–161</sub> | 163 <sub>163–163</sub> | 565 <sub>562–567</sub> |
| sum | 128 | long tail | 189 <sub>189–193</sub> | C/N/C | 191 <sub>187–192</sub> | 196 <sub>189–199</sub> | 661 <sub>570–719</sub> | 185 <sub>184–185</sub> | – | 1,188 <sub>1,011–1,285</sub> | 1,114 <sub>1,009–1,126</sub> | fails |
| sum | 128 | bursty | 168 <sub>168–169</sub> | N/N/N | 180 <sub>179–181</sub> | 169 <sub>168–169</sub> | 482 <sub>476–484</sub> | 171 <sub>170–171</sub> | – | 914 <sub>911–915</sub> | 757 <sub>756–817</sub> | fails |
| max | 1 | uniform | 33 <sub>33–33</sub> | N/N/N | 48 <sub>48–48</sub> | 33 <sub>33–33</sub> | 15 <sub>15–15</sub> | 31 <sub>30–31</sub> | 32 <sub>32–32</sub> | 42 <sub>41–42</sub> | 29 <sub>28–29</sub> | 114 <sub>113–116</sub> |
| max | 1 | long tail | 34 <sub>33–43</sub> | C/N/N | 42 <sub>42–43</sub> | 34 <sub>33–53</sub> | 459 <sub>353–3,790</sub> | 30 <sub>29–49</sub> | 31 <sub>30–52</sub> | 42 <sub>40–158</sub> | 311 <sub>266–2,299</sub> | 10,138 <sub>9,029–10,857</sub> (10 runs) |
| max | 1 | bursty | 33 <sub>32–33</sub> | N/N/N | 39 <sub>39–40</sub> | 33 <sub>33–34</sub> | 158 <sub>112–159</sub> | 29 <sub>28–29</sub> | 26 <sub>25–26</sub> | 38 <sub>38–38</sub> | 82 <sub>81–83</sub> | 2,208 <sub>2,206–2,210</sub> |
| max | 128 | uniform | 132 <sub>130–138</sub> | C/C/C | 133 <sub>130–134</sub> | 168 <sub>167–168</sub> | 156 <sub>155–156</sub> | 167 <sub>166–168</sub> | – | 161 <sub>161–161</sub> | 162 <sub>162–163</sub> | 536 <sub>535–538</sub> |
| max | 128 | long tail | 183 <sub>181–186</sub> | C/N/C | 180 <sub>178–181</sub> | 192 <sub>188–197</sub> | 324 <sub>291–340</sub> | 181 <sub>179–181</sub> | – | 1,140 <sub>990–1,230</sub> | 1,139 <sub>1,138–1,230</sub> | fails |
| max | 128 | bursty | 167 <sub>167–168</sub> | N/N/N | 180 <sub>179–181</sub> | 168 <sub>167–169</sub> | 266 <sub>222–266</sub> | 170 <sub>169–170</sub> | – | 901 <sub>899–903</sub> | 910 <sub>908–926</sub> | fails |
| softmax | 1 | uniform | 35 <sub>35–36</sub> | N/N/N | 55 <sub>54–55</sub> | 35 <sub>35–36</sub> | – | 46 <sub>46–46</sub> | 75 <sub>75–75</sub> | 59 <sub>59–59</sub> | 47 <sub>47–48</sub> | 164 <sub>162–167</sub> |
| softmax | 1 | long tail | 54 <sub>51–81</sub> | C/N/C | 54 <sub>53–81</sub> | 56 <sub>51–263</sub> | – | 65 <sub>62–93</sub> | 72 <sub>71–110</sub> | 84 <sub>74–494</sub> | 677 <sub>563–5,418</sub> | 12,566 <sub>11,017–13,504</sub> (10 runs) |
| softmax | 1 | bursty | 36 <sub>35–37</sub> | N/N/N | 46 <sub>46–47</sub> | 36 <sub>35–37</sub> | – | 36 <sub>35–39</sub> | 62 <sub>61–62</sub> | 54 <sub>54–54</sub> | 167 <sub>166–167</sub> | 2,649 <sub>2,646–2,656</sub> |
| softmax | 128 | uniform | 310 <sub>308–312</sub> | C/C/C | 310 <sub>309–311</sub> | 321 <sub>320–321</sub> | – | 314 <sub>313–315</sub> | – | 314 <sub>313–314</sub> | 458 <sub>457–458</sub> | 1,194 <sub>1,193–1,195</sub> |
| softmax | 128 | long tail | 403 <sub>401–409</sub> | C/C/C | 404 <sub>399–406</sub> | 485 <sub>461–506</sub> | – | 392 <sub>385–392</sub> | – | 2,438 <sub>2,071–2,700</sub> | 4,436 <sub>3,703–4,715</sub> | fails |
| softmax | 128 | bursty | 506 <sub>505–507</sub> | C/C/C | 506 <sub>504–506</sub> | 506 <sub>505–508</sub> | – | 479 <sub>478–480</sub> | – | 1,937 <sub>1,926–1,944</sub> | 3,428 <sub>3,424–3,438</sub> | fails |
| softmax bf16 | 128 | long tail | 305 <sub>300–309</sub> | C/C/C | 302 <sub>294–307</sub> | 459 <sub>427–472</sub> | – | 220 <sub>220–222</sub> | – | 2,872 <sub>2,486–3,144</sub> | 4,264 <sub>3,785–4,791</sub> | fails |

Cold L2:

| Op | D | Lengths | ragged | chose | ragged, classified | ragged-naive | torch.segment_reduce | Triton (tuned) | CUB | CUDA, block per segment | NJT compiled | NJT eager |
|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sum | 1 | uniform | 36 <sub>36–37</sub> | N/N/N | 55 <sub>54–56</sub> | 36 <sub>36–37</sub> | 22 <sub>21–22</sub> | 34 <sub>34–35</sub> | 34 <sub>34–34</sub> | 45 <sub>45–45</sub> | 31 <sub>30–32</sub> | 152 <sub>150–153</sub> |
| sum | 1 | long tail | 45 <sub>42–63</sub> | C/N/N | 54 <sub>54–63</sub> | 44 <sub>42–131</sub> | 1,208 <sub>986–9,560</sub> | 44 <sub>42–80</sub> | 35 <sub>35–74</sub> | 61 <sub>55–316</sub> | 287 <sub>239–2,192</sub> | 9,851 <sub>8,870–10,878</sub> (10 runs) |
| sum | 1 | bursty | 37 <sub>37–38</sub> | N/N/N | 50 <sub>50–51</sub> | 37 <sub>37–38</sub> | 519 <sub>517–528</sub> | 35 <sub>33–35</sub> | 32 <sub>31–32</sub> | 42 <sub>41–43</sub> | 79 <sub>77–80</sub> | 2,230 <sub>2,168–2,241</sub> |
| sum | 128 | uniform | 229 <sub>229–230</sub> | C/C/C | 229 <sub>228–230</sub> | 218 <sub>218–219</sub> | 207 <sub>207–207</sub> | 219 <sub>218–219</sub> | – | 212 <sub>212–213</sub> | 189 <sub>189–190</sub> | 620 <sub>615–621</sub> |
| sum | 128 | long tail | 248 <sub>241–252</sub> | C/N/C | 249 <sub>244–251</sub> | 247 <sub>241–251</sub> | 718 <sub>624–779</sub> | 236 <sub>235–236</sub> | – | 1,236 <sub>1,062–1,337</sub> | 1,121 <sub>1,026–1,130</sub> | fails |
| sum | 128 | bursty | 224 <sub>223–224</sub> | N/N/N | 232 <sub>232–233</sub> | 223 <sub>223–224</sub> | 538 <sub>530–538</sub> | 225 <sub>225–226</sub> | – | 961 <sub>960–962</sub> | 758 <sub>757–827</sub> | fails |
| max | 1 | uniform | 36 <sub>36–37</sub> | N/N/N | 56 <sub>55–56</sub> | 36 <sub>36–37</sub> | 18 <sub>18–18</sub> | 33 <sub>33–34</sub> | 36 <sub>36–36</sub> | 45 <sub>45–45</sub> | 31 <sub>31–32</sub> | 118 <sub>117–120</sub> |
| max | 1 | long tail | 39 <sub>39–59</sub> | C/N/N | 54 <sub>52–59</sub> | 40 <sub>39–80</sub> | 783 <sub>621–6,242</sub> | 43 <sub>41–76</sub> | 37 <sub>36–79</sub> | 60 <sub>55–311</sub> | 312 <sub>268–2,319</sub> | 9,827 <sub>8,838–10,800</sub> (10 runs) |
| max | 1 | bursty | 38 <sub>38–39</sub> | N/N/N | 51 <sub>50–51</sub> | 39 <sub>38–39</sub> | 225 <sub>214–227</sub> | 34 <sub>34–35</sub> | 33 <sub>32–34</sub> | 42 <sub>41–43</sub> | 85 <sub>83–86</sub> | 2,214 <sub>2,211–2,217</sub> |
| max | 128 | uniform | 229 <sub>228–229</sub> | C/C/C | 228 <sub>227–229</sub> | 218 <sub>218–219</sub> | 205 <sub>205–205</sub> | 218 <sub>218–219</sub> | – | 213 <sub>212–213</sub> | 189 <sub>189–189</sub> | 588 <sub>588–590</sub> |
| max | 128 | long tail | 238 <sub>237–239</sub> | C/N/C | 236 <sub>235–238</sub> | 244 <sub>239–248</sub> | 375 <sub>341–394</sub> | 232 <sub>230–233</sub> | – | 1,182 <sub>1,047–1,283</sub> | 1,162 <sub>1,161–1,247</sub> | fails |
| max | 128 | bursty | 223 <sub>223–224</sub> | N/N/N | 232 <sub>232–233</sub> | 223 <sub>223–224</sub> | 289 <sub>266–290</sub> | 225 <sub>224–226</sub> | – | 950 <sub>949–951</sub> | 918 <sub>910–927</sub> | fails |
| softmax | 1 | uniform | 41 <sub>40–41</sub> | N/N/N | 66 <sub>65–66</sub> | 41 <sub>41–41</sub> | – | 47 <sub>47–48</sub> | 77 <sub>76–77</sub> | 60 <sub>60–60</sub> | 48 <sub>48–49</sub> | 165 <sub>163–168</sub> |
| softmax | 1 | long tail | 70 <sub>69–101</sub> | C/N/C | 70 <sub>68–100</sub> | 78 <sub>70–351</sub> | – | 69 <sub>65–95</sub> | 80 <sub>73–110</sub> | 84 <sub>80–497</sub> | 678 <sub>567–5,267</sub> | 12,264 <sub>11,000–13,537</sub> (10 runs) |
| softmax | 1 | bursty | 45 <sub>44–47</sub> | N/N/N | 63 <sub>61–64</sub> | 45 <sub>43–47</sub> | – | 37 <sub>37–41</sub> | 64 <sub>63–65</sub> | 55 <sub>55–55</sub> | 169 <sub>168–169</sub> | 2,656 <sub>2,646–2,658</sub> |
| softmax | 128 | uniform | 346 <sub>346–347</sub> | C/C/C | 346 <sub>346–347</sub> | 331 <sub>331–331</sub> | – | 324 <sub>324–324</sub> | – | 331 <sub>330–331</sub> | 477 <sub>477–478</sub> | 1,222 <sub>1,220–1,223</sub> |
| softmax | 128 | long tail | 422 <sub>419–429</sub> | C/C/C | 423 <sub>421–432</sub> | 507 <sub>481–528</sub> | – | 412 <sub>404–413</sub> | – | 2,463 <sub>2,095–2,732</sub> | 4,089 <sub>3,718–4,475</sub> | fails |
| softmax | 128 | bursty | 518 <sub>517–520</sub> | C/C/C | 519 <sub>518–520</sub> | 521 <sub>520–522</sub> | – | 491 <sub>490–494</sub> | – | 1,956 <sub>1,950–1,975</sub> | 3,419 <sub>3,142–3,424</sub> | fails |
| softmax bf16 | 128 | long tail | 326 <sub>318–333</sub> | C/C/C | 329 <sub>318–334</sub> | 474 <sub>443–485</sub> | – | 247 <sub>245–248</sub> | – | 2,889 <sub>2,500–3,159</sub> | 3,989 <sub>3,798–4,364</sub> | fails |

On the long tail the seed sets the longest segment: 300,412 rows for seed
711 at D=1 and 29,049 and 35,765 for the others, so the upper quartile of a
D=1 long-tail cell is mostly seed 711. `summary.json` lists every seed's
median and chosen schedule.

What the numbers say:

- **Classification alone loses to one block per segment in 10 of 19 warm
  workloads (14 cold); the per-call choice removes those losses.** Forced
  classification is 19 to 55% slower on every D=1 workload but the long-tail
  softmax, and 7% slower on the bursty D=128 sum and max. The traces
  (`results/bench/profile.txt`) show why: on the uniform D=1 sum the
  classified GPU span is 30 µs (classify 6.5, the CTA workers 18.7, and the
  empty warp and split chains 2.7 to 2.8 µs each) against one 14.8 µs kernel,
  and its host side takes 24 µs against 14 (`results/bench/host.txt`: six
  launches, a stream fork and a join instead of one launch). On the bursty
  D=128 sum the 40 long segments already fill 160 blocks of one block per
  segment (149 µs), so splitting them (147 µs) buys nothing and the prologue
  and the warp chain beside it are extra. Classification wins where many
  short blocks or one long block hurt: the uniform D=128 sum, whose 16,384
  short blocks take 149 µs against 101 µs for persistent CTA workers, and
  long segments, such as seed 711's 300,412-row D=1 segment (sum 46 µs
  against 84, softmax 82 against 264).
- **Version 1's choice stays within noise of block per segment with a warm L2.**
  Over all 15 runs the adaptive median is at most 1.4% above `ragged-naive`
  (D=1 bursty sum, 32.7 against 32.3 µs, the same kernel); on the held-out
  seed 713 at most 1.1% (`results/bench/calibration/evaluation.txt`). Where it
  classifies it is 17 to 34% faster than block per segment on the uniform
  D=128 sum and max and the long-tail D=128 softmax (f32 and bf16).
- **With a cold L2 it is up to 5% slower than block per segment on uniform
  D=128 inputs.** The model is calibrated on warm times, where classification
  wins those three workloads (sum 130 against 167 µs); cold, block per
  segment is faster (sum 218 against 229, max 218 against 229, softmax 331
  against 346), and the model still classifies.
- **The cold D=128 uniform sum still reads at 0.59 TB/s, mostly because of
  the flush.** It takes 229 µs (block per segment 218) against 133 µs for
  reading 128 MiB at 1.01 TB/s. The benchmark's flush leaves the L2 full of
  dirty lines, which are written back while the call reads: after a flush
  that only reads, the same call takes 175 µs classified and 163 µs (0.82
  TB/s, host time included) block per segment, and `torch.segment_reduce` 152
  instead of 206 (`results/bench/cold-l2.txt`). Vector loads did not change
  the cold time (227 µs scalar). Compiled NJT stays the fastest cold
  uniform D=128 sum and max (189 µs).
- **Vector access helps block per segment and the f32 classified softmax,
  and hurts bf16.** With 16-byte accesses and four times the row lanes per
  block, one block per segment takes 196 µs instead of 236 on the long-tail
  D=128 sum, 485 instead of 572 on the long-tail softmax and 506 instead of
  534 on the bursty softmax; classification takes 404 instead of 437 on the
  long-tail softmax; elsewhere both schedules are within 4% of one feature
  per thread. On the bf16 long-tail softmax, where a thread owns eight
  features and its online CTA kernel needs 159 registers instead of 45 (3 blocks
  per SM), it is slower: 302 µs classified against 264 µs with one feature
  per thread, and block per segment 459 against 442.
- **The comparisons still win at D=1 and on skewed D=128 inputs.**
  `torch.segment_reduce` takes 15 µs on uniform sum and max, where ragged
  takes 33; CUB takes 25 to 29 µs on the long-tail and bursty sum and the
  bursty max (ragged 33 to 35), Triton 30 µs on the long-tail max (34) and
  36 on the bursty softmax (ragged 36, 1% slower). Tuned Triton is 1 to 6%
  faster on the long-tail D=128 sum, max and softmax and the bursty
  softmax, and 1.38 times faster on the bf16 long-tail softmax (220 against
  305 µs). ragged
  beats every comparison on the uniform D=128 sum, max and softmax, the
  bursty D=128 sum and max, and the uniform and long-tail D=1 softmax.
- **Length hints** are now mostly redundant: exact bounds still compile the
  uniform D=128 sum to one CTA class, which is block per segment (167 µs
  against 130), and the per-call choice already gives the D=1 workloads the
  block-per-segment time.
- **NJT** with exact sequence lengths compiles to fast code for uniform
  lengths (28 µs for the D=1 sum, the fastest cold D=128 uniform sum and max)
  and is 2 to 14 times slower than ragged on skewed ones. Eager NJT pads to the
  longest segment: it fails on an int32 index limit for skewed D=128 inputs
  and for seed 711 at D=1, and takes 0.1 to 12.6 ms elsewhere.

**Ablations**, warm, µs (`results/bench/runs/ablation/`), each turning off
one optimization of the classified schedule in the same build:
`--disable-optimization online_fusion` (three passes in every class),
`--disable-optimization row_pipelining` and `--serial-classes`:

| Workload | ragged, classified | three passes in every class | rows not pipelined | classes serial |
|---|---:|---:|---:|---:|
| sum, D=1, uniform | 48 <sub>48–49</sub> | – | 49 <sub>48–49</sub> | 55 <sub>55–56</sub> |
| sum, D=1, long tail | 44 <sub>42–45</sub> | – | 44 <sub>43–49</sub> | 49 <sub>49–50</sub> |
| sum, D=1, bursty | 39 <sub>39–39</sub> | – | 39 <sub>39–40</sub> | 42 <sub>42–43</sub> |
| sum, D=128, uniform | 132 <sub>130–136</sub> | – | 134 <sub>130–136</sub> | 136 <sub>135–137</sub> |
| sum, D=128, long tail | 191 <sub>187–192</sub> | – | 192 <sub>189–195</sub> | 202 <sub>198–203</sub> |
| sum, D=128, bursty | 180 <sub>179–181</sub> | – | 182 <sub>182–182</sub> | 184 <sub>183–185</sub> |
| max, D=1, uniform | 48 <sub>48–48</sub> | – | 48 <sub>48–49</sub> | 55 <sub>55–56</sub> |
| max, D=1, long tail | 42 <sub>42–43</sub> | – | 44 <sub>43–48</sub> | 49 <sub>49–50</sub> |
| max, D=1, bursty | 39 <sub>39–40</sub> | – | 40 <sub>39–40</sub> | 43 <sub>43–43</sub> |
| max, D=128, uniform | 133 <sub>130–134</sub> | – | 134 <sub>132–136</sub> | 136 <sub>134–137</sub> |
| max, D=128, long tail | 180 <sub>178–181</sub> | – | 189 <sub>187–190</sub> | 189 <sub>187–189</sub> |
| max, D=128, bursty | 180 <sub>179–181</sub> | – | 181 <sub>181–182</sub> | 183 <sub>183–184</sub> |
| softmax, D=1, uniform | 55 <sub>54–55</sub> | 79 <sub>79–80</sub> | 53 <sub>52–53</sub> | 60 <sub>60–61</sub> |
| softmax, D=1, long tail | 54 <sub>53–81</sub> | 68 <sub>67–89</sub> | 55 <sub>54–107</sub> | 65 <sub>63–92</sub> |
| softmax, D=1, bursty | 46 <sub>46–47</sub> | 59 <sub>58–59</sub> | 48 <sub>47–48</sub> | 53 <sub>52–53</sub> |
| softmax, D=128, uniform | 310 <sub>309–311</sub> | 624 <sub>623–624</sub> | 310 <sub>309–312</sub> | 321 <sub>320–322</sub> |
| softmax, D=128, long tail | 404 <sub>399–406</sub> | 631 <sub>629–633</sub> | 493 <sub>479–508</sub> | 451 <sub>442–457</sub> |
| softmax, D=128, bursty | 506 <sub>504–506</sub> | 651 <sub>651–653</sub> | 505 <sub>504–506</sub> | 511 <sub>510–512</sub> |
| softmax bf16, D=128, long tail | 302 <sub>294–307</sub> | – | – | – |

Online statistics are faster in every softmax workload: three passes take
26 to 101% longer. Concurrent class chains are faster in every f32 workload:
serial chains take 1 to 20% longer. Row pipelining helps the long-tail
D=128 workloads most (max 189 against 180 µs, softmax 493 against 404) and is
slower in three: the uniform D=1 softmax by 3%, the uniform and bursty D=128
softmax by 0.1 to 0.2%.

**Vector access** (`--disable-optimization vectorize`, both schedules, D=128),
µs:

| Workload | classified | classified, scalar | naive | naive, scalar |
|---|---:|---:|---:|---:|
| sum, D=128, uniform, warm | 132 <sub>130–136</sub> | 135 <sub>130–138</sub> | 167 <sub>167–168</sub> | 166 <sub>166–166</sub> |
| sum, D=128, uniform, cold | 229 <sub>228–230</sub> | 227 <sub>227–228</sub> | 218 <sub>218–219</sub> | 216 <sub>216–217</sub> |
| sum, D=128, long tail, warm | 191 <sub>187–192</sub> | 195 <sub>192–196</sub> | 196 <sub>189–199</sub> | 236 <sub>223–244</sub> |
| sum, D=128, long tail, cold | 249 <sub>244–251</sub> | 251 <sub>248–254</sub> | 247 <sub>241–251</sub> | 289 <sub>274–298</sub> |
| sum, D=128, bursty, warm | 180 <sub>179–181</sub> | 181 <sub>180–181</sub> | 169 <sub>168–169</sub> | 170 <sub>170–171</sub> |
| sum, D=128, bursty, cold | 232 <sub>232–233</sub> | 231 <sub>231–232</sub> | 223 <sub>223–224</sub> | 226 <sub>226–227</sub> |
| max, D=128, uniform, warm | 133 <sub>130–134</sub> | 138 <sub>132–140</sub> | 168 <sub>167–168</sub> | 166 <sub>166–166</sub> |
| max, D=128, uniform, cold | 228 <sub>227–229</sub> | 227 <sub>227–228</sub> | 218 <sub>218–219</sub> | 215 <sub>215–216</sub> |
| max, D=128, long tail, warm | 180 <sub>178–181</sub> | 184 <sub>182–186</sub> | 192 <sub>188–197</sub> | 195 <sub>188–198</sub> |
| max, D=128, long tail, cold | 236 <sub>235–238</sub> | 241 <sub>239–243</sub> | 244 <sub>239–248</sub> | 247 <sub>241–250</sub> |
| max, D=128, bursty, warm | 180 <sub>179–181</sub> | 181 <sub>181–181</sub> | 168 <sub>167–169</sub> | 167 <sub>166–168</sub> |
| max, D=128, bursty, cold | 232 <sub>232–233</sub> | 232 <sub>231–233</sub> | 223 <sub>223–224</sub> | 222 <sub>222–223</sub> |
| softmax, D=128, uniform, warm | 310 <sub>309–311</sub> | 314 <sub>313–315</sub> | 321 <sub>320–321</sub> | 335 <sub>335–336</sub> |
| softmax, D=128, uniform, cold | 346 <sub>346–347</sub> | 346 <sub>346–347</sub> | 331 <sub>331–331</sub> | 343 <sub>343–344</sub> |
| softmax, D=128, long tail, warm | 404 <sub>399–406</sub> | 437 <sub>429–443</sub> | 485 <sub>461–506</sub> | 572 <sub>528–598</sub> |
| softmax, D=128, long tail, cold | 423 <sub>421–432</sub> | 458 <sub>451–472</sub> | 507 <sub>481–528</sub> | 593 <sub>551–622</sub> |
| softmax, D=128, bursty, warm | 506 <sub>504–506</sub> | 506 <sub>506–510</sub> | 506 <sub>505–508</sub> | 534 <sub>523–549</sub> |
| softmax, D=128, bursty, cold | 519 <sub>518–520</sub> | 518 <sub>517–520</sub> | 521 <sub>520–522</sub> | 542 <sub>534–593</sub> |
| softmax bf16, D=128, long tail, warm | 302 <sub>294–307</sub> | 264 <sub>260–275</sub> | 459 <sub>427–472</sub> | 442 <sub>396–469</sub> |
| softmax bf16, D=128, long tail, cold | 329 <sub>318–334</sub> | 301 <sub>299–319</sub> | 474 <sub>443–485</sub> | 469 <sub>425–498</sub> |

**Length hints** (`--length-hints`, exact bounds), warm, µs:

| Workload | ragged | ragged with exact length hints | classes with hints | ragged-naive |
|---|---:|---:|---|---:|
| softmax, D=1, uniform | 35 <sub>35–36</sub> | 36 <sub>35–36</sub> | cta | 35 <sub>35–36</sub> |
| softmax, D=128, uniform | 310 <sub>308–312</sub> | 321 <sub>321–321</sub> | cta | 321 <sub>320–321</sub> |
| sum, D=1, long tail | 35 <sub>34–46</sub> | 35 <sub>34–46</sub> | warp+cta+split | 34 <sub>34–83</sub> |
| sum, D=1, uniform | 33 <sub>33–34</sub> | 33 <sub>33–33</sub> | cta | 34 <sub>33–34</sub> |
| sum, D=128, long tail | 189 <sub>189–193</sub> | 193 <sub>191–194</sub> | warp+cta+split | 196 <sub>189–199</sub> |
| sum, D=128, uniform | 130 <sub>129–132</sub> | 167 <sub>167–168</sub> | cta | 167 <sub>167–168</sub> |

**Kernel traces** (`results/bench/profile.txt`, seed 711). With uniform
lengths at D=128 every 64-row segment is in the CTA class, whose online
softmax kernel takes 281 µs warm to read and write 128 MiB each. On the D=128
long-tail softmax the CTA, warp and split classes take 291, 68 and 39 µs run
one after another; concurrently the call's GPU span is 398 µs instead of 428,
although every kernel runs longer while they share the GPU. Cold, the CTA
kernel of the uniform D=128 sum takes 200 µs against 101 warm.

## The cost model on held-out shapes

**Version 1 on the first held-out set.** `python -m bench.suite --groups
heldout --repetitions 3` at commit `a7a1428` (567 runs: 7 shapes × sum,
max, softmax × uniform, long-tail, bursty × seeds 711 to 713 × 3
repetitions), then `python -m bench.calibrate evaluate results/bench/runs
--group heldout --seeds 711 712 713` at that commit
(`calibration/heldout-shapes.txt`). Version 1 was fitted on D=1 with 16,384
segments in 1,048,576 rows and D=128 with 4,096 segments in 262,144 rows.
Per shape, over its 27 (workload, seed) inputs, the adaptive call against
the faster of the two schedules measured in the same runs, warm L2:

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
three repetitions per seed do not always outvote. The traces at the end of
`profile.txt` show both errors: with 65,536 segments of 16 rows, block per
segment's one kernel takes 53.6 µs against a 30.0 µs classified span (14.6
µs of it the warp class); on the 64 MiB uniform D=32 and D=64 sums, which
stay in the L2, it takes 16 µs against 25 to 27, and the classified call
adds 24 µs of host time against 15. Version 1 had put classification's
fixed cost into its per-segment term and charged DRAM time for every
element ([docs/design-tradeoffs.md](../../docs/design-tradeoffs.md#choosing-classification-or-one-block-per-segment-per-call)).

**The refit.** These 189 inputs then informed the change, so they became
training data: version 2 is fitted on `runs/main` and `runs/heldout` (246
inputs in 9 shapes, `calibration/fit.txt`), with the MiB of the working set
beyond the 72 MiB L2 in place of the row bytes and the margin chosen by
leave-one-shape-out cross-validation (0%; 203 of 246 left-out inputs within
3% warm). In-sample, on the same runs (`calibration/refit-training.txt`),
its choices are within 3% for 160 of the 189 first held-out inputs (version
1's choices: 125; the adaptive call as measured with version 1: 115) and
for 50 of the 57 main inputs (version 1: 56); the 7 are the bursty D=128 sum
and max, where it classifies, and seed 712 of the long-tail D=1 softmax.

**Version 2 on the second held-out set.** Fixed in `bench/suite.py`
(`HELDOUT2_SHAPES`, commit `5551923`) before version 2 ran: 8 shapes whose
segment and row counts no earlier group used, D from 1 to 512, and bimodal
and exponential lengths, which no earlier group measured. `python -m
bench.suite --groups heldout2 --repetitions 3` at commit `6b0b842` (432
runs, all exit 0), then, once, `python -m bench.calibrate evaluate
results/bench/runs --group heldout2` and the same with `--model
calibration/cost_model-v1.json`, which replays version 1's choices on the
same runs (`calibration/heldout2.txt`). Per shape, over its 18 inputs, the
adaptive call (version 2) against the faster schedule, warm L2, and version
1's choices:

| D | Segments | Rows | Within 3% | Median excess | Largest excess | Version 1 within 3% |
|---:|---:|---:|---:|---:|---|---:|
| 1 | 200,000 | 3,200,000 | 11 of 18 | +2.1% | +10% (softmax, exponential) | 17 of 18 |
| 2 | 32,768 | 2,097,152 | 10 of 18 | +1.4% | +37% (max, bimodal) | 9 of 18 |
| 8 | 131,072 | 1,310,720 | 18 of 18 | +0.1% | +1% (max, exponential) | 0 of 18 |
| 16 | 12,288 | 786,432 | 13 of 18 | +1.4% | +5% (sum, exponential) | 15 of 18 |
| 48 | 6,000 | 393,216 | 17 of 18 | +0.7% | +4% (sum, bimodal) | 0 of 18 |
| 96 | 2,500 | 160,000 | 17 of 18 | +0.4% | +3% (max, exponential) | 6 of 18 |
| 192 | 768 | 196,608 | 9 of 18 | +3.4% | +5% (sum, bimodal) | 9 of 18 |
| 512 | 512 | 32,768 | 17 of 18 | +0.8% | +6% (softmax, bimodal) | 5 of 18 |

In all, the adaptive call is within 3% for 112 of 144 inputs warm and 104
cold (the model's choice alone, against the same runs' schedule medians:
119 and 107); version 1's choices are within 3% for 61 and 65. Version 2's
systematic errors: on the D=2 bimodal sum and max it runs block per segment,
34 to 37% slower than classification; on the D=1 softmax with 200,000
segments it classifies, 8 to 10% slower; at D=192 with 768 segments it
classifies the sum and max, 4 to 5% slower. Version 1, which runs block per
segment on both, is better at D=1 with 200,000 segments (17 against 11 of
18), where the two schedules are within a few percent for the sum and max
and block per segment is 8 to 10% faster for the softmax, and at D=16 (15
against 13).

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
