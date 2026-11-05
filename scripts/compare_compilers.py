"""Compare the GPU dialect, PTX and kernel manifest two ragged-compile builds emit for sm_89 programs.

    PYTHONPATH=python:. python scripts/compare_compilers.py OLD/bin/ragged-compile build/bin/ragged-compile

The programs are those of every benchmark-suite group, and each op, dtype, D, offset type with and
without length hints, vector access and row pipelining, mask, scale and in-place output.
"""
import itertools, subprocess, sys, tempfile
from dataclasses import replace
from pathlib import Path
from bench.precompile import specs
from ragged import Hints, Optimizations, ProgramSpec

old, new = sys.argv[1], sys.argv[2]
programs = list(specs(["main", "hints", "ablation", "compare", "heldout", "heldout2"], [711, 712, 713], "sm_89").values())
for op, dtype, d, offsets in itertools.product(("sum", "max", "softmax"), ("f32", "bf16"), (1, 2, 7, 24, 40, 1024), ("i32", "i64")):
    base = ProgramSpec(op, dtype, d, offsets)
    programs += [base, replace(base, strategy="naive"), replace(base, hints=Hints(max_len=32, offsets_checked=True)),
                 replace(base, hints=Hints(min_len=40, max_len=5000)),
                 replace(base, optimizations=Optimizations(vectorize=False, row_pipelining=False))]
    if op == "softmax":
        programs += [replace(base, mask=True), replace(base, alias="inplace", strategy="naive"), replace(base, scale=0.5)]
seen, differ = set(), []
with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    for spec in programs:
        if spec.ir_hash in seen:
            continue
        seen.add(spec.ir_hash)
        (tmp / "in.mlir").write_text(spec.to_mlir())
        outputs = []
        for tool in (old, new):
            got = []
            for emit in ("gpu", "ptx"):
                out = tmp / f"out.{emit}"
                r = subprocess.run([tool, str(tmp / "in.mlir"), "-o", str(out), f"--emit={emit}"], capture_output=True)
                got.append((r.returncode, out.read_bytes() if out.exists() else b"", r.stderr))
                if emit == "ptx":
                    side = Path(str(out) + ".json")
                    got.append(side.read_bytes() if side.exists() else b"")
                for f in tmp.glob("out.*"):
                    f.unlink()
            outputs.append(got)
        if outputs[0] != outputs[1]:
            differ.append(spec.signature + " " + spec.ir_hash[:12])
print(f"{len(seen)} sm_89 programs; GPU dialect, PTX and manifest differ for {len(differ)}")
for line in differ[:10]:
    print("  differs:", line)
