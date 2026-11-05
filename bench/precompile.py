"""Compile ahead of time every ragged program that suite groups run.

For a GPU machine without the MLIR toolchain (another glibc, no CUDA
toolkit): the ragged-* methods of every case, seed and method override that
``bench.suite`` would run are compiled for ``--chip`` into ``--cache``, with
their block-per-segment companions, and ``<cache>/toolchain.json`` records
the compiler's identity. ``bench.suite --toolchain-record <cache>/toolchain.json
--cache <cache>`` then loads them and runs no compiler. No GPU is used.

    python -m bench.precompile --chip sm_86 --groups heldout heldout2 compare --cache build/sm86-cache
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from bench.distributions import make_lengths  # noqa: E402
from bench.run import build_parser, ragged_spec  # noqa: E402
from bench.suite import cases  # noqa: E402
from ragged import ArtifactCache, Compiler  # noqa: E402


def specs(groups: list[str], seeds: list[int], chip: str, methods: list[str] | None = None) -> dict:
    """{ir_hash: spec} of the ragged programs the selected suite cases compile."""
    out = {}
    for group, name, flags in cases():
        if group.split("/")[0] not in groups:
            continue
        if methods:
            flags = flags[:flags.index("--methods") + 1] + methods
        for seed in seeds:
            args = build_parser().parse_args([*flags, "--chip", chip, "--seed", str(seed), "--output", "unused"])
            if args.fixed_segments:
                args.total_elements = None
            args.length_bounds = {}
            if args.length_hints:
                lengths = make_lengths(args.distribution, args.segments, seed=args.seed,
                                       total_elements=args.total_elements, mean_length=args.mean_length)
                args.length_bounds = {"min_len": int(lengths.min()) if len(lengths) else 0,
                                      "max_len": int(lengths.max()) if len(lengths) else 0}
            for method in args.methods:
                if method.startswith("ragged-"):
                    spec = ragged_spec(method, args, float(np.float32(args.scale)), args.mask)
                    out.setdefault(spec.ir_hash + spec.chip, spec)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--chip", choices=("sm_86", "sm_89"), required=True)
    parser.add_argument("--groups", nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[711, 712, 713])
    parser.add_argument("--methods", nargs="+", help="the method override bench.suite will be given")
    parser.add_argument("--compiler", default="build/bin/ragged-compile")
    args = parser.parse_args()
    compiler = Compiler(args.compiler)
    cache = ArtifactCache(args.cache, max_specializations=4096)
    selected = specs(args.groups, args.seeds, args.chip, args.methods)
    compiled = 0
    for spec in selected.values():
        program = compiler.compile(spec, cache)
        compiled += not program.cache_hit
    compiler.record_toolchain(args.cache / "toolchain.json", chips=(args.chip,))
    print(f"{len(selected)} programs for {args.chip} in {args.cache} ({compiled} compiled now); "
          f"record {args.cache / 'toolchain.json'}")


if __name__ == "__main__":
    main()
