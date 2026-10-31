"""Emit source IR or invoke a configured native compiler."""
import argparse
import sys

from .compiler import Compiler
from .errors import RaggedError
from .model import Hints, Optimizations, ProgramSpec
from .optimizations import BOOL_OPTIONS


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="ragged", description=__doc__)
    parser.add_argument("op", choices=("sum", "max", "softmax"))
    parser.add_argument("--dtype", choices=("f32", "bf16"), default="f32")
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--offsets-dtype", choices=("i32", "i64"), default="i64")
    parser.add_argument("--max-len", type=int)
    parser.add_argument("--min-len", type=int)
    parser.add_argument("--offsets-checked", action="store_true")
    parser.add_argument("--strategy", choices=("auto", "naive"), default="auto")
    parser.add_argument("--inplace", action="store_true")
    parser.add_argument("--force-fusion", action="store_true")
    parser.add_argument("--debug-visits", action="store_true")
    parser.add_argument("--mask", action="store_true", help="accept a dense f32 additive softmax mask")
    parser.add_argument("--scale", type=float, default=1.0)
    for name in BOOL_OPTIONS:
        parser.add_argument("--disable-" + name.replace("_", "-"), dest=name, action="store_false", default=True)
    parser.add_argument("--feature-tile", type=int)
    parser.add_argument("--smem-per-cta", type=int, default=49152,
                        help="maximum measured static plus dynamic shared bytes per CTA")
    parser.add_argument("--regs-budget", type=int, default=65536,
                        help="maximum measured registers per thread times CTA threads")
    parser.add_argument("--emit", choices=("source", "legal", "plan", "gpu", "nvvm", "ptx", "cubin"), default="source")
    parser.add_argument("--compiler")
    parser.add_argument("--features", default="+ptx80")
    parser.add_argument("--explain", action="store_true")
    parser.add_argument("-o", "--output")
    args = parser.parse_args(argv)
    try:
        spec = ProgramSpec(args.op, args.dtype, args.feature_dim, args.offsets_dtype,
                           Hints(args.max_len, args.min_len, args.offsets_checked), args.strategy,
                           alias="inplace" if args.inplace else "none",
                           force_fusion=args.force_fusion, scale=args.scale, debug_visits=args.debug_visits, mask=args.mask,
                           optimizations=Optimizations(**{name:getattr(args,name) for name in BOOL_OPTIONS},feature_tile=args.feature_tile),
                           smem_per_cta=args.smem_per_cta, regs_budget=args.regs_budget)
        if args.emit == "source":
            output = spec.to_mlir().encode()
        else:
            result = Compiler(args.compiler).lower(spec, emit=args.emit, explain=args.explain, features=args.features)
            output = result.output
            if result.diagnostics:
                sys.stderr.write(result.diagnostics)
        if args.output:
            from pathlib import Path
            Path(args.output).write_bytes(output)
        elif args.emit == "cubin":
            parser.error("--emit=cubin requires -o to avoid writing binary to the terminal")
        else:
            sys.stdout.write(output.decode())
        return 0
    except RaggedError as exc:
        print(f"ragged: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
