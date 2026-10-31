#!/usr/bin/env python3
"""Build comparison-only CUDA/CUB baselines without requiring a GPU."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nvcc', default='/usr/local/cuda-13.2/bin/nvcc')
    parser.add_argument('--cccl', type=Path, default=Path('.deps/cccl-3.5.0'))
    parser.add_argument('--output', type=Path, default=Path('build-baselines'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve().parents[1] / 'baselines/cuda/baselines.cu'
    library = args.output.resolve() / 'libragged_baselines.so'
    include = [args.cccl / path for path in ('cub', 'thrust', 'libcudacxx/include')]
    if any(not path.is_dir() for path in include):
        parser.error('need the official CCCL v3.5.0 archive extracted at --cccl')
    root = source.parents[2]

    def shown(path):
        # Record paths inside the checkout relative to it, so build.json and the
        # benchmark records that embed it do not name the build machine.
        path = Path(path).resolve()
        return str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
    command = [args.nvcc, '-std=c++17', '-O3', '--fmad=false', '-arch=sm_89',
               '--shared', '-Xcompiler=-fPIC', '-Xptxas=-v',
               *['-I' + shown(path) for path in include], shown(source), '-o', shown(library)]
    report = {'schema': 1, 'scope': 'compile only; device correctness and performance unverified',
              'command': command, 'source_sha256': sha256(source), 'cccl_expected': '3.5.0',
              'nvcc_version': subprocess.run([args.nvcc, '--version'], capture_output=True,
                                            text=True, check=True).stdout}
    started = time.perf_counter()
    proc = subprocess.run(command, capture_output=True, text=True, cwd=root)
    (args.output / 'nvcc.log').write_text(proc.stdout + proc.stderr)
    report.update(exit_code=proc.returncode, wall_seconds=time.perf_counter() - started,
                  source_unchanged=report['source_sha256'] == sha256(source))
    if proc.returncode == 0:
        report['library_sha256'] = sha256(library)
    (args.output / 'build.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f"nvcc exit {proc.returncode}; evidence: {args.output / 'build.json'}")
    raise SystemExit(proc.returncode or (not report['source_unchanged']))


if __name__ == '__main__':
    main()
