#!/usr/bin/env python3
"""Compile handwritten Triton baselines to real sm_89 cubins, with no GPU run."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import subprocess

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('evidence/triton-compilation'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ['TRITON_CACHE_DIR'] = str((args.output / 'cache').resolve())
    import triton
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    from baselines.triton_reference import _reduce, _normalize, toolchain
    source = Path(__file__).resolve().parents[1] / 'baselines/triton_reference.py'
    report = {'scope': 'CPU compilation to sm_89 cubin only; no GPU numerical or performance evidence',
              'toolchain': toolchain(), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'target': ['cuda', 89, 32],
              'options': {'num_warps': 4, 'enable_fp_fusion': False, 'enable_reflect_ftz': False}, 'cases': []}
    for dtype, offsets, d, phase in itertools.product(('fp32', 'bf16'), ('i32', 'i64'), (1, 7, 16, 128, 1024),
                                                       ('sum', 'max', 'max_stats', 'sum_stats', 'normalize')):
        name = f'{phase}-{dtype}-{offsets}-d{d}'
        record = {'name': name, 'passed': False}
        report['cases'].append(record)
        try:
            constants = {'D': d, 'TILE': min(32, triton.next_power_of_2(d)), 'ROWS': 32}
            sig = {'X': '*' + dtype, 'Offsets': '*' + offsets}
            if phase == 'normalize':
                fn = _normalize
                sig.update(Maximum='*fp32', Total='*fp32', Out='*' + dtype)
            else:
                fn = _reduce
                constants['OP'] = 0 if phase == 'sum' else 2 if phase == 'sum_stats' else 1
                sig.update(Out='*fp32' if phase.endswith('stats') else '*' + dtype,
                           Maximum='*fp32' if phase.endswith('stats') else '*' + dtype)
            compiled = triton.compile(ASTSource(fn, sig, constexprs=constants),
                                      target=GPUTarget('cuda', 89, 32), options=report['options'])
            directory = args.output / name
            directory.mkdir(exist_ok=True)
            for kind in ('ttir', 'ttgir', 'llir', 'ptx', 'cubin'):
                data = compiled.asm[kind]
                (directory / ('program.' + kind)).write_bytes(data if isinstance(data, bytes) else data.encode())
            binary = compiled.asm['cubin']
            if not binary.startswith(b'\x7fELF') or '.target sm_89' not in compiled.asm['ptx']:
                raise ValueError('missing actual sm_89 ELF cubin or matching PTX')
            command = [report['toolchain']['ptxas_path'], '-arch=sm_89', '-v',
                       str(directory / 'program.ptx'), '-o', str(directory / 'independent.cubin')]
            assembled = subprocess.run(command, capture_output=True, text=True)
            (directory / 'ptxas.log').write_text(assembled.stdout + assembled.stderr)
            if assembled.returncode:
                raise ValueError('independent pinned ptxas assembly failed')
            record.update(passed=True, cubin_sha256=hashlib.sha256(binary).hexdigest(),
                          signature=sig, constants=constants, metadata=compiled.metadata._asdict(),
                          independent_command=command)
        except Exception as exc:
            record['error'] = f'{type(exc).__name__}: {exc}'
        print(name, record['passed'], record.get('error', ''), flush=True)
        (args.output / 'results.json').write_text(json.dumps(report, indent=2, default=asdict) + '\n')
    report['source_unchanged'] = report['source_sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
    report['passed'] = all(r['passed'] for r in report['cases']) and report['source_unchanged']
    (args.output / 'results.json').write_text(json.dumps(report, indent=2, default=asdict) + '\n')
    raise SystemExit(not report['passed'])


if __name__ == '__main__':
    main()
