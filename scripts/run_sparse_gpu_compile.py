#!/usr/bin/env python3
"""Compile the CSR semantic probe through upstream sparse GPU lowering.

This preserves host LLVM IR with embedded cubins; it does not run a GPU, time
a baseline, or decide the sparse-vs-CUB comparison.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mlir-opt', type=Path, default=ROOT / '.deps/llvm-build/bin/mlir-opt')
    parser.add_argument('--output', type=Path, default=ROOT / 'evidence/sparse-gpu-compile')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'test/sparse/sparse-segment-semantics.mlir'
    text = source.read_text()
    report = {'timestamp': datetime.now(timezone.utc).isoformat(),
              'scope': 'sparse CSR GPU compilation only; D=1; no execution, timing or scope-narrowing conclusion',
              'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'tool_version': subprocess.run([str(args.mlir_opt), '--version'], capture_output=True,
                                             text=True, check=True).stdout, 'cases': []}
    for dtype in ('f64', 'f32'):
        directory = args.output / dtype
        directory.mkdir(exist_ok=True)
        transformed = text if dtype == 'f64' else text.replace('0xFFF0000000000000', '0xFF800000').replace(
            '0x7FF8000000000000', '0x7FC00000').replace('0x7FF0000000000000', '0x7F800000').replace('f64', 'f32')
        input_path = directory / 'input.mlir'
        input_path.write_text(transformed)
        command = [str(args.mlir_opt), str(input_path),
                   '--sparsifier=enable-runtime-library=false enable-buffer-initialization=true '
                   'parallelization-strategy=dense-outer-loop gpu-triple=nvptx64-nvidia-cuda '
                   'gpu-chip=sm_89 gpu-features=+ptx80 gpu-format=bin',
                   '--mlir-print-ir-after=sparse-gpu-codegen', '-o', str(directory / 'program.llvm.mlir')]
        result = subprocess.run(command, capture_output=True, text=True, timeout=180)
        (directory / 'compile.log').write_text(result.stdout + result.stderr)
        output = (directory / 'program.llvm.mlir').read_text() if result.returncode == 0 else ''
        # Escaped ELF header in the final GPU binary object proves device objects
        # actually exist; a successful host-only fallback is not a GPU result.
        elf_objects = output.count('\\7FELF')
        record = {'dtype': dtype, 'command': command, 'exit_code': result.returncode,
                  'input_sha256': hashlib.sha256(input_path.read_bytes()).hexdigest(),
                  'gpu_functions_in_intermediate': len(re.findall(r'gpu.func .* kernel', result.stderr)),
                  'embedded_elf_objects': elf_objects,
                  'passed': result.returncode == 0 and elf_objects > 0}
        if record['passed']:
            match = re.search(r'bin = "([^\"]*)"', output)
            if not match:
                raise ValueError('could not locate the serialized GPU object')
            literal = match.group(1)
            binary = bytearray()
            index = 0
            while index < len(literal):
                if literal[index] == '\\':
                    binary.append(int(literal[index + 1:index + 3], 16))
                    index += 3
                else:
                    binary.extend(literal[index].encode('ascii'))
                    index += 1
            if not binary.startswith(b'\x7fELF'):
                raise ValueError('serialized binary lacks the ELF header')
            cubin = directory / 'program.cubin'
            cubin.write_bytes(binary)
            inspect_command = ['/usr/local/cuda-13.2/bin/cuobjdump', '--dump-resource-usage', str(cubin)]
            inspected = subprocess.run(inspect_command, capture_output=True, text=True, timeout=30)
            (directory / 'cubin-resources.log').write_text(inspected.stdout + inspected.stderr)
            record.update(cubin_sha256=hashlib.sha256(binary).hexdigest(), cubin_bytes=len(binary),
                          inspect_command=inspect_command, inspect_exit_code=inspected.returncode)
            record['passed'] &= inspected.returncode == 0
        report['cases'].append(record)
        print(dtype, record, flush=True)
    report['passed'] = all(case['passed'] for case in report['cases'])
    (args.output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    raise SystemExit(not report['passed'])


if __name__ == '__main__':
    main()
