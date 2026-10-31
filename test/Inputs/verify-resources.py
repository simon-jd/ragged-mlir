"""Check real assembler measurements and rejection before output publication."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

compiler, source, temporary = sys.argv[1:]
root = Path(temporary)
root.mkdir(exist_ok=True)

def run(source, stage, target, success=True, env=None):
    result = subprocess.run([compiler, str(source), '--emit='+stage, '-o', str(target)], capture_output=True, text=True, env=env)
    assert (result.returncode == 0) is success, result.stderr
    return result

target = root/'program.cubin'
run(source, 'cubin', target)
metadata = json.loads(Path(str(target)+'.json').read_text())
resources = metadata['resources']
assert resources['status'] == 'measured' and resources['source'] == 'ptxas'
entries = metadata['kernels'] + metadata['validation']['kernels']
assert [r['name'] for r in resources['kernels']] == [k['name'] for k in entries]
for measured, launch in zip(resources['kernels'], entries):
    threads = launch['block'][0]*launch['block'][1]*launch['block'][2]
    assert measured['threads_per_cta'] == threads
    assert measured['registers_per_cta'] == measured['registers_per_thread']*threads
    assert measured['shared_bytes_per_cta'] == measured['static_shared_bytes']+launch['shared_bytes']
    assert all(type(measured[k]) is int and measured[k] >= 0 for k in measured if k != 'name')
run(source, 'ptx', root/'program.ptx')
unmeasured = json.loads((root/'program.ptx.json').read_text())['resources']
assert unmeasured == {'status':'unmeasured', 'source':None, 'kernels':[], 'requested':resources['requested']}
run(source, 'gpu', root/'program.gpu')
assert 'ragged.resources' in (root/'program.gpu').read_text() and 'status = "unmeasured"' in (root/'program.gpu').read_text()
original = Path(source).read_text()
saved = {str(p):p.read_bytes() for p in [target, Path(str(target)+'.json'), Path(str(target)+'.ptxas.log')]}
for field, old in [('regs_budget',65536), ('smem_per_cta',49152)]:
    bad = root/(field+'.mlir')
    bad.write_text(original.replace(f'{field} = {old}', f'{field} = 1'))
    for output in [target, root/(field+'.cubin')]:
        result = run(bad, 'cubin', output, False)
        assert 'ptxas resource validation failed' in result.stderr and 'exceeding '+field in result.stderr
        if output != target:
            assert not output.exists() and not Path(str(output)+'.json').exists() and not Path(str(output)+'.ptxas.log').exists()
    assert all(Path(p).read_bytes() == content for p,content in saved.items())

# A successful assembler with contradictory duplicate fields is insufficient
# evidence. In particular the first smem clause must not hide a larger second.
fake = root/'fault-bin'
fake.mkdir(exist_ok=True)
wrapper = fake/'ptxas'
wrapper.write_text(f'''#!{sys.executable}
import re, subprocess, sys
result = subprocess.run([{shutil.which('ptxas')!r}, *sys.argv[1:]], capture_output=True)
log = result.stderr.decode()
if '--version' not in sys.argv:
    log, count = re.subn(r'(Used [0-9]+ registers[^\\n]*)', r'\\1, 0 bytes smem, 999999 bytes smem', log, count=1)
    assert count == 1
sys.stdout.buffer.write(result.stdout)
sys.stderr.write(log)
sys.exit(result.returncode)
''')
wrapper.chmod(0o755)
env = dict(os.environ, PATH=str(fake)+os.pathsep+os.environ['PATH'])
result = run(source, 'cubin', target, False, env)
assert 'duplicate or invalid shared memory record' in result.stderr
assert all(Path(p).read_bytes() == content for p,content in saved.items())
