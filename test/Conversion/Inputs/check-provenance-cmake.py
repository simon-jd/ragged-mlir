"""Exercise configure rejection without relying on the installed SDK's origin."""
from pathlib import Path
import subprocess
import sys
import tempfile

module = Path(sys.argv[1]).resolve() / "cmake/LLVMProvenance.cmake"
with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    header = root / "include/llvm/Support/VCSRevision.h"
    header.parent.mkdir(parents=True)
    header.write_text("#undef LLVM_REVISION\n")
    script = root / "check.cmake"
    script.write_text(
        f'set(LLVM_INCLUDE_DIRS "{root}/include")\n'
        'set(LLVM_PACKAGE_VERSION "23.1.3")\n'
        f'include("{module}")\n'
    )
    missing = subprocess.run(["cmake", "-P", str(script)], capture_output=True, text=True)
    assert missing.returncode != 0 and "does not define LLVM_REVISION" in missing.stderr, missing
    bad_archive = root / "wrong.tar.xz"
    bad_archive.write_bytes(b"unrecognized source bytes")
    wrong = subprocess.run([
        "cmake", f"-DRAGGED_LLVM_SOURCE_ARCHIVE={bad_archive}", "-P", str(script)
    ], capture_output=True, text=True)
    assert wrong.returncode != 0 and "refusing unverified fallback" in wrong.stderr, wrong
    header.write_text('#define LLVM_REVISION "test-sdk-revision"\n')
    known = subprocess.run(["cmake", "-P", str(script)], capture_output=True, text=True)
    assert known.returncode == 0, known.stderr
