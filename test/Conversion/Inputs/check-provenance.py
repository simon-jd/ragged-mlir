import json
import sys

report = json.load(sys.stdin)
assert report["llvm_commit"] == "0d261d1ca552c95a8f007e061c787ac7132fbcbc", report
assert report["llvm_version"] == "23.1.3", report
source = report["llvm_revision_source"]
assert source in ("llvm-vcs-header", "verified-source-archive"), report
if source == "verified-source-archive":
    assert report["llvm_source_archive_sha256"] == (
        "c44186a7762ed28954be72e5ff6df9808e0779d4f1bf014ecc4e7e211d31ee34"
    ), report
else:
    assert "llvm_source_archive_sha256" not in report, report
