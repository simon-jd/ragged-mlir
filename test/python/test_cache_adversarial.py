"""Independent executable-cache fault tests; synthetic bytes never reach CUDA."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields
import json
from pathlib import Path

import pytest

from ragged.cache import ArtifactCache, CacheKey
from ragged.errors import CacheMismatch, CacheMiss, SpecializationLimit
from ragged.model import Hints, ProgramSpec


def make_key(max_len=16):
    return CacheKey.from_spec(
        ProgramSpec("softmax", "f32", 16, "i64", hints=Hints(max_len=max_len)),
        llvm_commit="test-llvm-commit", ptxas_version="test-ptxas-version",
    )


def publish(cache, key):
    return cache.publish(key, b"synthetic-cache-test-payload", {"kernel": "test_kernel"})


@pytest.mark.parametrize("field", [field.name for field in fields(CacheKey)])
def test_each_key_field_is_checked_before_cache_hit(tmp_path, field):
    cache, key = ArtifactCache(tmp_path), make_key()
    artifact = publish(cache, key)
    manifest_path = artifact.path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    value = manifest["key"][field]
    manifest["key"][field] = value+1 if type(value) is int else value+"-corrupt"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(CacheMismatch, match="key differs"):
        cache.load(key)


@pytest.mark.parametrize("damage", ["missing_manifest", "missing_cubin", "truncated_json", "altered_cubin", "empty_cubin", "extra_manifest_field", "bool_schema", "wrong_signature", "wrong_digest", "nan_metadata"])
def test_partial_and_corrupt_entries_fail_closed(tmp_path, damage):
    cache, key = ArtifactCache(tmp_path), make_key()
    artifact = publish(cache, key)
    manifest_path, cubin_path = artifact.path / "manifest.json", artifact.path / "kernel.cubin"
    manifest = json.loads(manifest_path.read_text())
    if damage == "missing_manifest":
        manifest_path.unlink()
    elif damage == "missing_cubin":
        cubin_path.unlink()
    elif damage == "truncated_json":
        manifest_path.write_text('{"schema":')
    elif damage == "altered_cubin":
        cubin_path.write_bytes(b"altered")
    elif damage == "empty_cubin":
        cubin_path.write_bytes(b"")
    else:
        if damage == "extra_manifest_field":
            manifest["unexpected"] = 1
        elif damage == "bool_schema":
            manifest["schema"] = True
        elif damage == "wrong_signature":
            manifest["signature"] += ":bad"
        elif damage == "wrong_digest":
            manifest["cache_key"] = "0" * 64
        elif damage == "nan_metadata":
            manifest["metadata"]["bad"] = float("nan")
        manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(CacheMismatch):
        cache.load(key)


@pytest.mark.parametrize("part", ["manifest.json", "kernel.cubin"])
def test_symlink_artifact_is_rejected(tmp_path, part):
    cache, key = ArtifactCache(tmp_path), make_key()
    artifact = publish(cache, key)
    original = artifact.path / part
    moved = tmp_path / (part + ".outside")
    original.rename(moved)
    original.symlink_to(moved)
    with pytest.raises(CacheMismatch, match="symbolic link"):
        cache.load(key)


def test_interrupted_atomic_publication_never_creates_hit(tmp_path, monkeypatch):
    cache, key = ArtifactCache(tmp_path), make_key()

    def interrupted(*args):
        raise OSError("simulated interruption before atomic rename")

    monkeypatch.setattr("ragged.cache.os.rename", interrupted)
    with pytest.raises(OSError, match="simulated interruption"):
        publish(cache, key)
    with pytest.raises(CacheMiss):
        cache.load(key)
    assert not list(tmp_path.rglob(".pending-*"))


def test_parallel_publish_enforces_signature_limit(tmp_path):
    cache = ArtifactCache(tmp_path)

    def attempt(max_len):
        key = make_key(max_len)
        try:
            publish(cache, key)
            return key
        except SpecializationLimit:
            return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(attempt, range(20)))
    accepted = [key for key in results if key is not None]
    assert len(accepted) == 16
    assert sum(result is None for result in results) == 4
    for key in accepted:
        assert cache.load(key).key == key
    assert len(list(tmp_path.rglob("manifest.json"))) == 16


def test_parallel_identical_publication_is_idempotent(tmp_path):
    cache, key = ArtifactCache(tmp_path), make_key()
    with ThreadPoolExecutor(max_workers=8) as pool:
        artifacts = list(pool.map(lambda _: publish(cache, key), range(16)))
    assert len({artifact.path for artifact in artifacts}) == 1
    assert len(list(tmp_path.rglob("manifest.json"))) == 1


def test_publication_copies_metadata_and_refuses_replacement(tmp_path):
    cache, key = ArtifactCache(tmp_path), make_key()
    metadata = {"kernels": [{"name": "original"}]}
    cache.publish(key, b"synthetic", metadata)
    metadata["kernels"][0]["name"] = "mutated"
    assert cache.load(key).metadata == {"kernels": [{"name": "original"}]}
    with pytest.raises(CacheMismatch, match="immutable"):
        cache.publish(key, b"different", {"kernels": [{"name": "original"}]})


def test_metadata_mutation_cannot_change_launch_schedule(tmp_path):
    cache, key = ArtifactCache(tmp_path), make_key()
    artifact = publish(cache, key)
    manifest_path = artifact.path / "manifest.json"
    data = json.loads(manifest_path.read_text())
    data["metadata"]["kernel"] = "different_but_valid_symbol"
    data["metadata"]["alias_safe"] = True
    manifest_path.write_text(json.dumps(data))
    with pytest.raises(CacheMismatch, match="metadata"):
        cache.load(key)
