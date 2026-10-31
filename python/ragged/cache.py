"""Content-addressed cubin storage with fail-closed validation and atomic publish."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterator

from .errors import CacheError, CacheMismatch, CacheMiss, SpecializationLimit, ValidationError
from .model import DIALECT_VERSION, ProgramSpec, canonical_json


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate manifest field: {key}")
        result[key] = value
    return result


@dataclass(frozen=True)
class CacheKey:
    ir_hash: str
    dialect_version: str
    llvm_commit: str
    ptxas_version: str
    chip: str
    features: str
    strategy: str
    dtype: str
    feature_dim: int
    offsets_dtype: str
    op: str
    compiler_sha256: str = "unbound"

    def __post_init__(self) -> None:
        if not isinstance(self.ir_hash, str) or len(self.ir_hash) != 64 or any(c not in "0123456789abcdef" for c in self.ir_hash):
            raise ValidationError("ir_hash must be a SHA256 hex digest")
        if self.compiler_sha256 != "unbound" and (not isinstance(self.compiler_sha256, str)
                or len(self.compiler_sha256) != 64 or any(c not in "0123456789abcdef" for c in self.compiler_sha256)):
            raise ValidationError("compiler_sha256 must be a SHA256 hex digest or unbound for synthetic cache tests")
        for name in ("dialect_version", "llvm_commit", "ptxas_version", "features"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValidationError(f"cache key requires a nonempty {name}")
        ProgramSpec(self.op, self.dtype, self.feature_dim, self.offsets_dtype,
                    strategy=self.strategy, chip=self.chip)

    @classmethod
    def from_spec(cls, spec: ProgramSpec, *, llvm_commit: str, ptxas_version: str,
                  features: str = "+ptx80", dialect_version: str = DIALECT_VERSION,
                  compiler_sha256: str = "unbound") -> CacheKey:
        return cls(spec.ir_hash, dialect_version, llvm_commit, ptxas_version, spec.chip,
                   features, spec.strategy, spec.dtype, spec.feature_dim, spec.offsets_dtype, spec.op, compiler_sha256)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def digest(self) -> str:
        return _sha(canonical_json(self.to_dict()).encode())

    @property
    def signature(self) -> str:
        return f"{self.op}:{self.dtype}:D{self.feature_dim}:{self.offsets_dtype}:{self.chip}"


@dataclass(frozen=True)
class Artifact:
    path: Path
    key: CacheKey
    cubin: bytes
    metadata: dict[str, Any]


class ArtifactCache:
    """Linux cache; per-signature process lock protects the specialization budget.

    An artifact is visible only after both files have been fsynced and its temporary
    directory has been renamed. Readers never treat partial files as cache hits.
    Existing artifacts are immutable; corruption is surfaced for the caller to
    investigate rather than silently overwriting evidence.
    """

    SCHEMA = 1
    MAX_SPECIALIZATIONS = 16

    def __init__(self, root: str | os.PathLike[str], *, max_specializations: int | None = None):
        self.root = Path(root).expanduser()
        if max_specializations is not None:
            if type(max_specializations) is not int or max_specializations < 1:
                raise ValidationError("max_specializations must be a positive integer")
            self.MAX_SPECIALIZATIONS = max_specializations

    def _signature_dir(self, key: CacheKey) -> Path:
        return self.root / _sha(key.signature.encode())

    def entry_path(self, key: CacheKey) -> Path:
        return self._signature_dir(key) / key.digest

    @contextmanager
    def _locked(self, key: CacheKey) -> Iterator[Path]:
        directory = self._signature_dir(key)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / ".publish.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield directory
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def load(self, key: CacheKey) -> Artifact:
        path = self.entry_path(key)
        if not path.exists():
            raise CacheMiss(f"no compiled artifact for {key.digest}")
        try:
            if path.is_symlink() or not path.is_dir():
                raise CacheMismatch(f"cache entry must be a directory: {path}")
            for name in ("manifest.json", "kernel.cubin"):
                if (path / name).is_symlink():
                    raise CacheMismatch(f"cache artifact must not be a symbolic link: {name}")
            manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
            expected_fields = {"schema", "key", "cache_key", "signature", "cubin_sha256", "metadata", "metadata_sha256"}
            if not isinstance(manifest, dict) or set(manifest) != expected_fields:
                raise CacheMismatch("cache manifest schema fields differ")
            if type(manifest["schema"]) is not int or manifest["schema"] != self.SCHEMA:
                raise CacheMismatch("cache manifest schema version differs")
            # Canonical JSON preserves type distinctions (True is not integer 1).
            if canonical_json(manifest["key"]) != canonical_json(key.to_dict()):
                raise CacheMismatch("cache manifest key differs from requested compiler/toolchain/target key")
            if manifest["cache_key"] != key.digest or manifest["signature"] != key.signature:
                raise CacheMismatch("cache manifest digest or signature differs")
            if not isinstance(manifest["metadata"], dict):
                raise CacheMismatch("cache metadata must be an object")
            if _sha(canonical_json(manifest["metadata"]).encode()) != manifest["metadata_sha256"]:
                raise CacheMismatch("kernel metadata does not match manifest SHA256")
            cubin = (path / "kernel.cubin").read_bytes()
            if not cubin or _sha(cubin) != manifest["cubin_sha256"]:
                raise CacheMismatch("cubin is empty or does not match manifest SHA256")
            return Artifact(path, key, cubin, manifest["metadata"])
        except CacheError:
            raise
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise CacheMismatch(f"incomplete or invalid cache entry {path}: {exc}") from exc

    def publish(self, key: CacheKey, cubin: bytes, metadata: dict[str, Any]) -> Artifact:
        if not isinstance(cubin, bytes) or not cubin:
            raise ValidationError("cubin must be nonempty bytes from the compiler")
        if not isinstance(metadata, dict):
            raise ValidationError("artifact metadata must be a dictionary")
        # Round-trip makes the caller's mutable metadata independent of publication.
        metadata = json.loads(canonical_json(metadata))
        manifest = {"schema": self.SCHEMA, "key": key.to_dict(), "cache_key": key.digest,
                    "signature": key.signature, "cubin_sha256": _sha(cubin), "metadata": metadata,
                    "metadata_sha256": _sha(canonical_json(metadata).encode())}
        with self._locked(key) as directory:
            destination = self.entry_path(key)
            if destination.exists():
                artifact = self.load(key)
                if artifact.cubin != cubin or artifact.metadata != metadata:
                    raise CacheMismatch("refusing to replace immutable artifact with different content")
                return artifact
            entries = [p for p in directory.iterdir() if not p.name.startswith(".")]
            if len(entries) >= self.MAX_SPECIALIZATIONS:
                raise SpecializationLimit(
                    f"signature {key.signature} already has {self.MAX_SPECIALIZATIONS} specializations")
            temporary = Path(tempfile.mkdtemp(prefix=".pending-", dir=directory))
            try:
                for name, data in (("kernel.cubin", cubin),
                                   ("manifest.json", (canonical_json(manifest) + "\n").encode())):
                    with (temporary / name).open("xb") as stream:
                        stream.write(data)
                        stream.flush()
                        os.fsync(stream.fileno())
                self._fsync_dir(temporary)
                os.rename(temporary, destination)
                self._fsync_dir(directory)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
        return self.load(key)

    @staticmethod
    def _fsync_dir(path: Path) -> None:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
