"""Frontend for the actual C++ ragged-compile executable.

No numerical interpreter is used as a fallback for missing LLVM/CUDA tools.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

from .cache import Artifact, ArtifactCache, CacheKey, _unique_object
from .errors import CacheMismatch, CacheMiss, CompilationError, ToolchainUnavailable
from .model import DIALECT_VERSION, ProgramSpec, canonical_json


@dataclass(frozen=True)
class CompilationResult:
    output: bytes
    diagnostics: str
    emit: str
    command: tuple[str, ...]


class Compiler:
    def __init__(self, executable: str | os.PathLike[str] | None = None, *, timeout: float = 300):
        self.executable = str(executable or os.environ.get("RAGGED_COMPILE", "ragged-compile"))
        self.timeout = timeout

    def _executable(self) -> str:
        executable = shutil.which(self.executable)
        if executable is None:
            raise ToolchainUnavailable(
                f"ragged-compile not found: {self.executable}; build the C++ MLIR tools and set RAGGED_COMPILE")
        return executable

    def _compiler_hash(self) -> str:
        try:
            with open(self._executable(), "rb") as executable:
                return hashlib.file_digest(executable, "sha256").hexdigest()
        except OSError as exc:
            raise ToolchainUnavailable(f"cannot identify actual compiler binary: {exc}") from exc

    def lower(self, spec: ProgramSpec, *, emit: str = "gpu", explain: bool = False,
              features: str = "+ptx80") -> CompilationResult:
        if emit not in ("legal", "plan", "gpu", "nvvm", "ptx", "cubin"):
            raise ValueError("emit must be legal, plan, gpu, nvvm, ptx or cubin")
        with tempfile.TemporaryDirectory(prefix="ragged-compile-") as directory:
            source = Path(directory) / "input.mlir"
            output = Path(directory) / ("program.cubin" if emit == "cubin" else "output.mlir")
            source.write_text(spec.to_mlir(), encoding="utf-8")
            return self._invoke(source, output, emit, explain, spec.chip, features)

    def _invoke(self, source: Path, output: Path, emit: str, explain: bool,
                chip: str, features: str) -> CompilationResult:
        command = [self._executable(), str(source), "-o", str(output), f"--emit={emit}",
                   f"--chip={chip}", f"--features={features}"]
        if explain:
            command.append("--explain")
        try:
            result = subprocess.run(command, capture_output=True, timeout=self.timeout, check=False)
        except subprocess.TimeoutExpired as exc:
            raise CompilationError(f"ragged-compile exceeded {self.timeout:g} seconds") from exc
        except OSError as exc:
            raise ToolchainUnavailable(f"cannot execute ragged-compile: {exc}") from exc
        diagnostic = result.stderr.decode("utf-8", errors="replace")
        if result.returncode:
            raise CompilationError(f"ragged-compile failed ({result.returncode}):\n{diagnostic}")
        if not output.is_file() or not output.stat().st_size:
            raise CompilationError("ragged-compile returned success without writing a nonempty artifact")
        data = output.read_bytes()
        if emit == "cubin" and not data.startswith(b"\x7fELF"):
            raise CompilationError("--emit=cubin did not produce an ELF cubin; refusing to cache textual IR")
        return CompilationResult(data, diagnostic, emit, tuple(command))

    def toolchain(self, *, chip: str = "sm_89", features: str = "+ptx80") -> dict[str, str]:
        command = [self._executable(), "--toolchain-json", f"--chip={chip}", f"--features={features}"]
        try:
            result = subprocess.run(command, capture_output=True, timeout=self.timeout, check=False)
        except subprocess.TimeoutExpired as exc:
            raise ToolchainUnavailable("compiler toolchain probe timed out") from exc
        except OSError as exc:
            raise ToolchainUnavailable(f"cannot probe compiler toolchain: {exc}") from exc
        if result.returncode:
            raise ToolchainUnavailable("compiler toolchain probe failed: " + result.stderr.decode(errors="replace"))
        try:
            data = json.loads(result.stdout, object_pairs_hook=_unique_object)
        except (ValueError, UnicodeError) as exc:
            raise ToolchainUnavailable("compiler --toolchain-json did not produce valid JSON") from exc
        required = ("llvm_commit", "llvm_version", "ptxas_version", "dialect_version", "chip", "features")
        if not isinstance(data, dict) or any(not isinstance(data.get(field), str) or not data[field].strip()
                                             for field in required):
            raise ToolchainUnavailable("compiler toolchain report is missing measured identity fields")
        if any(data[field].lower() in ("unknown", "unavailable", "not found")
               for field in ("llvm_commit", "ptxas_version")):
            raise ToolchainUnavailable("compiler toolchain identity is not measured")
        if data["chip"] != chip or data["features"] != features:
            raise CompilationError("compiler toolchain target does not match requested chip/features")
        if data["dialect_version"] != DIALECT_VERSION:
            raise CompilationError("compiler dialect version differs from the frontend")
        return data

    def compile(self, spec: ProgramSpec, cache: ArtifactCache, *, llvm_commit: str | None = None,
                ptxas_version: str | None = None, features: str = "+ptx80"):
        """Compile a cubin and its ABI manifest, or validate an exact cache hit.

        Toolchain identities are probed from the actual compiler, including on
        cache hits. Optional caller identities are assertions, never provenance.
        A mismatched/corrupt entry is an error; only an absent entry recompiles.
        """
        from .runtime import CompiledProgram
        compiler_sha256 = self._compiler_hash()
        actual = self.toolchain(chip=spec.chip, features=features)
        for field, expected in (("llvm_commit", llvm_commit), ("ptxas_version", ptxas_version)):
            if expected is not None and actual[field] != expected:
                raise CompilationError(f"requested {field} differs from the actual compiler toolchain")
        key = CacheKey.from_spec(spec, llvm_commit=actual["llvm_commit"], ptxas_version=actual["ptxas_version"],
                                 features=features, dialect_version=actual["dialect_version"],
                                 compiler_sha256=compiler_sha256)
        if self._compiler_hash() != compiler_sha256:
            raise CompilationError("compiler executable changed during toolchain inspection; retry after the build finishes")
        try:
            cached = cache.load(key)
            if "resources" not in cached.metadata:
                raise CacheMismatch("cached compiler artifact lacks measured ptxas resources")
            return CompiledProgram(spec, cached, cache_hit=True)
        except CacheMiss:
            pass
        with tempfile.TemporaryDirectory(prefix="ragged-compile-") as directory:
            source = Path(directory) / "input.mlir"
            output = Path(directory) / "program.cubin"
            source.write_text(spec.to_mlir(), encoding="utf-8")
            result = self._invoke(source, output, "cubin", True, spec.chip, features)
            sidecar = output.with_suffix(output.suffix + ".json")
            if not sidecar.is_file():
                raise CompilationError("compiler emitted no .cubin.json kernel ABI manifest; cannot safely launch cubin")
            try:
                metadata: dict[str, Any] = json.loads(sidecar.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
            except (OSError, ValueError) as exc:
                raise CompilationError(f"invalid compiler ABI manifest: {exc}") from exc
            CompiledProgram.validate_metadata(metadata)
            if "resources" not in metadata:
                raise CompilationError("compiler emitted no measured ptxas resources; artifact will not be published")
            if canonical_json(metadata.get("toolchain")) != canonical_json(actual):
                raise CacheMismatch("cubin provenance differs from the measured compiler toolchain")
            if self._compiler_hash() != compiler_sha256:
                raise CompilationError("compiler executable changed during compilation; artifact will not be published")
            metadata["compiler_sha256"] = compiler_sha256
            # Bind manifest dimensions and scheduling to the actual source
            # before publication. Structural metadata validity alone cannot
            # establish e.g. the feature width of a scratch-free kernel.
            CompiledProgram(spec, Artifact(output, key, result.output, metadata))
            artifact = cache.publish(key, result.output, metadata)
            return CompiledProgram(spec, artifact, cache_hit=False)
