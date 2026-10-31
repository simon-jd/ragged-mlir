"""Errors distinguish invalid programs, corrupt artifacts and missing tools."""


class RaggedError(RuntimeError):
    pass


class ValidationError(RaggedError, ValueError):
    def __init__(self, message, *, condition=None, index=None, errors=None):
        super().__init__(message)
        self.condition = condition
        self.index = index
        self.errors = dict(errors or {})


class CacheError(RaggedError):
    pass


class CacheMiss(CacheError):
    pass


class CacheMismatch(CacheError):
    pass


class SpecializationLimit(CacheError):
    pass


class ToolchainUnavailable(RaggedError):
    pass


class CompilationError(RaggedError):
    pass


class RuntimeUnavailable(RaggedError):
    pass
