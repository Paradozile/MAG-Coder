"""Generation adapter errors."""

from __future__ import annotations


class GenerationError(RuntimeError):
    """Base class for generation adapter errors."""


class GeneratorConfigError(GenerationError):
    """Raised when a generator is missing required configuration."""


class MissingAPIKeyError(GeneratorConfigError):
    """Raised when a configured API key environment variable is unavailable."""


class GeneratorHTTPError(GenerationError):
    """Raised when a remote HTTP generation request fails."""
