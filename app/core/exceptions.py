"""
app/core/exceptions.py
----------------------
Custom application exception hierarchy.

Using a custom hierarchy allows:
  1. Precise error mapping to HTTP status codes in FastAPI handlers.
  2. Structured error logging (type, message, context).
  3. Graceful degradation (e.g., GuardrailException → 422, not 500).
"""


class FinSightBaseException(Exception):
    """Root exception for all FinSight AI errors."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class IngestionError(FinSightBaseException):
    """Raised when document loading or chunking fails."""


class EmbeddingError(FinSightBaseException):
    """Raised when embedding generation fails."""


class RetrievalError(FinSightBaseException):
    """Raised when vector or keyword search fails."""


class LLMProviderError(FinSightBaseException):
    """Raised when all LLM providers are exhausted."""


class GuardrailViolation(FinSightBaseException):
    """Raised when input or output fails a safety check."""

    def __init__(self, message: str, guard_type: str, details: dict | None = None):
        super().__init__(message, details)
        self.guard_type = guard_type  # "input" | "output"


class SessionNotFoundError(FinSightBaseException):
    """Raised when a requested session ID does not exist."""


class EvaluationError(FinSightBaseException):
    """Raised when the RAGAS evaluation pipeline fails."""
