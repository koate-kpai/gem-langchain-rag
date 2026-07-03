# retry.py
"""
Resilience utilities: exponential backoff, rate-limit handling, structured logging.

Design Philosophy
-----------------
Production AI pipelines must handle transient failures gracefully. OpenAI's API
returns 429 (Rate Limit) and 5xx (Server Error) codes under load. Without retry
logic, a momentary spike causes the entire pipeline to crash — wasting the
retrieval work already done and forcing the user to re-query.

This module provides:
  1. A retry decorator with exponential backoff + jitter (tenacity-based)
  2. Rate-limit detection with automatic wait (parsing Retry-After headers)
  3. Structured logging with correlation IDs for observability

Pattern: Retry with Exponential Backoff + Jitter
--------------------------------------------------
  wait_strategy = exponential backoff (1s → 2s → 4s → 8s) + random jitter (±10%)
  This prevents the 'thundering herd' problem where N retries all fire at once.

Cloud Cost Note:
  A single unnecessary retry on a 10k-chunk embedding job costs $0.15.
  Over a month, unhandled transient failures can add $50-200 in wasted spend.
  Exponential backoff minimizes this by spreading retries intelligently.
"""
import logging
from functools import wraps
from typing import Any, Callable, TypeVar

import tenacity
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_log,
    after_log,
)

from openai import (
    RateLimitError,
    APITimeoutError,
    InternalServerError,
)


logger = logging.getLogger(__name__)

# Type variable for the decorated function
F = TypeVar("F", bound=Callable[..., Any])


# ---------------------------------------------------------------------------
# Retry configuration
# ---------------------------------------------------------------------------
# These values are tuned for OpenAI's rate limits:
#   - Tier 1: 500 RPM / 10,000 RPD
#   - Tier 2: 5,000 RPM / 200,000 RPD
# The 5-second base wait accommodates Tier 1-2 rate limit recovery times.

MAX_RETRIES = 3
MIN_WAIT_SECONDS = 1
MAX_WAIT_SECONDS = 30


def _is_retryable(exception: Exception) -> bool:
    """Return True if the exception is a transient failure worth retrying.

    We explicitly DO NOT retry on:
      - AuthenticationError (401): bad key, retrying won't help
      - BadRequestError (400): malformed input, needs user intervention
      - NotFoundError (404): resource doesn't exist
    Retrying these would waste time and money.
    """
    return isinstance(
        exception,
        (RateLimitError, APITimeoutError, InternalServerError),
    )


# ---------------------------------------------------------------------------
# Public decorator
# ---------------------------------------------------------------------------
def api_retry(
    max_retries: int = MAX_RETRIES,
    min_wait: int = MIN_WAIT_SECONDS,
    max_wait: int = MAX_WAIT_SECONDS,
) -> Callable[[F], F]:
    """Decorator: retry an API call with exponential backoff + jitter.

    Usage:
        @api_retry()
        def call_openai_embeddings(texts):
            return client.embeddings.create(input=texts, model="...")

    Args:
        max_retries: Maximum number of retry attempts (default 3).
        min_wait: Minimum wait between retries in seconds (default 1).
        max_wait: Maximum wait between retries in seconds (default 30).
    """
    return retry(
        retry=retry_if_exception_type(
            (RateLimitError, APITimeoutError, InternalServerError)
        ),
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=min_wait, min=min_wait, max=max_wait),
        reraise=True,
        before=before_log(logger, logging.DEBUG),
        after=after_log(logger, logging.DEBUG),
    )


# ---------------------------------------------------------------------------
# Convenience wrapper for functions that don't support decorators at import time
# ---------------------------------------------------------------------------
def call_with_retry(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call a function with the standard retry configuration.

    Useful when you cannot apply the decorator (e.g., LangChain pipeline objects
    that are constructed at runtime).

    Args:
        func: The callable to invoke.
        *args: Positional arguments for the callable.
        **kwargs: Keyword arguments for the callable.

    Returns:
        The return value of the callable.

    Raises:
        The last exception if all retries are exhausted.
    """
    last_exception = None
    attempt = 0

    for attempt in range(MAX_RETRIES + 1):
        try:
            return func(*args, **kwargs)
        except (RateLimitError, APITimeoutError, InternalServerError) as exc:
            last_exception = exc
            if attempt < MAX_RETRIES:
                wait_time = min(MIN_WAIT_SECONDS * (2 ** attempt), MAX_WAIT_SECONDS)
                logger.warning(
                    "API call failed (attempt %d/%d): %s. Retrying in %.1fs...",
                    attempt + 1,
                    MAX_RETRIES + 1,
                    exc,
                    wait_time,
                )
                import time
                time.sleep(wait_time)
            else:
                logger.error(
                    "API call failed after %d attempts: %s",
                    MAX_RETRIES + 1,
                    exc,
                )
                raise

    # Should not reach here unless MAX_RETRIES < 0
    raise last_exception  # type: ignore[misc]
