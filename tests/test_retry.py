# test_retry.py
"""
Tests for retry.py — exponential backoff, error classification, and resilience patterns.

These tests verify:
  1. call_with_retry succeeds on the first try (no overhead for success)
  2. call_with_retry retries on transient errors (429, 500, 503)
  3. call_with_retry does NOT retry on fatal errors (401, 400, 404)
  4. All retries exhausted re-raises the last exception
  5. @api_retry decorator works correctly
"""
from unittest.mock import MagicMock

import pytest
from openai import (
    RateLimitError,
    APITimeoutError,
    InternalServerError,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
)

from retry import call_with_retry, api_retry, MAX_RETRIES

# Helper: build a proper mock response for OpenAI error constructors
# The openai library v1+ requires 'response' and 'body' keyword arguments
def _make_err_response(status_code: int = 429):
    response = MagicMock()
    response.status_code = status_code
    response.headers = {"retry-after": "1"}
    response.request = MagicMock()
    return response, {"error": {"message": "test error"}}


class TestCallWithRetry:
    def test_success_on_first_try(self):
        """Happy path: function succeeds immediately, no retries."""
        func = MagicMock(return_value="success")
        result = call_with_retry(func)
        assert result == "success"
        func.assert_called_once()

    def test_retry_on_rate_limit(self):
        """RateLimitError (429) should be retried with backoff."""
        resp, body = _make_err_response(429)
        func = MagicMock()
        func.side_effect = [RateLimitError("rate limited", response=resp, body=body), "success"]
        result = call_with_retry(func)
        assert result == "success"
        assert func.call_count == 2

    def test_retry_on_timeout(self):
        """APITimeoutError should be retried."""
        from httpx import Request
        request = Request("GET", "https://api.openai.com/v1/embeddings")
        func = MagicMock()
        func.side_effect = [APITimeoutError(request=request), "success"]
        result = call_with_retry(func)
        assert result == "success"
        assert func.call_count == 2

    def test_retry_on_server_error(self):
        """InternalServerError (5xx) should be retried."""
        resp, body = _make_err_response(500)
        func = MagicMock()
        func.side_effect = [InternalServerError("server error", response=resp, body=body), "success"]
        result = call_with_retry(func)
        assert result == "success"
        assert func.call_count == 2

    def test_does_not_retry_on_auth_error(self):
        """AuthenticationError (401) should NOT be retried — bad key won't become good."""
        resp, body = _make_err_response(401)
        func = MagicMock(side_effect=AuthenticationError("bad key", response=resp, body=body))
        with pytest.raises(AuthenticationError):
            call_with_retry(func)
        func.assert_called_once()

    def test_does_not_retry_on_bad_request(self):
        """BadRequestError (400) should NOT be retried — malformed input needs user fix."""
        resp, body = _make_err_response(400)
        func = MagicMock(side_effect=BadRequestError("bad request", response=resp, body=body))
        with pytest.raises(BadRequestError):
            call_with_retry(func)
        func.assert_called_once()

    def test_does_not_retry_on_not_found(self):
        """NotFoundError (404) should NOT be retried — resource doesn't exist."""
        resp, body = _make_err_response(404)
        func = MagicMock(side_effect=NotFoundError("not found", response=resp, body=body))
        with pytest.raises(NotFoundError):
            call_with_retry(func)
        func.assert_called_once()

    def test_all_retries_exhausted_raises(self):
        """After MAX_RETRIES + 1 failures, the last exception should be raised."""
        resp, body = _make_err_response(429)
        func = MagicMock(side_effect=RateLimitError("persistent rate limit", response=resp, body=body))
        with pytest.raises(RateLimitError):
            call_with_retry(func)
        # MAX_RETRIES retries + 1 initial attempt
        assert func.call_count == MAX_RETRIES + 1


class TestApiRetryDecorator:
    def test_decorator_retries_on_rate_limit(self):
        """@api_retry should retry RateLimitError."""
        resp, body = _make_err_response(429)
        call_count = 0

        @api_retry(max_retries=2, min_wait=0.01)
        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RateLimitError("rate limit", response=resp, body=body)
            return "ok"

        result = flaky_func()
        assert result == "ok"
        assert call_count == 2

    def test_decorator_does_not_retry_auth(self):
        """@api_retry should NOT retry AuthenticationError."""
        resp, body = _make_err_response(401)

        @api_retry(max_retries=2, min_wait=0.01)
        def flaky_func():
            raise AuthenticationError("bad key", response=resp, body=body)

        with pytest.raises(AuthenticationError):
            flaky_func()
