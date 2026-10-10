"""Unit tests for retry with exponential backoff.

This module tests the retry_with_backoff helper: successful first attempts,
retry counts, backoff timing, exception filtering, and callbacks.
"""

import time

import pytest

from pg_mcp.models.errors import DatabaseError, LLMError, SecurityViolationError
from pg_mcp.resilience.retry import retry_with_backoff


class TestRetryWithBackoff:
    """Test suite for retry_with_backoff."""

    @pytest.mark.asyncio
    async def test_success_first_attempt(self) -> None:
        """Test no retry when the first attempt succeeds."""
        calls = 0

        async def op() -> str:
            nonlocal calls
            calls += 1
            return "ok"

        result = await retry_with_backoff(
            op,
            max_retries=3,
            retry_delay=0.01,
            backoff_factor=2.0,
            operation_name="test_op",
        )
        assert result == "ok"
        assert calls == 1

    @pytest.mark.asyncio
    async def test_retries_then_succeeds(self) -> None:
        """Test operation succeeds after transient failures."""
        calls = 0

        async def op() -> str:
            nonlocal calls
            calls += 1
            if calls < 3:
                raise LLMError("transient failure")
            return "recovered"

        result = await retry_with_backoff(
            op,
            max_retries=3,
            retry_delay=0.01,
            backoff_factor=2.0,
            operation_name="test_op",
        )
        assert result == "recovered"
        assert calls == 3

    @pytest.mark.asyncio
    async def test_raises_after_exhausted_retries(self) -> None:
        """Test the last exception propagates after all retries fail."""
        calls = 0

        async def op() -> str:
            nonlocal calls
            calls += 1
            raise LLMError(f"failure {calls}")

        with pytest.raises(LLMError, match="failure 3"):
            await retry_with_backoff(
                op,
                max_retries=2,
                retry_delay=0.01,
                backoff_factor=1.0,
                operation_name="test_op",
            )
        assert calls == 3  # 1 initial attempt + 2 retries

    @pytest.mark.asyncio
    async def test_non_retryable_exception_propagates_immediately(self) -> None:
        """Test exceptions outside retry_on are raised without retry."""
        calls = 0

        async def op() -> str:
            nonlocal calls
            calls += 1
            raise SecurityViolationError("deterministic failure")

        with pytest.raises(SecurityViolationError):
            await retry_with_backoff(
                op,
                max_retries=3,
                retry_delay=0.01,
                backoff_factor=2.0,
                retry_on=(LLMError,),
                operation_name="test_op",
            )
        assert calls == 1

    @pytest.mark.asyncio
    async def test_retry_on_filters_exception_types(self) -> None:
        """Test only the configured exception types trigger retries."""
        calls = 0

        async def op() -> str:
            nonlocal calls
            calls += 1
            if calls < 2:
                raise DatabaseError("retryable")
            return "ok"

        result = await retry_with_backoff(
            op,
            max_retries=3,
            retry_delay=0.01,
            backoff_factor=1.0,
            retry_on=(DatabaseError,),
            operation_name="test_op",
        )
        assert result == "ok"
        assert calls == 2

    @pytest.mark.asyncio
    async def test_backoff_timing(self) -> None:
        """Test delays grow exponentially between retries."""
        calls = 0

        async def op() -> str:
            nonlocal calls
            calls += 1
            if calls < 3:
                raise LLMError("transient")
            return "ok"

        start = time.perf_counter()
        await retry_with_backoff(
            op,
            max_retries=3,
            retry_delay=0.05,
            backoff_factor=2.0,
            operation_name="test_op",
        )
        elapsed = time.perf_counter() - start
        # Expected sleeps: 0.05s (first retry) + 0.10s (second retry) = 0.15s
        assert elapsed >= 0.14

    @pytest.mark.asyncio
    async def test_on_retry_callback_invoked(self) -> None:
        """Test the on_retry callback receives the error and 1-based attempt."""
        attempts: list[tuple[str, int]] = []

        async def op() -> str:
            raise LLMError("always fails")

        with pytest.raises(LLMError):
            await retry_with_backoff(
                op,
                max_retries=2,
                retry_delay=0.01,
                backoff_factor=1.0,
                on_retry=lambda e, attempt: attempts.append((str(e), attempt)),
                operation_name="test_op",
            )

        assert attempts == [("always fails", 1), ("always fails", 2)]

    @pytest.mark.asyncio
    async def test_zero_retries_single_attempt(self) -> None:
        """Test max_retries=0 performs exactly one attempt."""
        calls = 0

        async def op() -> str:
            nonlocal calls
            calls += 1
            raise LLMError("fail")

        with pytest.raises(LLMError):
            await retry_with_backoff(
                op,
                max_retries=0,
                retry_delay=0.01,
                backoff_factor=1.0,
                operation_name="test_op",
            )
        assert calls == 1
