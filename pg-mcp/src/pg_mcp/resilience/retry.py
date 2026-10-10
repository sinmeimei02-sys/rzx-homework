"""Retry strategy with exponential backoff for transient failures.

This module provides a reusable async retry helper that honors the
``ResilienceConfig`` settings (max_retries, retry_delay, backoff_factor).
Only transient exception types are retried; deterministic failures
(e.g. security violations) propagate immediately.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


async def retry_with_backoff(
    func: Callable[[], Awaitable[T]],
    *,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    backoff_factor: float = 2.0,
    retry_on: tuple[type[Exception], ...] = (Exception,),
    on_retry: Callable[[Exception, int], None] | None = None,
    operation_name: str = "operation",
) -> T:
    """Execute an async operation with retry and exponential backoff.

    The operation is attempted up to ``max_retries + 1`` times. The delay
    before retry ``n`` (0-based) is ``retry_delay * backoff_factor ** n``.
    Exceptions not listed in ``retry_on`` propagate immediately without
    being retried.

    Args:
        func: Zero-argument async callable to execute.
        max_retries: Maximum number of retries after the first attempt.
        retry_delay: Base delay in seconds before the first retry.
        backoff_factor: Multiplier applied to the delay after each retry.
        retry_on: Exception types that should trigger a retry.
        on_retry: Optional callback invoked as ``on_retry(error, attempt)``
            before each retry sleep (attempt is 1-based).
        operation_name: Human-readable name used in log messages.

    Returns:
        T: The result of the first successful call.

    Raises:
        Exception: The last raised exception if all attempts fail, or any
            non-retryable exception immediately.

    Example:
        >>> result = await retry_with_backoff(
        ...     lambda: generate_sql(question),
        ...     max_retries=3,
        ...     retry_delay=1.0,
        ...     backoff_factor=2.0,
        ...     retry_on=(LLMError,),
        ...     operation_name="sql_generation",
        ... )
    """
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            return await func()
        except retry_on as e:
            last_error = e
            if attempt >= max_retries:
                logger.error(
                    "%s failed after %d attempt(s), giving up",
                    operation_name,
                    attempt + 1,
                )
                raise
            delay = retry_delay * (backoff_factor**attempt)
            logger.warning(
                "%s failed (attempt %d/%d), retrying in %.2fs: %s",
                operation_name,
                attempt + 1,
                max_retries + 1,
                delay,
                e,
            )
            if on_retry is not None:
                on_retry(e, attempt + 1)
            await asyncio.sleep(delay)

    # Unreachable: the loop either returns or raises on the final attempt.
    raise RuntimeError(f"{operation_name} failed after retries: {last_error}")
