"""Unit tests for request tracing and context propagation."""

import logging

import pytest

from pg_mcp.observability.tracing import (
    TracingLogger,
    clear_request_id,
    generate_request_id,
    get_request_id,
    request_context,
    set_request_id,
    trace_async,
    trace_sync,
)


class TestRequestIdContext:
    """Test request ID generation and context management."""

    def test_generate_request_id_is_uuid(self) -> None:
        """Test generated request IDs are UUID strings."""
        request_id = generate_request_id()
        assert len(request_id) == 36
        assert request_id.count("-") == 4
        assert generate_request_id() != request_id

    def test_set_and_get_request_id(self) -> None:
        """Test setting a request ID makes it visible in context."""
        set_request_id("test-id")
        try:
            assert get_request_id() == "test-id"
        finally:
            clear_request_id()
        assert get_request_id() is None

    @pytest.mark.asyncio
    async def test_request_context_generates_id(self) -> None:
        """Test request_context generates and clears an ID."""
        assert get_request_id() is None
        async with request_context() as request_id:
            assert request_id is not None
            assert get_request_id() == request_id
        assert get_request_id() is None

    @pytest.mark.asyncio
    async def test_request_context_explicit_id(self) -> None:
        """Test request_context honors an explicit request ID."""
        async with request_context("custom-id") as request_id:
            assert request_id == "custom-id"
            assert get_request_id() == "custom-id"
        assert get_request_id() is None

    @pytest.mark.asyncio
    async def test_request_context_propagates_through_tasks(self) -> None:
        """Test the contextvar propagates into child coroutines."""
        seen: dict[str, str | None] = {}

        async def child() -> None:
            seen["request_id"] = get_request_id()

        async with request_context("parent-id"):
            await child()

        assert seen["request_id"] == "parent-id"

    @pytest.mark.asyncio
    async def test_request_context_clears_on_exception(self) -> None:
        """Test the context is cleared even when the body raises."""
        with pytest.raises(RuntimeError):
            async with request_context("boom-id"):
                raise RuntimeError("failure")
        assert get_request_id() is None


class TestTraceDecorators:
    """Test trace_async and trace_sync decorators."""

    @pytest.mark.asyncio
    async def test_trace_async_returns_result(self) -> None:
        """Test the async decorator passes through results and context."""

        @trace_async(operation="op")
        async def work(value: int) -> int:
            assert get_request_id() == "tid"
            return value * 2

        async with request_context("tid"):
            assert await work(21) == 42

    @pytest.mark.asyncio
    async def test_trace_async_without_context(self) -> None:
        """Test the async decorator works outside a request context."""

        @trace_async()
        async def work() -> str:
            return "ok"

        assert await work() == "ok"

    def test_trace_sync_returns_result(self) -> None:
        """Test the sync decorator passes through results and context."""

        @trace_sync(operation="sync_op")
        def work(value: int) -> int:
            assert get_request_id() == "tid"
            return value + 1

        set_request_id("tid")
        try:
            assert work(1) == 2
        finally:
            clear_request_id()

    def test_trace_sync_without_context(self) -> None:
        """Test the sync decorator works outside a request context."""

        @trace_sync()
        def work() -> str:
            return "ok"

        assert work() == "ok"


class TestTracingLogger:
    """Test TracingLogger request ID injection."""

    @pytest.mark.asyncio
    async def test_log_records_include_request_id(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test log records emitted inside a context carry the request ID."""
        tracing = TracingLogger("test.tracing")
        with caplog.at_level(logging.INFO, logger="test.tracing"):
            async with request_context("ctx-id"):
                tracing.info("hello %s", "world")

        record = caplog.records[0]
        assert record.request_id == "ctx-id"  # type: ignore[attr-defined]
        assert record.getMessage() == "hello world"

    @pytest.mark.asyncio
    async def test_log_records_without_context(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test logging outside a context still works."""
        tracing = TracingLogger("test.tracing")
        with caplog.at_level(logging.WARNING, logger="test.tracing"):
            tracing.warning("no context")

        assert caplog.records[0].getMessage() == "no context"

    @pytest.mark.asyncio
    async def test_exception_logging(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test the exception helper logs the active exception."""
        tracing = TracingLogger("test.tracing")
        with caplog.at_level(logging.ERROR, logger="test.tracing"):
            try:
                raise ValueError("kaboom")
            except ValueError:
                tracing.exception("caught")

        assert caplog.records[0].exc_info is not None
