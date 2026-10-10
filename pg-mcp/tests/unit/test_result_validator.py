"""Unit tests for the result validation service.

These tests mock the OpenAI client so they run deterministically without
network access.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from pg_mcp.config.settings import OpenAIConfig, ValidationConfig
from pg_mcp.models.errors import LLMError, LLMTimeoutError, LLMUnavailableError
from pg_mcp.services.result_validator import ResultValidator


def _make_validator() -> ResultValidator:
    """Create a validator with a mocked OpenAI client."""
    validator = ResultValidator(
        openai_config=OpenAIConfig(api_key="sk-test"),
        validation_config=ValidationConfig(confidence_threshold=70),
    )
    validator.client = MagicMock()
    validator.client.chat.completions.create = AsyncMock()
    return validator


def _completion(content: str | None) -> MagicMock:
    """Build a fake ChatCompletion response object."""
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=content))]
    return response


class TestResultValidator:
    """Test suite for ResultValidator."""

    @pytest.mark.asyncio
    async def test_disabled_returns_acceptable_without_api_call(self) -> None:
        """Test disabled validation short-circuits with confidence 100."""
        validator = _make_validator()
        validator.validation_config = ValidationConfig(enabled=False)

        result = await validator.validate(
            question="q", sql="SELECT 1", results=[{"v": 1}], row_count=1
        )

        assert result.confidence == 100
        assert result.is_acceptable is True
        validator.client.chat.completions.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_high_confidence_acceptable(self) -> None:
        """Test a high-confidence JSON response is acceptable."""
        validator = _make_validator()
        validator.client.chat.completions.create.return_value = _completion(
            '{"confidence": 90, "explanation": "Matches", "suggestion": null}'
        )

        result = await validator.validate(
            question="q", sql="SELECT 1", results=[{"v": 1}], row_count=1
        )

        assert result.confidence == 90
        assert result.is_acceptable is True
        assert result.explanation == "Matches"

    @pytest.mark.asyncio
    async def test_low_confidence_not_acceptable(self) -> None:
        """Test a low-confidence response is marked not acceptable."""
        validator = _make_validator()
        validator.client.chat.completions.create.return_value = _completion(
            '{"confidence": 30, "explanation": "Mismatch", "suggestion": "Check filters"}'
        )

        result = await validator.validate(
            question="q", sql="SELECT 1", results=[{"v": 1}], row_count=1
        )

        assert result.confidence == 30
        assert result.is_acceptable is False
        assert result.suggestion == "Check filters"

    @pytest.mark.asyncio
    async def test_invalid_json_returns_moderate_confidence(self) -> None:
        """Test unparseable LLM output yields confidence 60 and a warning."""
        validator = _make_validator()
        validator.client.chat.completions.create.return_value = _completion("not json at all")

        result = await validator.validate(
            question="q", sql="SELECT 1", results=[{"v": 1}], row_count=1
        )

        assert result.confidence == 60
        assert result.is_acceptable is False

    @pytest.mark.asyncio
    async def test_out_of_range_confidence_clamped(self) -> None:
        """Test float confidence outside 0-100 is clamped to bounds."""
        validator = _make_validator()
        validator.client.chat.completions.create.return_value = _completion(
            '{"confidence": 250, "explanation": "x"}'
        )

        result = await validator.validate(
            question="q", sql="SELECT 1", results=[{"v": 1}], row_count=1
        )
        assert result.confidence == 100

    @pytest.mark.asyncio
    async def test_non_numeric_confidence_defaults_50(self) -> None:
        """Test non-numeric confidence falls back to the 50 default."""
        validator = _make_validator()
        validator.client.chat.completions.create.return_value = _completion(
            '{"confidence": "high", "explanation": "x"}'
        )

        result = await validator.validate(
            question="q", sql="SELECT 1", results=[{"v": 1}], row_count=1
        )
        assert result.confidence == 50

    @pytest.mark.asyncio
    async def test_empty_content_raises_llm_error(self) -> None:
        """Test empty message content raises LLMError."""
        validator = _make_validator()
        validator.client.chat.completions.create.return_value = _completion(None)

        with pytest.raises(LLMError, match="empty message content"):
            await validator.validate(question="q", sql="SELECT 1", results=[{"v": 1}], row_count=1)

    @pytest.mark.asyncio
    async def test_no_choices_raises_llm_error(self) -> None:
        """Test a response without choices raises LLMError."""
        validator = _make_validator()
        response = MagicMock()
        response.choices = []
        validator.client.chat.completions.create.return_value = response

        with pytest.raises(LLMError, match="empty response"):
            await validator.validate(question="q", sql="SELECT 1", results=[{"v": 1}], row_count=1)

    @pytest.mark.asyncio
    async def test_timeout_raises_llm_timeout_error(self) -> None:
        """Test TimeoutError from the API is wrapped in LLMTimeoutError."""
        validator = _make_validator()
        validator.client.chat.completions.create.side_effect = TimeoutError("slow")

        with pytest.raises(LLMTimeoutError, match="timed out"):
            await validator.validate(question="q", sql="SELECT 1", results=[{"v": 1}], row_count=1)

    @pytest.mark.asyncio
    async def test_auth_error_raises_llm_unavailable(self) -> None:
        """Test authentication failures are wrapped in LLMUnavailableError."""
        validator = _make_validator()
        validator.client.chat.completions.create.side_effect = Exception("invalid api_key provided")

        with pytest.raises(LLMUnavailableError, match="authentication"):
            await validator.validate(question="q", sql="SELECT 1", results=[{"v": 1}], row_count=1)

    @pytest.mark.asyncio
    async def test_rate_limit_raises_llm_unavailable(self) -> None:
        """Test rate limit errors are wrapped in LLMUnavailableError."""
        validator = _make_validator()
        validator.client.chat.completions.create.side_effect = Exception("rate_limit exceeded")

        with pytest.raises(LLMUnavailableError, match="rate limit"):
            await validator.validate(question="q", sql="SELECT 1", results=[{"v": 1}], row_count=1)

    @pytest.mark.asyncio
    async def test_generic_error_raises_llm_error(self) -> None:
        """Test unexpected API errors are wrapped in LLMError."""
        validator = _make_validator()
        validator.client.chat.completions.create.side_effect = RuntimeError("boom")

        with pytest.raises(LLMError, match="Result validation failed"):
            await validator.validate(question="q", sql="SELECT 1", results=[{"v": 1}], row_count=1)
