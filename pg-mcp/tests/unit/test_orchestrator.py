"""Unit tests for QueryOrchestrator.

This module tests the orchestrator's coordination of the query pipeline,
including retry logic, error handling, and integration with all components.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from pg_mcp.config.settings import ResilienceConfig, ValidationConfig
from pg_mcp.models.errors import (
    DatabaseError,
    LLMError,
    RateLimitExceededError,
    SecurityViolationError,
    SQLParseError,
)
from pg_mcp.models.query import (
    QueryRequest,
    ResultValidationResult,
    ReturnType,
)
from pg_mcp.models.schema import ColumnInfo, DatabaseSchema, TableInfo
from pg_mcp.observability.metrics import MetricsCollector
from pg_mcp.resilience.circuit_breaker import CircuitState
from pg_mcp.resilience.rate_limiter import MultiRateLimiter
from pg_mcp.services.orchestrator import QueryOrchestrator


class TestDatabaseResolution:
    """Test database name resolution logic."""

    @pytest.fixture
    def mock_pools(self) -> dict[str, MagicMock]:
        """Create mock connection pools."""
        return {
            "db1": MagicMock(),
            "db2": MagicMock(),
        }

    @pytest.fixture
    def orchestrator(self, mock_pools: dict[str, MagicMock]) -> QueryOrchestrator:
        """Create orchestrator with mocked components."""
        return QueryOrchestrator(
            sql_generator=MagicMock(),
            sql_validator=MagicMock(),
            sql_executors={"test_db": MagicMock()},
            result_validator=MagicMock(),
            schema_cache=MagicMock(),
            pools=mock_pools,
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(),
        )

    def test_resolve_database_specified_valid(self, orchestrator: QueryOrchestrator) -> None:
        """Test resolving a specified valid database."""
        result = orchestrator._resolve_database("db1")
        assert result == "db1"

    def test_resolve_database_specified_invalid(self, orchestrator: QueryOrchestrator) -> None:
        """Test resolving a specified but invalid database."""
        with pytest.raises(DatabaseError) as exc_info:
            orchestrator._resolve_database("nonexistent")

        assert "not found" in str(exc_info.value).lower()
        assert "db1" in exc_info.value.details["available_databases"]
        assert "db2" in exc_info.value.details["available_databases"]

    def test_resolve_database_auto_select_single(self) -> None:
        """Test auto-selecting when only one database available."""
        orchestrator = QueryOrchestrator(
            sql_generator=MagicMock(),
            sql_validator=MagicMock(),
            sql_executors={"test_db": MagicMock()},
            result_validator=MagicMock(),
            schema_cache=MagicMock(),
            pools={"only_db": MagicMock()},
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(),
        )

        result = orchestrator._resolve_database(None)
        assert result == "only_db"

    def test_resolve_database_auto_select_multiple_fails(
        self, orchestrator: QueryOrchestrator
    ) -> None:
        """Test that auto-select fails when multiple databases available."""
        with pytest.raises(DatabaseError) as exc_info:
            orchestrator._resolve_database(None)

        assert "multiple databases" in str(exc_info.value).lower()
        assert "db1" in exc_info.value.details["available_databases"]

    def test_resolve_database_no_databases(self) -> None:
        """Test error when no databases configured."""
        orchestrator = QueryOrchestrator(
            sql_generator=MagicMock(),
            sql_validator=MagicMock(),
            sql_executors={"test_db": MagicMock()},
            result_validator=MagicMock(),
            schema_cache=MagicMock(),
            pools={},
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(),
        )

        with pytest.raises(DatabaseError) as exc_info:
            orchestrator._resolve_database(None)

        assert "no databases configured" in str(exc_info.value).lower()


class TestSQLGenerationWithRetry:
    """Test SQL generation with retry logic."""

    @pytest.fixture
    def mock_schema(self) -> DatabaseSchema:
        """Create mock database schema."""
        return DatabaseSchema(
            database_name="test_db",
            tables=[
                TableInfo(
                    schema_name="public",
                    table_name="users",
                    columns=[
                        ColumnInfo(
                            name="id",
                            data_type="integer",
                            is_nullable=False,
                            is_primary_key=True,
                        ),
                        ColumnInfo(
                            name="name",
                            data_type="varchar(255)",
                            is_nullable=False,
                        ),
                    ],
                )
            ],
            version="15.0",
        )

    @pytest.mark.asyncio
    async def test_generate_sql_success_first_attempt(self, mock_schema: DatabaseSchema) -> None:
        """Test successful SQL generation on first attempt."""
        # Setup mocks
        mock_generator = AsyncMock()
        mock_generator.generate.return_value = "SELECT * FROM users;"

        mock_validator = MagicMock()
        mock_validator.validate_or_raise.return_value = None  # No exception = valid

        orchestrator = QueryOrchestrator(
            sql_generator=mock_generator,
            sql_validator=mock_validator,
            sql_executors={"test_db": MagicMock()},
            result_validator=MagicMock(),
            schema_cache=MagicMock(),
            pools={"test_db": MagicMock()},
            resilience_config=ResilienceConfig(max_retries=3),
            validation_config=ValidationConfig(),
        )

        # Execute
        sql, validation_result, _tokens = await orchestrator._generate_sql_with_retry(
            question="Get all users",
            schema=mock_schema,
            request_id="test-123",
        )

        # Verify
        assert sql == "SELECT * FROM users;"
        assert validation_result.is_valid is True
        assert validation_result.is_select is True
        mock_generator.generate.assert_called_once()
        mock_validator.validate_or_raise.assert_called_once_with("SELECT * FROM users;")

    @pytest.mark.asyncio
    async def test_generate_sql_retry_on_validation_failure(
        self, mock_schema: DatabaseSchema
    ) -> None:
        """Test retry logic when validation fails."""
        # Setup mocks - first attempt fails validation, second succeeds
        mock_generator = AsyncMock()
        mock_generator.generate.side_effect = [
            "SELECT * FROM user;",  # First attempt (wrong table name)
            "SELECT * FROM users;",  # Second attempt (correct)
        ]

        mock_validator = MagicMock()
        # First call raises error, second call succeeds
        mock_validator.validate_or_raise.side_effect = [
            SQLParseError('relation "user" does not exist'),
            None,  # Success on second attempt
        ]

        orchestrator = QueryOrchestrator(
            sql_generator=mock_generator,
            sql_validator=mock_validator,
            sql_executors={"test_db": MagicMock()},
            result_validator=MagicMock(),
            schema_cache=MagicMock(),
            pools={"test_db": MagicMock()},
            resilience_config=ResilienceConfig(max_retries=3),
            validation_config=ValidationConfig(),
        )

        # Execute
        sql, validation_result, _tokens = await orchestrator._generate_sql_with_retry(
            question="Get all users",
            schema=mock_schema,
            request_id="test-123",
        )

        # Verify
        assert sql == "SELECT * FROM users;"
        assert validation_result.is_valid is True
        assert mock_generator.generate.call_count == 2
        assert mock_validator.validate_or_raise.call_count == 2

        # Verify retry included error feedback
        second_call = mock_generator.generate.call_args_list[1]
        assert second_call.kwargs["previous_attempt"] == "SELECT * FROM user;"
        assert 'relation "user" does not exist' in second_call.kwargs["error_feedback"]

    @pytest.mark.asyncio
    async def test_generate_sql_fails_after_max_retries(self, mock_schema: DatabaseSchema) -> None:
        """Test failure after exhausting all retries."""
        # Setup mocks - all attempts fail validation
        mock_generator = AsyncMock()
        mock_generator.generate.return_value = "DELETE FROM users;"

        mock_validator = MagicMock()
        mock_validator.validate_or_raise.side_effect = SecurityViolationError(
            "DELETE statements are not allowed"
        )

        orchestrator = QueryOrchestrator(
            sql_generator=mock_generator,
            sql_validator=mock_validator,
            sql_executors={"test_db": MagicMock()},
            result_validator=MagicMock(),
            schema_cache=MagicMock(),
            pools={"test_db": MagicMock()},
            resilience_config=ResilienceConfig(max_retries=2),
            validation_config=ValidationConfig(),
        )

        # Execute and verify exception
        with pytest.raises(SecurityViolationError) as exc_info:
            await orchestrator._generate_sql_with_retry(
                question="Delete all users",
                schema=mock_schema,
                request_id="test-123",
            )

        assert "DELETE statements are not allowed" in str(exc_info.value)
        # Should attempt max_retries + 1 times (initial + retries)
        assert mock_generator.generate.call_count == 3
        assert orchestrator.circuit_breaker.failure_count == 1

    @pytest.mark.asyncio
    async def test_generate_sql_circuit_breaker_open(self, mock_schema: DatabaseSchema) -> None:
        """Test that open circuit breaker prevents SQL generation."""
        orchestrator = QueryOrchestrator(
            sql_generator=AsyncMock(),
            sql_validator=MagicMock(),
            sql_executors={"test_db": MagicMock()},
            result_validator=MagicMock(),
            schema_cache=MagicMock(),
            pools={"test_db": MagicMock()},
            resilience_config=ResilienceConfig(circuit_breaker_threshold=1),
            validation_config=ValidationConfig(),
        )

        # Manually open the circuit breaker
        orchestrator.circuit_breaker._state = CircuitState.OPEN
        orchestrator.circuit_breaker._failure_count = 5

        # Attempt should fail immediately
        with pytest.raises(LLMError) as exc_info:
            await orchestrator._generate_sql_with_retry(
                question="Get all users",
                schema=mock_schema,
                request_id="test-123",
            )

        assert "temporarily unavailable" in str(exc_info.value).lower()
        assert "circuit breaker" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_generate_sql_unexpected_error(self, mock_schema: DatabaseSchema) -> None:
        """Test handling of unexpected errors during generation."""
        mock_generator = AsyncMock()
        mock_generator.generate.side_effect = RuntimeError("Unexpected error")

        orchestrator = QueryOrchestrator(
            sql_generator=mock_generator,
            sql_validator=MagicMock(),
            sql_executors={"test_db": MagicMock()},
            result_validator=MagicMock(),
            schema_cache=MagicMock(),
            pools={"test_db": MagicMock()},
            resilience_config=ResilienceConfig(max_retries=1),
            validation_config=ValidationConfig(),
        )

        with pytest.raises(LLMError) as exc_info:
            await orchestrator._generate_sql_with_retry(
                question="Get all users",
                schema=mock_schema,
                request_id="test-123",
            )

        assert "unexpectedly" in str(exc_info.value).lower()
        assert orchestrator.circuit_breaker.failure_count == 1


class TestResultValidation:
    """Test result validation logic."""

    @pytest.mark.asyncio
    async def test_validate_results_success(self) -> None:
        """Test successful result validation."""
        mock_validator = AsyncMock()
        mock_validator.validate.return_value = ResultValidationResult(
            confidence=85,
            explanation="Results match the question well",
            suggestion=None,
            is_acceptable=True,
        )

        orchestrator = QueryOrchestrator(
            sql_generator=MagicMock(),
            sql_validator=MagicMock(),
            sql_executors={"test_db": MagicMock()},
            result_validator=mock_validator,
            schema_cache=MagicMock(),
            pools={"test_db": MagicMock()},
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(enabled=True),
        )

        confidence = await orchestrator._validate_results_safely(
            question="Count users",
            sql="SELECT COUNT(*) FROM users",
            results=[{"count": 42}],
            row_count=1,
            request_id="test-123",
        )

        assert confidence == 85
        mock_validator.validate.assert_called_once()

    @pytest.mark.asyncio
    async def test_validate_results_disabled(self) -> None:
        """Test that validation is skipped when disabled."""
        mock_validator = AsyncMock()

        orchestrator = QueryOrchestrator(
            sql_generator=MagicMock(),
            sql_validator=MagicMock(),
            sql_executors={"test_db": MagicMock()},
            result_validator=mock_validator,
            schema_cache=MagicMock(),
            pools={"test_db": MagicMock()},
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(enabled=False),
        )

        confidence = await orchestrator._validate_results_safely(
            question="Count users",
            sql="SELECT COUNT(*) FROM users",
            results=[{"count": 42}],
            row_count=1,
            request_id="test-123",
        )

        assert confidence == 100
        mock_validator.validate.assert_not_called()

    @pytest.mark.asyncio
    async def test_validate_results_failure_does_not_raise(self) -> None:
        """Test that validation failures don't raise exceptions."""
        mock_validator = AsyncMock()
        mock_validator.validate.side_effect = Exception("Validation failed")

        orchestrator = QueryOrchestrator(
            sql_generator=MagicMock(),
            sql_validator=MagicMock(),
            sql_executors={"test_db": MagicMock()},
            result_validator=mock_validator,
            schema_cache=MagicMock(),
            pools={"test_db": MagicMock()},
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(enabled=True),
        )

        # Should not raise, returns the configured threshold as a neutral
        # confidence instead of fabricating a perfect score
        confidence = await orchestrator._validate_results_safely(
            question="Count users",
            sql="SELECT COUNT(*) FROM users",
            results=[{"count": 42}],
            row_count=1,
            request_id="test-123",
        )

        assert confidence == ValidationConfig().confidence_threshold


class TestExecuteQueryFlow:
    """Test complete query execution flow."""

    @pytest.fixture
    def mock_schema(self) -> DatabaseSchema:
        """Create mock database schema."""
        return DatabaseSchema(
            database_name="test_db",
            tables=[
                TableInfo(
                    schema_name="public",
                    table_name="users",
                    columns=[
                        ColumnInfo(
                            name="id",
                            data_type="integer",
                            is_nullable=False,
                            is_primary_key=True,
                        ),
                        ColumnInfo(
                            name="name",
                            data_type="varchar(255)",
                            is_nullable=False,
                        ),
                    ],
                )
            ],
            version="15.0",
        )

    @pytest.mark.asyncio
    async def test_execute_query_sql_only(self, mock_schema: DatabaseSchema) -> None:
        """Test executing query with return_type=SQL."""
        # Setup mocks
        mock_generator = AsyncMock()
        mock_generator.generate.return_value = "SELECT * FROM users;"

        mock_validator = MagicMock()
        mock_validator.validate_or_raise.return_value = None

        mock_cache = MagicMock()
        mock_cache.get.return_value = mock_schema

        orchestrator = QueryOrchestrator(
            sql_generator=mock_generator,
            sql_validator=mock_validator,
            sql_executors={"test_db": MagicMock()},
            result_validator=MagicMock(),
            schema_cache=mock_cache,
            pools={"test_db": MagicMock()},
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(),
        )

        # Execute
        request = QueryRequest(
            question="Get all users",
            database="test_db",
            return_type=ReturnType.SQL,
        )
        response = await orchestrator.execute_query(request)

        # Verify
        assert response.success is True
        assert response.generated_sql == "SELECT * FROM users;"
        assert response.validation is not None
        assert response.validation.is_valid is True
        assert response.data is None  # No execution for SQL-only
        assert response.error is None

    @pytest.mark.asyncio
    async def test_execute_query_with_results(self, mock_schema: DatabaseSchema) -> None:
        """Test executing query with return_type=RESULT."""
        # Setup mocks
        mock_generator = AsyncMock()
        mock_generator.generate.return_value = "SELECT id, name FROM users;"

        mock_validator = MagicMock()
        mock_validator.validate_or_raise.return_value = None

        mock_executor = AsyncMock()
        mock_executor.execute.return_value = (
            [
                {"id": 1, "name": "Alice"},
                {"id": 2, "name": "Bob"},
            ],
            2,  # total count
        )

        mock_result_validator = AsyncMock()
        mock_result_validator.validate.return_value = ResultValidationResult(
            confidence=90,
            explanation="Good results",
            suggestion=None,
            is_acceptable=True,
        )

        mock_cache = MagicMock()
        mock_cache.get.return_value = mock_schema

        orchestrator = QueryOrchestrator(
            sql_generator=mock_generator,
            sql_validator=mock_validator,
            sql_executors={"test_db": mock_executor},
            result_validator=mock_result_validator,
            schema_cache=mock_cache,
            pools={"test_db": MagicMock()},
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(enabled=True),
        )

        # Execute
        request = QueryRequest(
            question="Get all users",
            database="test_db",
            return_type=ReturnType.RESULT,
        )
        response = await orchestrator.execute_query(request)

        # Verify
        assert response.success is True
        assert response.generated_sql == "SELECT id, name FROM users;"
        assert response.data is not None
        assert response.data.row_count == 2
        assert len(response.data.rows) == 2
        assert response.data.columns == ["id", "name"]
        assert response.confidence == 90
        assert response.error is None

    @pytest.mark.asyncio
    async def test_execute_query_schema_not_cached(self) -> None:
        """Test loading schema when not in cache."""
        mock_schema = DatabaseSchema(
            database_name="test_db",
            tables=[],
            version="15.0",
        )

        # Setup mocks
        mock_cache = MagicMock()
        mock_cache.get.return_value = None  # Not in cache
        mock_cache.load = AsyncMock(return_value=mock_schema)

        mock_generator = AsyncMock()
        mock_generator.generate.return_value = "SELECT 1;"

        mock_validator = MagicMock()
        mock_validator.validate_or_raise.return_value = None

        mock_pool = MagicMock()

        orchestrator = QueryOrchestrator(
            sql_generator=mock_generator,
            sql_validator=mock_validator,
            sql_executors={"test_db": MagicMock()},
            result_validator=MagicMock(),
            schema_cache=mock_cache,
            pools={"test_db": mock_pool},
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(),
        )

        # Execute
        request = QueryRequest(
            question="Test query",
            database="test_db",
            return_type=ReturnType.SQL,
        )
        response = await orchestrator.execute_query(request)

        # Verify schema was loaded
        mock_cache.load.assert_called_once_with("test_db", mock_pool)
        assert response.success is True

    @pytest.mark.asyncio
    async def test_execute_query_schema_load_fails(self) -> None:
        """Test handling of schema load failure."""
        # Setup mocks
        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_cache.load = AsyncMock(side_effect=Exception("DB connection failed"))

        orchestrator = QueryOrchestrator(
            sql_generator=MagicMock(),
            sql_validator=MagicMock(),
            sql_executors={"test_db": MagicMock()},
            result_validator=MagicMock(),
            schema_cache=mock_cache,
            pools={"test_db": MagicMock()},
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(),
        )

        # Execute
        request = QueryRequest(
            question="Test query",
            database="test_db",
            return_type=ReturnType.SQL,
        )
        response = await orchestrator.execute_query(request)

        # Verify error response
        assert response.success is False
        assert response.error is not None
        assert "schema" in response.error.message.lower()
        assert response.generated_sql is None

    @pytest.mark.asyncio
    async def test_execute_query_validation_error(self) -> None:
        """Test handling of SQL validation errors."""
        mock_schema = DatabaseSchema(
            database_name="test_db",
            tables=[],
            version="15.0",
        )

        # Setup mocks
        mock_cache = MagicMock()
        mock_cache.get.return_value = mock_schema

        mock_generator = AsyncMock()
        mock_generator.generate.return_value = "DELETE FROM users;"

        mock_validator = MagicMock()
        mock_validator.validate_or_raise.side_effect = SecurityViolationError("DELETE not allowed")

        orchestrator = QueryOrchestrator(
            sql_generator=mock_generator,
            sql_validator=mock_validator,
            sql_executors={"test_db": MagicMock()},
            result_validator=MagicMock(),
            schema_cache=mock_cache,
            pools={"test_db": MagicMock()},
            resilience_config=ResilienceConfig(max_retries=1),
            validation_config=ValidationConfig(),
        )

        # Execute
        request = QueryRequest(
            question="Delete all users",
            database="test_db",
            return_type=ReturnType.SQL,
        )
        response = await orchestrator.execute_query(request)

        # Verify error response
        assert response.success is False
        assert response.error is not None
        assert "DELETE not allowed" in response.error.message
        assert response.error.code == "security_violation"

    @pytest.mark.asyncio
    async def test_execute_query_execution_error(self, mock_schema: DatabaseSchema) -> None:
        """Test handling of SQL execution errors."""
        # Setup mocks
        mock_cache = MagicMock()
        mock_cache.get.return_value = mock_schema

        mock_generator = AsyncMock()
        mock_generator.generate.return_value = "SELECT * FROM users;"

        mock_validator = MagicMock()
        mock_validator.validate_or_raise.return_value = None

        mock_executor = AsyncMock()
        mock_executor.execute.side_effect = DatabaseError("Query execution failed")

        orchestrator = QueryOrchestrator(
            sql_generator=mock_generator,
            sql_validator=mock_validator,
            sql_executors={"test_db": mock_executor},
            result_validator=MagicMock(),
            schema_cache=mock_cache,
            pools={"test_db": MagicMock()},
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(),
        )

        # Execute
        request = QueryRequest(
            question="Get all users",
            database="test_db",
            return_type=ReturnType.RESULT,
        )
        response = await orchestrator.execute_query(request)

        # Verify error response
        assert response.success is False
        assert response.error is not None
        assert "execution failed" in response.error.message.lower()
        assert response.error.code == "database_error"

    @pytest.mark.asyncio
    async def test_execute_query_unexpected_error(self, mock_schema: DatabaseSchema) -> None:
        """Test handling of unexpected errors."""
        # Setup mocks
        mock_cache = MagicMock()
        mock_cache.get.side_effect = RuntimeError("Unexpected error")

        orchestrator = QueryOrchestrator(
            sql_generator=MagicMock(),
            sql_validator=MagicMock(),
            sql_executors={"test_db": MagicMock()},
            result_validator=MagicMock(),
            schema_cache=mock_cache,
            pools={"test_db": MagicMock()},
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(),
        )

        # Execute
        request = QueryRequest(
            question="Get all users",
            database="test_db",
            return_type=ReturnType.SQL,
        )
        response = await orchestrator.execute_query(request)

        # Verify error response
        assert response.success is False
        assert response.error is not None
        assert response.error.code == "internal_error"
        assert "internal server error" in response.error.message.lower()

    @pytest.mark.asyncio
    async def test_execute_query_auto_select_database(self, mock_schema: DatabaseSchema) -> None:
        """Test auto-selecting database when only one available."""
        # Setup mocks
        mock_cache = MagicMock()
        mock_cache.get.return_value = mock_schema

        mock_generator = AsyncMock()
        mock_generator.generate.return_value = "SELECT 1;"

        mock_validator = MagicMock()
        mock_validator.validate_or_raise.return_value = None

        orchestrator = QueryOrchestrator(
            sql_generator=mock_generator,
            sql_validator=mock_validator,
            sql_executors={"only_db": MagicMock()},
            result_validator=MagicMock(),
            schema_cache=mock_cache,
            pools={"only_db": MagicMock()},  # Only one database
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(),
        )

        # Execute without specifying database
        request = QueryRequest(
            question="Test query",
            database=None,  # No database specified
            return_type=ReturnType.SQL,
        )
        response = await orchestrator.execute_query(request)

        # Verify
        assert response.success is True
        # Verify schema was fetched for auto-selected database
        mock_cache.get.assert_called_once_with("only_db")


class TestMultiDatabaseRouting:
    """Test per-database executor selection."""

    def _make_orchestrator(self, executors: dict[str, AsyncMock]) -> QueryOrchestrator:
        """Create an orchestrator with one executor per database."""
        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_schema = DatabaseSchema(database_name="db2", tables=[], version="15.0")
        mock_cache.load = AsyncMock(return_value=mock_schema)

        mock_generator = AsyncMock()
        mock_generator.generate.return_value = "SELECT 1;"

        validator = MagicMock()
        validator.validate_or_raise.return_value = None

        return QueryOrchestrator(
            sql_generator=mock_generator,
            sql_validator=validator,
            sql_executors=executors,
            result_validator=MagicMock(),
            schema_cache=mock_cache,
            pools={name: MagicMock() for name in executors},
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(),
        )

    @pytest.mark.asyncio
    async def test_request_routed_to_requested_executor(self) -> None:
        """Test a request for db2 executes against db2's executor only."""
        executor_db1 = AsyncMock()
        executor_db2 = AsyncMock()
        executor_db2.execute.return_value = ([{"v": 1}], 1)

        orchestrator = self._make_orchestrator({"db1": executor_db1, "db2": executor_db2})

        request = QueryRequest(question="q", database="db2")
        response = await orchestrator.execute_query(request)

        assert response.success is True
        executor_db2.execute.assert_called_once()
        executor_db1.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_missing_executor_rejected(self) -> None:
        """Test a request for a database without an executor fails safely."""
        executor_db1 = AsyncMock()
        orchestrator = self._make_orchestrator({"db1": executor_db1})

        request = QueryRequest(question="q", database="db2")
        response = await orchestrator.execute_query(request)

        assert response.success is False
        assert response.error is not None
        assert response.error.code == "database_error"
        assert "db1" in str(response.error.details)
        executor_db1.execute.assert_not_called()


class TestRateLimiting:
    """Test rate limiter integration in the request path."""

    def _make_orchestrator(
        self,
        rate_limiter: MultiRateLimiter,
        generator: AsyncMock | None = None,
        max_retries: int = 3,
    ) -> QueryOrchestrator:
        """Create an orchestrator wired with the given rate limiter."""
        mock_generator = generator if generator is not None else AsyncMock()
        mock_generator.generate.return_value = "SELECT 1;"

        validator = MagicMock()
        validator.validate_or_raise.return_value = None

        return QueryOrchestrator(
            sql_generator=mock_generator,
            sql_validator=validator,
            sql_executors={"db": AsyncMock()},
            result_validator=MagicMock(),
            schema_cache=MagicMock(),
            pools={"db": MagicMock()},
            resilience_config=ResilienceConfig(
                max_retries=max_retries,
                rate_limit_wait_timeout=0.1,
            ),
            validation_config=ValidationConfig(),
            rate_limiter=rate_limiter,
        )

    @pytest.mark.asyncio
    async def test_llm_generation_runs_inside_limiter(self) -> None:
        """Test LLM generation runs inside the LLM limiter slot."""
        rate_limiter = MultiRateLimiter(query_limit=1, llm_limit=1)
        orchestrator = self._make_orchestrator(rate_limiter)

        request = QueryRequest(question="q", database="db", return_type=ReturnType.SQL)
        response = await orchestrator.execute_query(request)

        assert response.success is True
        # Allow the limiter's async counter-decrement task to run
        await asyncio.sleep(0)
        assert rate_limiter.llm_limiter.active_count == 0  # released afterwards

    @pytest.mark.asyncio
    async def test_llm_rate_limit_timeout_raises_rate_limit_error(self) -> None:
        """Test LLM slot exhaustion converts to RateLimitExceededError."""
        rate_limiter = MultiRateLimiter(query_limit=1, llm_limit=1)
        orchestrator = self._make_orchestrator(rate_limiter, max_retries=0)

        # Exhaust the LLM limiter slot so generation cannot acquire it
        await rate_limiter.llm_limiter.acquire()

        with pytest.raises(RateLimitExceededError):
            await orchestrator._generate_sql_once(
                question="q", schema=MagicMock(), previous_sql=None, error_feedback=None
            )

        rate_limiter.llm_limiter.release()

    @pytest.mark.asyncio
    async def test_query_rate_limit_timeout_raises_rate_limit_error(self) -> None:
        """Test DB slot exhaustion converts to RateLimitExceededError."""
        rate_limiter = MultiRateLimiter(query_limit=1, llm_limit=1)
        orchestrator = self._make_orchestrator(rate_limiter, max_retries=0)

        await rate_limiter.query_limiter.acquire()
        with pytest.raises(RateLimitExceededError):
            await orchestrator._execute_with_resilience(
                executor=orchestrator.sql_executors["db"], sql="SELECT 1"
            )

        rate_limiter.query_limiter.release()


class TestMetricsInstrumentation:
    """Test Prometheus metrics recorded in the request path."""

    @pytest.mark.asyncio
    async def test_success_metrics_recorded(self) -> None:
        """Test query request and llm metrics on success."""
        mock_generator = AsyncMock()
        mock_generator.generate.return_value = "SELECT 1;"

        validator = MagicMock()
        validator.validate_or_raise.return_value = None

        executor = AsyncMock()
        executor.execute.return_value = ([{"v": 1}], 1)

        metrics = MetricsCollector()
        before_requests = metrics.query_requests.labels(
            status="success", database="db"
        )._value.get()
        before_llm = metrics.llm_calls.labels(operation="generate_sql")._value.get()

        orchestrator = QueryOrchestrator(
            sql_generator=mock_generator,
            sql_validator=validator,
            sql_executors={"db": executor},
            result_validator=MagicMock(),
            schema_cache=MagicMock(),
            pools={"db": MagicMock()},
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(),
            metrics=metrics,
        )

        request = QueryRequest(question="q", database="db")
        response = await orchestrator.execute_query(request)

        assert response.success is True
        assert (
            metrics.query_requests.labels(status="success", database="db")._value.get()
            == before_requests + 1
        )
        assert metrics.llm_calls.labels(operation="generate_sql")._value.get() == before_llm + 1

    @pytest.mark.asyncio
    async def test_sql_rejected_metric_on_validation_failure(self) -> None:
        """Test sql_rejected counter increments when validation rejects SQL."""
        mock_generator = AsyncMock()
        mock_generator.generate.return_value = "SELECT 1;"

        validator = MagicMock()
        validator.validate_or_raise.side_effect = SecurityViolationError("not allowed")

        metrics = MetricsCollector()
        before = metrics.sql_rejected.labels(reason="security_violation")._value.get()

        orchestrator = QueryOrchestrator(
            sql_generator=mock_generator,
            sql_validator=validator,
            sql_executors={"db": AsyncMock()},
            result_validator=MagicMock(),
            schema_cache=MagicMock(),
            pools={"db": MagicMock()},
            resilience_config=ResilienceConfig(max_retries=0),
            validation_config=ValidationConfig(),
            metrics=metrics,
        )

        request = QueryRequest(question="q", database="db", return_type=ReturnType.SQL)
        response = await orchestrator.execute_query(request)

        assert response.success is False
        assert metrics.sql_rejected.labels(reason="security_violation")._value.get() == before + 1

    @pytest.mark.asyncio
    async def test_error_status_metric_recorded(self) -> None:
        """Test query_requests counter uses the error code as status."""
        metrics = MetricsCollector()
        status_labels = metrics.query_requests.labels(status="database_error", database="unknown")
        before = status_labels._value.get()

        orchestrator = QueryOrchestrator(
            sql_generator=AsyncMock(),
            sql_validator=MagicMock(),
            sql_executors={},
            result_validator=MagicMock(),
            schema_cache=MagicMock(),
            pools={},
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(),
            metrics=metrics,
        )

        request = QueryRequest(question="q", database=None)
        response = await orchestrator.execute_query(request)

        assert response.success is False
        assert status_labels._value.get() == before + 1


class TestQuestionLengthEnforcement:
    """Test the configured question length limit."""

    @pytest.mark.asyncio
    async def test_overlong_question_rejected_before_llm(self) -> None:
        """Test questions longer than max_question_length fail without LLM usage."""
        mock_generator = AsyncMock()

        orchestrator = QueryOrchestrator(
            sql_generator=mock_generator,
            sql_validator=MagicMock(),
            sql_executors={"db": AsyncMock()},
            result_validator=MagicMock(),
            schema_cache=MagicMock(),
            pools={"db": MagicMock()},
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(max_question_length=10),
        )

        request = QueryRequest(question="x" * 11, database="db")
        response = await orchestrator.execute_query(request)

        assert response.success is False
        assert response.error is not None
        assert response.error.code == "question_too_long"
        mock_generator.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_question_at_limit_allowed(self) -> None:
        """Test questions exactly at the limit pass the length guard."""
        mock_generator = AsyncMock()
        mock_generator.generate.return_value = "SELECT 1;"

        validator = MagicMock()
        validator.validate_or_raise.return_value = None

        orchestrator = QueryOrchestrator(
            sql_generator=mock_generator,
            sql_validator=validator,
            sql_executors={"db": AsyncMock()},
            result_validator=MagicMock(),
            schema_cache=MagicMock(),
            pools={"db": MagicMock()},
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(max_question_length=10),
        )

        request = QueryRequest(question="x" * 10, database="db", return_type=ReturnType.SQL)
        response = await orchestrator.execute_query(request)

        assert response.success is True


class TestResultConfidenceReporting:
    """Test honest confidence reporting for result validation."""

    @pytest.mark.asyncio
    async def test_low_confidence_reported_and_metric_recorded(self) -> None:
        """Test low confidence is passed through and recorded, not masked."""
        mock_result_validator = AsyncMock()
        mock_result_validator.validate.return_value = ResultValidationResult(
            confidence=40,
            explanation="Results do not answer the question",
            suggestion=None,
            is_acceptable=False,
        )

        metrics = MetricsCollector()
        low_confidence_labels = metrics.result_validation_failures.labels(outcome="low_confidence")
        before = low_confidence_labels._value.get()

        orchestrator = QueryOrchestrator(
            sql_generator=AsyncMock(),
            sql_validator=MagicMock(),
            sql_executors={"db": AsyncMock()},
            result_validator=mock_result_validator,
            schema_cache=MagicMock(),
            pools={"db": MagicMock()},
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(enabled=True, confidence_threshold=70),
            metrics=metrics,
        )

        confidence = await orchestrator._validate_results_safely(
            question="q",
            sql="SELECT 1",
            results=[{"v": 1}],
            row_count=1,
            request_id="test",
        )

        assert confidence == 40  # honest score, not inflated
        assert low_confidence_labels._value.get() == before + 1

    @pytest.mark.asyncio
    async def test_validation_error_returns_threshold(self) -> None:
        """Test validation exceptions yield the threshold, not 100."""
        mock_result_validator = AsyncMock()
        mock_result_validator.validate.side_effect = Exception("LLM down")

        metrics = MetricsCollector()
        error_labels = metrics.result_validation_failures.labels(outcome="error")
        before = error_labels._value.get()

        orchestrator = QueryOrchestrator(
            sql_generator=AsyncMock(),
            sql_validator=MagicMock(),
            sql_executors={"db": AsyncMock()},
            result_validator=mock_result_validator,
            schema_cache=MagicMock(),
            pools={"db": MagicMock()},
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(enabled=True, confidence_threshold=70),
            metrics=metrics,
        )

        confidence = await orchestrator._validate_results_safely(
            question="q",
            sql="SELECT 1",
            results=[{"v": 1}],
            row_count=1,
            request_id="test",
        )

        assert confidence == 70
        assert error_labels._value.get() == before + 1

    @pytest.mark.asyncio
    async def test_validation_disabled_returns_100(self) -> None:
        """Test disabled validation returns the neutral 100 score."""
        orchestrator = QueryOrchestrator(
            sql_generator=AsyncMock(),
            sql_validator=MagicMock(),
            sql_executors={"db": AsyncMock()},
            result_validator=AsyncMock(),
            schema_cache=MagicMock(),
            pools={"db": MagicMock()},
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(enabled=False),
        )

        confidence = await orchestrator._validate_results_safely(
            question="q",
            sql="SELECT 1",
            results=[{"v": 1}],
            row_count=1,
            request_id="test",
        )
        assert confidence == 100


class TestTransientRetryIntegration:
    """Test transient-failure retry through the orchestrator."""

    @pytest.mark.asyncio
    async def test_connection_error_retried_then_succeeds(self) -> None:
        """Test database connection errors are retried with backoff."""
        from pg_mcp.models.errors import DatabaseConnectionError

        executor = AsyncMock()
        executor.execute.side_effect = [
            DatabaseConnectionError("connection refused"),
            ([{"v": 1}], 1),
        ]

        orchestrator = QueryOrchestrator(
            sql_generator=AsyncMock(),
            sql_validator=MagicMock(),
            sql_executors={"db": executor},
            result_validator=MagicMock(),
            schema_cache=MagicMock(),
            pools={"db": MagicMock()},
            resilience_config=ResilienceConfig(max_retries=2, retry_delay=0.1, backoff_factor=1.0),
            validation_config=ValidationConfig(enabled=False),
        )

        _results, count = await orchestrator._execute_with_resilience(
            executor=executor, sql="SELECT 1"
        )

        assert count == 1
        assert executor.execute.call_count == 2

    @pytest.mark.asyncio
    async def test_non_connection_db_error_not_retried(self) -> None:
        """Test deterministic database errors propagate without retry."""
        executor = AsyncMock()
        executor.execute.side_effect = DatabaseError("syntax error")

        orchestrator = QueryOrchestrator(
            sql_generator=AsyncMock(),
            sql_validator=MagicMock(),
            sql_executors={"db": executor},
            result_validator=MagicMock(),
            schema_cache=MagicMock(),
            pools={"db": MagicMock()},
            resilience_config=ResilienceConfig(max_retries=3, retry_delay=0.1, backoff_factor=1.0),
            validation_config=ValidationConfig(enabled=False),
        )

        with pytest.raises(DatabaseError):
            await orchestrator._execute_with_resilience(executor=executor, sql="SELECT bogus")
        assert executor.execute.call_count == 1
