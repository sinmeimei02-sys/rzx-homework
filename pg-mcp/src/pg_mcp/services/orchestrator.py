"""Query orchestrator for coordinating the complete query flow.

This module provides the QueryOrchestrator class that coordinates all components
of the query processing pipeline: SQL generation, validation, execution, and result
validation. It implements per-database executor routing, retry logic with
exponential backoff, rate limiting, metrics instrumentation, request tracing,
and error handling.
"""

import logging
import time
from typing import Any

from asyncpg import Pool

from pg_mcp.cache.schema_cache import SchemaCache
from pg_mcp.config.settings import ResilienceConfig, ValidationConfig
from pg_mcp.models.errors import (
    DatabaseConnectionError,
    DatabaseError,
    ErrorCode,
    LLMError,
    PgMcpError,
    RateLimitExceededError,
    SchemaLoadError,
    SecurityViolationError,
    SQLParseError,
)
from pg_mcp.models.query import (
    ErrorDetail,
    QueryRequest,
    QueryResponse,
    QueryResult,
    ReturnType,
    ValidationResult,
)
from pg_mcp.observability.metrics import MetricsCollector
from pg_mcp.observability.metrics import metrics as default_metrics
from pg_mcp.observability.tracing import request_context
from pg_mcp.resilience.circuit_breaker import CircuitBreaker
from pg_mcp.resilience.rate_limiter import MultiRateLimiter
from pg_mcp.resilience.retry import retry_with_backoff
from pg_mcp.services.result_validator import ResultValidator
from pg_mcp.services.sql_executor import SQLExecutor
from pg_mcp.services.sql_generator import SQLGenerator
from pg_mcp.services.sql_validator import SQLValidator

logger = logging.getLogger(__name__)


class QueryOrchestrator:
    """Orchestrates the complete query processing pipeline.

    This class coordinates SQL generation, validation, execution, and result
    validation. It implements per-database executor selection, retry logic
    with error feedback and exponential backoff, rate limiting for LLM and
    database operations, circuit breaker pattern for fault tolerance,
    Prometheus metrics, and request tracing.

    Example:
        >>> orchestrator = QueryOrchestrator(
        ...     sql_generator=generator,
        ...     sql_validator=validator,
        ...     sql_executors={"mydb": executor},
        ...     result_validator=result_validator,
        ...     schema_cache=cache,
        ...     pools={"mydb": pool},
        ...     resilience_config=resilience_config,
        ...     validation_config=validation_config,
        ... )
        >>> response = await orchestrator.execute_query(QueryRequest(
        ...     question="How many users?",
        ...     database="mydb"
        ... ))
    """

    def __init__(
        self,
        sql_generator: SQLGenerator,
        sql_validator: SQLValidator,
        sql_executors: dict[str, SQLExecutor],
        result_validator: ResultValidator,
        schema_cache: SchemaCache,
        pools: dict[str, Pool],
        resilience_config: ResilienceConfig,
        validation_config: ValidationConfig,
        rate_limiter: MultiRateLimiter | None = None,
        metrics: MetricsCollector | None = None,
    ) -> None:
        """Initialize query orchestrator.

        Args:
            sql_generator: SQL generation service.
            sql_validator: SQL validation service.
            sql_executors: SQL execution services keyed by database name.
            result_validator: Result validation service.
            schema_cache: Schema cache instance.
            pools: Dictionary mapping database names to connection pools.
            resilience_config: Resilience configuration for retries and circuit breaker.
            validation_config: Validation configuration including thresholds.
            rate_limiter: Optional rate limiter for LLM and database operations.
            metrics: Metrics collector (defaults to the shared singleton).
        """
        self.sql_generator = sql_generator
        self.sql_validator = sql_validator
        self.sql_executors = sql_executors
        self.result_validator = result_validator
        self.schema_cache = schema_cache
        self.pools = pools
        self.resilience_config = resilience_config
        self.validation_config = validation_config
        self.rate_limiter = rate_limiter
        self.metrics = metrics if metrics is not None else default_metrics

        # Create circuit breaker for LLM calls
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=resilience_config.circuit_breaker_threshold,
            recovery_timeout=resilience_config.circuit_breaker_timeout,
        )

    async def execute_query(self, request: QueryRequest) -> QueryResponse:
        """Execute complete query flow from question to results.

        This method orchestrates the entire pipeline:
        1. Open a traced request context
        2. Enforce the configured question length limit
        3. Resolve and validate database name, select its executor
        4. Load schema from cache
        5. Generate and validate SQL with retry logic
        6. Execute SQL (if return_type == RESULT), rate-limited
        7. Validate results (optional)
        8. Return structured response

        Args:
            request: Query request containing question and parameters.

        Returns:
            QueryResponse: Complete response with SQL, results, or error information.

        Example:
            >>> response = await orchestrator.execute_query(
            ...     QueryRequest(question="Count all users", return_type="result")
            ... )
            >>> if response.success:
            ...     print(f"Found {response.data.row_count} rows")
        """
        async with request_context() as request_id:
            database_name: str | None = None
            status = "success"
            start_time = self._get_current_time_ms()

            try:
                # Step 1: Enforce question length limit before any LLM usage
                max_length = self.validation_config.max_question_length
                if len(request.question) > max_length:
                    raise PgMcpError(
                        message=(f"Question exceeds maximum length of {max_length} characters"),
                        code=ErrorCode.QUESTION_TOO_LONG,
                        details={
                            "question_length": len(request.question),
                            "max_length": max_length,
                        },
                    )

                # Step 2: Resolve database name and select its executor
                database_name = self._resolve_database(request.database)
                executor = self._select_executor(database_name)
                logger.debug(
                    "Resolved database",
                    extra={"request_id": request_id, "database": database_name},
                )

                # Step 3: Get schema from cache
                schema = self.schema_cache.get(database_name)
                if schema is None:
                    # Schema not in cache, load it
                    pool = self.pools.get(database_name)
                    if pool is None:
                        raise DatabaseError(
                            message=(
                                f"No connection pool available for database '{database_name}'"
                            ),
                            details={"database": database_name},
                        )
                    try:
                        schema = await self.schema_cache.load(database_name, pool)
                    except Exception as e:
                        raise SchemaLoadError(
                            message=(
                                f"Failed to load schema for database '{database_name}': {e!s}"
                            ),
                            details={"database": database_name, "error": str(e)},
                        ) from e

                logger.debug(
                    "Schema loaded",
                    extra={
                        "request_id": request_id,
                        "database": database_name,
                        "tables": len(schema.tables),
                    },
                )

                # Step 4: Generate and validate SQL with retry logic
                (
                    generated_sql,
                    validation_result,
                    tokens_used,
                ) = await self._generate_sql_with_retry(
                    question=request.question,
                    schema=schema,
                    request_id=request_id,
                )

                # Step 5: If return_type is SQL, return early
                if request.return_type == ReturnType.SQL:
                    logger.info(
                        "Returning SQL only",
                        extra={
                            "request_id": request_id,
                            "sql_length": len(generated_sql),
                        },
                    )
                    return QueryResponse(
                        success=True,
                        generated_sql=generated_sql,
                        validation=validation_result,
                        data=None,
                        error=None,
                        confidence=100,
                        tokens_used=tokens_used,
                    )

                # Step 6: Execute SQL (rate-limited, transient-failure retry)
                logger.debug("Executing SQL", extra={"request_id": request_id})
                execution_start = self._get_current_time_ms()

                results, total_count = await self._execute_with_resilience(
                    executor=executor,
                    sql=generated_sql,
                )

                execution_time_ms = self._get_current_time_ms() - execution_start
                logger.info(
                    "SQL executed successfully",
                    extra={
                        "request_id": request_id,
                        "row_count": total_count,
                        "execution_time_ms": execution_time_ms,
                    },
                )

                # Step 7: Validate results (non-blocking, failures don't fail the request)
                result_confidence = await self._validate_results_safely(
                    question=request.question,
                    sql=generated_sql,
                    results=results,
                    row_count=total_count,
                    request_id=request_id,
                )

                # Step 8: Build successful response
                query_result = QueryResult(
                    columns=list(results[0].keys()) if results else [],
                    rows=results,
                    row_count=len(results),  # Limited row count (after max_rows applied)
                    execution_time_ms=execution_time_ms,
                )

                return QueryResponse(
                    success=True,
                    generated_sql=generated_sql,
                    validation=validation_result,
                    data=query_result,
                    error=None,
                    confidence=result_confidence,
                    tokens_used=tokens_used,
                )

            except PgMcpError as e:
                # Handle known application errors
                status = e.code.value
                logger.warning(
                    "Query execution failed with known error",
                    extra={
                        "request_id": request_id,
                        "error_code": e.code,
                        "error_message": str(e),
                    },
                )
                return QueryResponse(
                    success=False,
                    generated_sql=None,
                    validation=None,
                    data=None,
                    error=ErrorDetail(
                        code=e.code.value,
                        message=e.message,
                        details=e.details,
                    ),
                    confidence=0,
                    tokens_used=None,
                )
            except Exception as e:
                # Handle unexpected errors
                status = ErrorCode.INTERNAL_ERROR.value
                logger.exception(
                    "Query execution failed with unexpected error",
                    extra={"request_id": request_id},
                )
                return QueryResponse(
                    success=False,
                    generated_sql=None,
                    validation=None,
                    data=None,
                    error=ErrorDetail(
                        code=ErrorCode.INTERNAL_ERROR.value,
                        message=f"Internal server error: {e!s}",
                        details={"error_type": type(e).__name__},
                    ),
                    confidence=0,
                    tokens_used=None,
                )
            finally:
                duration_seconds = (self._get_current_time_ms() - start_time) / 1000.0
                self.metrics.query_duration.observe(duration_seconds)
                self.metrics.increment_query_request(
                    status=status,
                    database=database_name or "unknown",
                )

    def _resolve_database(self, database: str | None) -> str:
        """Resolve database name from request or auto-select.

        If database is specified, validate it exists.
        If not specified and only one database available, auto-select it.

        Args:
            database: Database name from request (optional).

        Returns:
            str: Resolved database name.

        Raises:
            DatabaseError: If database is invalid or cannot be auto-selected.

        Example:
            >>> name = orchestrator._resolve_database("mydb")  # Validates "mydb" exists
            >>> name = orchestrator._resolve_database(None)  # Auto-selects if only one DB
        """
        if database is not None:
            # Validate specified database exists
            if database not in self.pools:
                raise DatabaseError(
                    message=f"Database '{database}' not found",
                    details={
                        "requested_database": database,
                        "available_databases": list(self.pools.keys()),
                    },
                )
            return database

        # Auto-select if only one database available
        available_dbs = list(self.pools.keys())
        if len(available_dbs) == 0:
            raise DatabaseError(
                message="No databases configured",
                details={},
            )
        if len(available_dbs) == 1:
            return available_dbs[0]

        # Multiple databases, must specify
        raise DatabaseError(
            message="Multiple databases available, please specify which to query",
            details={"available_databases": available_dbs},
        )

    def _select_executor(self, database_name: str) -> SQLExecutor:
        """Select the executor bound to the resolved database.

        Args:
            database_name: Resolved database name.

        Returns:
            SQLExecutor: The executor for the requested database.

        Raises:
            DatabaseError: If no executor is configured for the database.
        """
        executor = self.sql_executors.get(database_name)
        if executor is None:
            raise DatabaseError(
                message=f"No SQL executor configured for database '{database_name}'",
                details={
                    "database": database_name,
                    "configured_databases": list(self.sql_executors.keys()),
                },
            )
        return executor

    async def _generate_sql_with_retry(
        self,
        question: str,
        schema: Any,
        request_id: str,
    ) -> tuple[str, ValidationResult, int | None]:
        """Generate and validate SQL with retry logic on validation failures.

        This method implements a retry loop that:
        1. Checks circuit breaker state
        2. Generates SQL using LLM (rate-limited, with transient-error backoff)
        3. Validates the generated SQL
        4. On validation failure, retries with error feedback
        5. Records success/failure to circuit breaker and metrics

        Args:
            question: User's natural language question.
            schema: Database schema for context.
            request_id: Request ID for tracking.

        Returns:
            tuple: (generated_sql, validation_result, tokens_used)

        Raises:
            LLMError: If circuit breaker is open or generation fails.
            SecurityViolationError: If SQL fails validation after all retries.
            SQLParseError: If SQL cannot be parsed.

        Example:
            >>> sql, validation, tokens = await orchestrator._generate_sql_with_retry(
            ...     question="Count users",
            ...     schema=db_schema,
            ...     request_id="123",
            ... )
        """
        # Check circuit breaker
        if not self.circuit_breaker.allow_request():
            raise LLMError(
                message="SQL generation service is temporarily unavailable (circuit breaker open)",
                details={
                    "circuit_state": self.circuit_breaker.state,
                    "failure_count": self.circuit_breaker.failure_count,
                },
            )

        previous_sql: str | None = None
        error_feedback: str | None = None
        max_retries = self.resilience_config.max_retries
        tokens_used: int | None = None

        for attempt in range(max_retries + 1):
            try:
                logger.debug(
                    "Generating SQL",
                    extra={
                        "request_id": request_id,
                        "attempt": attempt + 1,
                        "max_retries": max_retries + 1,
                    },
                )

                # Generate SQL (rate-limited, retried on transient LLM errors)
                generated_sql = await self._generate_sql_once(
                    question=question,
                    schema=schema,
                    previous_sql=previous_sql,
                    error_feedback=error_feedback,
                )

                # Note: tokens_used would come from OpenAI response metadata if available
                # For now, we don't extract it, but it can be added later

                logger.debug(
                    "SQL generated",
                    extra={
                        "request_id": request_id,
                        "sql_length": len(generated_sql),
                    },
                )

                # Validate SQL
                try:
                    self.sql_validator.validate_or_raise(generated_sql)
                except (SecurityViolationError, SQLParseError) as validation_error:
                    self.metrics.increment_sql_rejected(validation_error.code.value)
                    if attempt < max_retries:
                        # Record as failure and retry with feedback
                        logger.warning(
                            "SQL validation failed, retrying with feedback",
                            extra={
                                "request_id": request_id,
                                "attempt": attempt + 1,
                                "error": str(validation_error),
                            },
                        )
                        previous_sql = generated_sql
                        error_feedback = str(validation_error)
                        continue
                    else:
                        # Out of retries, record failure and raise
                        self.circuit_breaker.record_failure()
                        logger.error(
                            "SQL validation failed after all retries",
                            extra={
                                "request_id": request_id,
                                "attempts": attempt + 1,
                                "error": str(validation_error),
                            },
                        )
                        raise

                # Validation successful
                self.circuit_breaker.record_success()
                logger.info(
                    "SQL generated and validated successfully",
                    extra={
                        "request_id": request_id,
                        "attempts": attempt + 1,
                    },
                )

                # Build validation result
                validation_result = ValidationResult(
                    is_valid=True,
                    is_select=True,
                    allows_data_modification=False,
                    uses_blocked_functions=[],
                    error_message=None,
                )

                return generated_sql, validation_result, tokens_used

            except LLMError:
                # Generation failed (including exhausted transient retries)
                self.circuit_breaker.record_failure()
                raise
            except RateLimitExceededError:
                # Rate limiting is not an LLM health failure; propagate as-is
                raise
            except (SecurityViolationError, SQLParseError):
                # Re-raise known validation errors (failure already recorded)
                raise
            except Exception as e:
                # Unexpected error during generation
                self.circuit_breaker.record_failure()
                logger.exception(
                    "Unexpected error during SQL generation",
                    extra={"request_id": request_id},
                )
                raise LLMError(
                    message=f"SQL generation failed unexpectedly: {e!s}",
                    details={"error_type": type(e).__name__},
                ) from e

        # Should not reach here, but just in case
        self.circuit_breaker.record_failure()
        raise LLMError(
            message="SQL generation failed after all retry attempts",
            details={"max_retries": max_retries},
        )

    async def _generate_sql_once(
        self,
        question: str,
        schema: Any,
        previous_sql: str | None,
        error_feedback: str | None,
    ) -> str:
        """Perform a single rate-limited LLM generation call.

        Transient LLM failures are retried with exponential backoff using
        the resilience configuration. Deterministic errors propagate
        immediately.

        Args:
            question: User's natural language question.
            schema: Database schema for context.
            previous_sql: Previously rejected SQL (for feedback retries).
            error_feedback: Validation error message from the previous attempt.

        Returns:
            str: The generated SQL query.

        Raises:
            LLMError: If generation fails after all transient retries.
        """

        async def _call() -> str:
            self.metrics.increment_llm_call("generate_sql")
            start = time.perf_counter()
            try:
                if self.rate_limiter is not None:
                    try:
                        async with self.rate_limiter.for_llm(
                            timeout=self.resilience_config.rate_limit_wait_timeout
                        ):
                            return await self.sql_generator.generate(
                                question=question,
                                schema=schema,
                                previous_attempt=previous_sql,
                                error_feedback=error_feedback,
                            )
                    except TimeoutError as e:
                        raise RateLimitExceededError(
                            message="Too many concurrent LLM calls, please retry later",
                            details={
                                "resource": "llm",
                                "limit": self.rate_limiter.llm_limiter.max_concurrent,
                                "wait_timeout": (self.resilience_config.rate_limit_wait_timeout),
                            },
                        ) from e
                return await self.sql_generator.generate(
                    question=question,
                    schema=schema,
                    previous_attempt=previous_sql,
                    error_feedback=error_feedback,
                )
            finally:
                self.metrics.observe_llm_latency("generate_sql", time.perf_counter() - start)

        return await retry_with_backoff(
            _call,
            max_retries=self.resilience_config.max_retries,
            retry_delay=self.resilience_config.retry_delay,
            backoff_factor=self.resilience_config.backoff_factor,
            retry_on=(LLMError,),
            operation_name="sql_generation",
        )

    async def _execute_with_resilience(
        self,
        executor: SQLExecutor,
        sql: str,
    ) -> tuple[list[dict[str, Any]], int]:
        """Execute SQL with rate limiting and transient-failure retry.

        Connection-class database errors (sqlstate 08xxx) are retried with
        exponential backoff; all other database errors propagate immediately.
        Database query duration is recorded to metrics.

        Args:
            executor: The executor bound to the target database.
            sql: Validated SQL query to execute.

        Returns:
            tuple: (results, total_row_count).

        Raises:
            DatabaseConnectionError: If connection failures persist.
            DatabaseError: If a non-transient database error occurs.
        """

        async def _call() -> tuple[list[dict[str, Any]], int]:
            start = time.perf_counter()
            try:
                if self.rate_limiter is not None:
                    try:
                        async with self.rate_limiter.for_queries(
                            timeout=self.resilience_config.rate_limit_wait_timeout
                        ):
                            return await executor.execute(sql)
                    except TimeoutError as e:
                        raise RateLimitExceededError(
                            message="Too many concurrent queries, please retry later",
                            details={
                                "resource": "queries",
                                "limit": self.rate_limiter.query_limiter.max_concurrent,
                                "wait_timeout": (self.resilience_config.rate_limit_wait_timeout),
                            },
                        ) from e
                return await executor.execute(sql)
            finally:
                self.metrics.observe_db_query_duration(time.perf_counter() - start)

        return await retry_with_backoff(
            _call,
            max_retries=self.resilience_config.max_retries,
            retry_delay=self.resilience_config.retry_delay,
            backoff_factor=self.resilience_config.backoff_factor,
            retry_on=(DatabaseConnectionError,),
            operation_name="sql_execution",
        )

    async def _validate_results_safely(
        self,
        question: str,
        sql: str,
        results: list[dict[str, Any]],
        row_count: int,
        request_id: str,
    ) -> int:
        """Validate query results with error handling (non-blocking).

        This method attempts to validate results using LLM, but failures
        don't cause the overall query to fail. The confidence score is
        reported honestly:

        - Validation disabled: returns 100 (no validation configured).
        - Validation error: returns the configured confidence threshold and
          records an ``error`` outcome metric.
        - ``is_acceptable`` False: logs a warning and records a
          ``low_confidence`` outcome metric.

        Args:
            question: User's original question.
            sql: Generated SQL query.
            results: Query results.
            row_count: Total row count.
            request_id: Request ID for tracking.

        Returns:
            int: Confidence score (0-100).

        Example:
            >>> confidence = await orchestrator._validate_results_safely(
            ...     question="Count users",
            ...     sql="SELECT COUNT(*) FROM users",
            ...     results=[{"count": 42}],
            ...     row_count=1,
            ...     request_id="123",
            ... )
        """
        if not self.validation_config.enabled:
            return 100

        try:
            logger.debug(
                "Validating results",
                extra={"request_id": request_id},
            )

            validation_result = await self.result_validator.validate(
                question=question,
                sql=sql,
                results=results,
                row_count=row_count,
            )

            if not validation_result.is_acceptable:
                logger.warning(
                    "Result validation confidence below threshold",
                    extra={
                        "request_id": request_id,
                        "confidence": validation_result.confidence,
                        "threshold": self.validation_config.confidence_threshold,
                        "explanation": validation_result.explanation,
                    },
                )
                self.metrics.increment_result_validation_failure("low_confidence")

            logger.info(
                "Result validation completed",
                extra={
                    "request_id": request_id,
                    "confidence": validation_result.confidence,
                    "is_acceptable": validation_result.is_acceptable,
                },
            )

            return validation_result.confidence

        except Exception as e:
            # Log but don't fail the query; report the threshold as a
            # neutral confidence instead of fabricating a perfect score.
            logger.warning(
                "Result validation failed, using threshold as default confidence",
                extra={
                    "request_id": request_id,
                    "error": str(e),
                },
            )
            self.metrics.increment_result_validation_failure("error")
            return self.validation_config.confidence_threshold

    @staticmethod
    def _get_current_time_ms() -> float:
        """Get current time in milliseconds.

        Returns:
            float: Current time in milliseconds since epoch.
        """
        return time.time() * 1000
