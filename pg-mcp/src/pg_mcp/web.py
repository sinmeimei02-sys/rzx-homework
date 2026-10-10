"""Web demo UI for the pg-mcp natural language query pipeline.

This module wraps the query orchestrator with a small FastAPI application:
a single-page chat-style UI plus two JSON endpoints. Unlike the stdio MCP
server, startup is best-effort per component — a missing OpenAI key or an
unreachable database does not prevent the UI from loading; the problem is
reported through the API and shown in the page.

Run with:
    uvicorn pg_mcp.web:app --host 127.0.0.1 --port 8000
or:
    python -m pg_mcp.web
"""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from pg_mcp.cache.schema_cache import SchemaCache
from pg_mcp.config.settings import DatabaseConfig, Settings
from pg_mcp.db.pool import close_pools, create_pool
from pg_mcp.observability.logging import configure_logging, get_logger
from pg_mcp.observability.metrics import MetricsCollector
from pg_mcp.resilience.rate_limiter import MultiRateLimiter
from pg_mcp.services.orchestrator import QueryOrchestrator
from pg_mcp.services.result_validator import ResultValidator
from pg_mcp.services.sql_executor import SQLExecutor
from pg_mcp.services.sql_generator import SQLGenerator
from pg_mcp.services.sql_validator import SQLValidator

logger = get_logger(__name__)

_STATIC_DIR = Path(__file__).with_name("static")


class WebState:
    """Mutable application state populated during startup."""

    def __init__(self) -> None:
        self.settings: Settings | None = None
        self.config_error: str | None = None
        self.orchestrator: QueryOrchestrator | None = None
        self.pools: dict[str, Any] = {}
        self.database_status: dict[str, dict[str, str]] = {}


state = WebState()


async def _init_components() -> None:
    """Initialize settings, pools, and the query orchestrator (best-effort)."""
    # 1. Settings — may fail when OPENAI_API_KEY is missing
    try:
        settings = Settings()
    except Exception as e:
        state.config_error = f"配置加载失败：{e}"
        logger.warning(f"Settings failed to load: {e}")
        return
    state.settings = settings

    configure_logging(
        level=settings.observability.log_level,
        log_format=settings.observability.log_format,
        enable_sensitive_filter=True,
    )

    # 2. Metrics HTTP server (optional, same behavior as the stdio server)
    if settings.observability.metrics_enabled:
        try:
            from prometheus_client import start_http_server

            start_http_server(settings.observability.metrics_port)
            logger.info(f"Metrics server started on port {settings.observability.metrics_port}")
        except Exception as e:
            logger.warning(f"Metrics server failed to start: {e}")

    # 3. Connection pools — best-effort per database so one unreachable
    #    database does not take the whole UI down
    pools: dict[str, Any] = {}
    db_configs: dict[str, DatabaseConfig] = {}
    for db_config in settings.databases:
        try:
            pool = await asyncio.wait_for(
                create_pool(db_config),
                timeout=10.0,
            )
            pools[db_config.name] = pool
            db_configs[db_config.name] = db_config
            state.database_status[db_config.name] = {"status": "ok", "error": ""}
            logger.info(f"Connected to database '{db_config.name}'")
        except Exception as e:
            state.database_status[db_config.name] = {
                "status": "unavailable",
                "error": str(e),
            }
            logger.warning(f"Database '{db_config.name}' unreachable: {e}")
    state.pools = pools

    if not pools:
        state.config_error = (
            state.config_error or "没有任何可连接的数据库，请检查 DATABASE_* / DATABASES 配置"
        )
        return

    # 4. Service components (only for reachable databases)
    schema_cache = SchemaCache(settings.cache)
    for db_name, pool in pools.items():
        try:
            await schema_cache.load(db_name, pool)
        except Exception as e:
            # Schema is re-attempted per request if preloading fails
            logger.warning(f"Schema preload failed for '{db_name}': {e}")

    sql_executors = {
        name: SQLExecutor(
            pool=pool,
            security_config=settings.security,
            db_config=db_configs[name],
        )
        for name, pool in pools.items()
    }

    state.orchestrator = QueryOrchestrator(
        sql_generator=SQLGenerator(settings.openai),
        sql_validator=SQLValidator(config=settings.security),
        sql_executors=sql_executors,
        result_validator=ResultValidator(
            openai_config=settings.openai,
            validation_config=settings.validation,
        ),
        schema_cache=schema_cache,
        pools=pools,
        resilience_config=settings.resilience,
        validation_config=settings.validation,
        rate_limiter=MultiRateLimiter(
            query_limit=settings.resilience.max_concurrent_queries,
            llm_limit=settings.resilience.max_concurrent_llm_calls,
        ),
        metrics=MetricsCollector(),
    )
    logger.info("Web UI initialized")


async def _shutdown() -> None:
    """Release database connection pools."""
    if state.pools:
        try:
            await asyncio.wait_for(close_pools(state.pools, timeout=5.0), timeout=8.0)
        except Exception as e:
            logger.warning(f"Error closing pools: {e}")


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Start up / shut down the web application."""
    await _init_components()
    yield
    await _shutdown()


class QueryBody(BaseModel):
    """Request body for /api/query."""

    question: str = Field(..., min_length=1, max_length=50000)
    database: str | None = None
    return_type: str = "result"


def _error(code: str, message: str, **details: Any) -> dict[str, Any]:
    """Build an error response in the same shape as the MCP query tool."""
    return {
        "success": False,
        "error": {"code": code, "message": message, "details": details or None},
        "tokens_used": 0,
    }


app = FastAPI(title="pg-mcp Web Demo", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def index() -> FileResponse:
    """Serve the single-page query UI."""
    return FileResponse(_STATIC_DIR / "index.html")


@app.get("/api/databases")
async def databases() -> dict[str, Any]:
    """Report configuration state and per-database reachability."""
    return {
        "config_error": state.config_error,
        "ready": state.orchestrator is not None,
        "databases": [{"name": name, **status} for name, status in state.database_status.items()],
    }


@app.post("/api/query")
async def query(body: QueryBody) -> dict[str, Any]:
    """Run a natural language query through the orchestrator."""
    if body.return_type not in ("sql", "result"):
        return _error(
            "INVALID_PARAMETER",
            f"无效的 return_type：'{body.return_type}'，只能是 'sql' 或 'result'。",
            return_type=body.return_type,
        )
    if state.config_error is not None and state.orchestrator is None:
        return _error("SERVER_NOT_INITIALIZED", state.config_error)
    if state.orchestrator is None:
        return _error("SERVER_NOT_INITIALIZED", "服务尚未初始化完成，请稍后重试。")

    from pg_mcp.models.query import QueryRequest, ReturnType

    try:
        request = QueryRequest(
            question=body.question,
            database=body.database,
            return_type=ReturnType(body.return_type),
        )
    except Exception as e:
        return _error("INVALID_REQUEST", f"请求参数无效：{e}", error=str(e))

    response = await state.orchestrator.execute_query(request)
    return response.to_dict()


class WebServerConfig(BaseSettings):
    """Web server bind configuration (WEB_HOST / WEB_PORT env vars)."""

    model_config = SettingsConfigDict(env_prefix="WEB_")

    host: str = "127.0.0.1"
    port: int = 8000


def main() -> None:
    """Run the web demo server (entry point: pg-mcp-web / python -m pg_mcp.web)."""
    import uvicorn

    config = WebServerConfig()
    uvicorn.run(
        "pg_mcp.web:app",
        host=config.host,
        port=config.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
