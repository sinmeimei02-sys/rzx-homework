"""Unit tests for database connection pool management.

These tests mock asyncpg so they run without a real PostgreSQL server.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pg_mcp.config.settings import DatabaseConfig
from pg_mcp.db.pool import close_pools, create_pool, create_pools


class TestCreatePool:
    """Test suite for create_pool."""

    @pytest.mark.asyncio
    async def test_create_pool_passes_config(self) -> None:
        """Test pool creation forwards all configuration parameters."""
        config = DatabaseConfig(
            host="db.host",
            port=5433,
            name="mydb",
            user="u",
            password="p",
            min_pool_size=2,
            max_pool_size=9,
            pool_timeout=15.0,
            command_timeout=45.0,
        )
        sentinel = MagicMock()

        with patch(
            "pg_mcp.db.pool.asyncpg.create_pool", new=AsyncMock(return_value=sentinel)
        ) as mock_create:
            pool = await create_pool(config)

        assert pool is sentinel
        kwargs = mock_create.call_args.kwargs
        assert kwargs["host"] == "db.host"
        assert kwargs["port"] == 5433
        assert kwargs["database"] == "mydb"
        assert kwargs["user"] == "u"
        assert kwargs["password"] == "p"
        assert kwargs["min_size"] == 2
        assert kwargs["max_size"] == 9
        assert kwargs["timeout"] == 15.0
        assert kwargs["command_timeout"] == 45.0

    @pytest.mark.asyncio
    async def test_create_pool_raises_when_none_returned(self) -> None:
        """Test a None pool result raises RuntimeError."""
        config = DatabaseConfig(name="mydb")

        with (
            patch("pg_mcp.db.pool.asyncpg.create_pool", new=AsyncMock(return_value=None)),
            pytest.raises(RuntimeError, match="mydb"),
        ):
            await create_pool(config)


class TestCreatePools:
    """Test suite for create_pools."""

    @pytest.mark.asyncio
    async def test_creates_pool_per_database(self) -> None:
        """Test one pool is created per configured database."""
        configs = [
            DatabaseConfig(name="db1", host="h1"),
            DatabaseConfig(name="db2", host="h2"),
        ]
        pools_by_name = {"db1": MagicMock(name="pool1"), "db2": MagicMock(name="pool2")}

        async def fake_create(config: DatabaseConfig) -> MagicMock:
            return pools_by_name[config.name]

        with patch("pg_mcp.db.pool.create_pool", new=fake_create):
            pools = await create_pools(configs)

        assert set(pools.keys()) == {"db1", "db2"}
        assert pools["db1"] is pools_by_name["db1"]
        assert pools["db2"] is pools_by_name["db2"]


class TestClosePools:
    """Test suite for close_pools."""

    @pytest.mark.asyncio
    async def test_graceful_close(self) -> None:
        """Test pools close gracefully when close() completes in time."""
        pool = MagicMock()
        pool.close = AsyncMock()

        await close_pools({"db": pool}, timeout=1.0)

        pool.close.assert_awaited_once()
        pool.terminate.assert_not_called()

    @pytest.mark.asyncio
    async def test_force_terminate_on_timeout(self) -> None:
        """Test pools are terminated when graceful close times out."""

        async def slow_close() -> None:
            await asyncio.sleep(5.0)

        pool = MagicMock()
        pool.close = slow_close

        await close_pools({"db": pool}, timeout=0.05)

        pool.terminate.assert_called_once()

    @pytest.mark.asyncio
    async def test_force_terminate_on_error(self) -> None:
        """Test pools are terminated when graceful close raises."""

        async def broken_close() -> None:
            raise RuntimeError("close failed")

        pool = MagicMock()
        pool.close = broken_close

        await close_pools({"db": pool}, timeout=1.0)

        pool.terminate.assert_called_once()

    @pytest.mark.asyncio
    async def test_closes_all_pools(self) -> None:
        """Test every pool in the dictionary is closed."""
        pools = {}
        for name in ("db1", "db2", "db3"):
            pool = MagicMock()
            pool.close = AsyncMock()
            pools[name] = pool

        await close_pools(pools, timeout=1.0)

        for pool in pools.values():
            pool.close.assert_awaited_once()
