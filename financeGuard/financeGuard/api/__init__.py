from __future__ import annotations

import logging

from financeGuard import app, db
from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("FinanceGuard")


_engine = None
_async_session_factory: async_sessionmaker[AsyncSession] | None = None


def _ensure_transactions_deposit_columns(sync_conn) -> None:
    inspector = inspect(sync_conn)
    if "transactions" not in inspector.get_table_names():
        return

    existing_columns = {column["name"] for column in inspector.get_columns("transactions")}
    missing_columns = {
        "deposit_channel": "ALTER TABLE transactions ADD COLUMN deposit_channel VARCHAR(32)",
        "deposit_details": "ALTER TABLE transactions ADD COLUMN deposit_details TEXT",
        "deposit_updated_at": "ALTER TABLE transactions ADD COLUMN deposit_updated_at DATETIME",
    }
    for column_name, statement in missing_columns.items():
        if column_name not in existing_columns:
            sync_conn.execute(text(statement))

try:
    _engine = create_async_engine(
        app.config.get("ASYNC_DATABASE_URI") or app.config["SQLALCHEMY_DATABASE_URI"],
        poolclass=NullPool,  # avoid cross-event-loop connection reuse in Flask async
        echo=False,  # set True to log generated SQL
    )
    _async_session_factory = async_sessionmaker(
        _engine,
        class_=AsyncSession,
        expire_on_commit=False,  # avoids detached instance issues
    )
except ModuleNotFoundError:
    log.error(
        "Async database driver missing. Install `aiomysql` to use MySQL async.",
        exc_info=True,
    )


def AsyncSessionFactory() -> AsyncSession:
    """
    Compatibility wrapper for code that expects a callable session factory.
    Returns an `AsyncSession`, or raises a clear error if async DB deps are missing.
    """
    if _async_session_factory is None:
        raise RuntimeError("Async database unavailable. Install `aiomysql`.")
    return _async_session_factory()


async def init_db() -> None:
    if _engine is None:
        raise RuntimeError("Async database unavailable. Install `aiomysql`.")
    async with _engine.begin() as conn:
        await conn.run_sync(db.Model.metadata.create_all)
        await conn.run_sync(_ensure_transactions_deposit_columns)
    log.info("Database tables ready")
