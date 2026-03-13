"""PostgreSQL connection pool for TowerControl timeseries storage.

Set the DATABASE_URL environment variable to override the default connection string:
    postgresql://towercontrol:towercontrol@localhost/towercontrol

The pool is initialised lazily on first use (or explicitly via ``init_pool()``).
Call ``close_pool()`` on shutdown to release connections cleanly.
"""

import logging
import os
from contextlib import contextmanager
from typing import Optional

import psycopg2
from psycopg2 import pool as _pool

log = logging.getLogger(__name__)

_DEFAULT_DSN = "postgresql://towercontrol:towercontrol@localhost/towercontrol"

_pool_inst: Optional[_pool.ThreadedConnectionPool] = None


def init_pool(dsn: Optional[str] = None, minconn: int = 1, maxconn: int = 5) -> None:
    """Create the shared connection pool.  Safe to call more than once (no-op if already open)."""
    global _pool_inst
    if _pool_inst is not None:
        return
    dsn = dsn or os.environ.get("DATABASE_URL", _DEFAULT_DSN)
    try:
        _pool_inst = _pool.ThreadedConnectionPool(minconn, maxconn, dsn)
        log.info("DB pool opened (%d–%d connections)", minconn, maxconn)
    except Exception:
        log.exception("Failed to create DB connection pool — timeseries writes will be skipped")


def close_pool() -> None:
    """Release all connections."""
    global _pool_inst
    if _pool_inst is not None:
        try:
            _pool_inst.closeall()
        except Exception:
            log.exception("Error closing DB pool")
        _pool_inst = None
        log.info("DB pool closed")


@contextmanager
def get_conn():
    """Yield a connection from the pool, returning it on exit.

    Usage::

        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(...)
            conn.commit()

    If the pool is not initialised the context manager yields ``None``.
    """
    if _pool_inst is None:
        yield None
        return
    conn = _pool_inst.getconn()
    try:
        yield conn
    finally:
        _pool_inst.putconn(conn)
