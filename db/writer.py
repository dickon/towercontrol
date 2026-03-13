"""Write timeseries data points to PostgreSQL.

Every function is fire-and-forget: exceptions are logged but never propagated
so that the main automation loop is never disrupted by database issues.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from db.connection import get_conn

log = logging.getLogger(__name__)


def _ts_to_dt(ts: float) -> datetime:
    """Convert a Unix epoch timestamp to a timezone-aware datetime."""
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def insert_wave_event(wave: int, ts: float) -> None:
    try:
        with get_conn() as conn:
            if conn is None:
                return
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO wave_events (time, wave) VALUES (%s, %s)",
                    (_ts_to_dt(ts), wave),
                )
            conn.commit()
    except Exception:
        log.exception("Failed to insert wave_event (wave=%s)", wave)


def insert_resource_rate(ts: float, cash_pm: Optional[float], coin_pm: Optional[float]) -> None:
    try:
        with get_conn() as conn:
            if conn is None:
                return
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO resource_rates (time, cash_pm, coin_pm) VALUES (%s, %s, %s)",
                    (_ts_to_dt(ts), cash_pm, coin_pm),
                )
            conn.commit()
    except Exception:
        log.exception("Failed to insert resource_rate")


def insert_spend_rate(ts: float, spend_pm: float) -> None:
    try:
        with get_conn() as conn:
            if conn is None:
                return
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO spend_rates (time, spend_pm) VALUES (%s, %s)",
                    (_ts_to_dt(ts), spend_pm),
                )
            conn.commit()
    except Exception:
        log.exception("Failed to insert spend_rate")


def insert_wave_rate(ts: float, waves_ph: float) -> None:
    try:
        with get_conn() as conn:
            if conn is None:
                return
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO wave_rates (time, waves_ph) VALUES (%s, %s)",
                    (_ts_to_dt(ts), waves_ph),
                )
            conn.commit()
    except Exception:
        log.exception("Failed to insert wave_rate")


def insert_action(ts: float, fx: float, fy: float, reason: str) -> None:
    try:
        with get_conn() as conn:
            if conn is None:
                return
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO actions (time, fx, fy, reason) VALUES (%s, %s, %s, %s)",
                    (_ts_to_dt(ts), fx, fy, reason),
                )
            conn.commit()
    except Exception:
        log.exception("Failed to insert action")
