-- TowerControl timeseries schema
-- Requires: PostgreSQL 16+ with TimescaleDB extension
-- Fallback: works with plain PostgreSQL (skip CREATE EXTENSION and create_hypertable calls)

CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Wave transition events
CREATE TABLE IF NOT EXISTS wave_events (
    time  TIMESTAMPTZ NOT NULL,
    wave  INTEGER     NOT NULL
);
SELECT create_hypertable('wave_events', 'time', if_not_exists => TRUE);

-- Cash/coin per-minute resource rates
CREATE TABLE IF NOT EXISTS resource_rates (
    time     TIMESTAMPTZ      NOT NULL,
    cash_pm  DOUBLE PRECISION,
    coin_pm  DOUBLE PRECISION
);
SELECT create_hypertable('resource_rates', 'time', if_not_exists => TRUE);

-- Upgrade spend rate (cash spent per minute, 30-min rolling window)
CREATE TABLE IF NOT EXISTS spend_rates (
    time      TIMESTAMPTZ      NOT NULL,
    spend_pm  DOUBLE PRECISION NOT NULL
);
SELECT create_hypertable('spend_rates', 'time', if_not_exists => TRUE);

-- Wave progression rate (waves per hour, 10-min rolling window)
CREATE TABLE IF NOT EXISTS wave_rates (
    time      TIMESTAMPTZ      NOT NULL,
    waves_ph  DOUBLE PRECISION NOT NULL
);
SELECT create_hypertable('wave_rates', 'time', if_not_exists => TRUE);

-- Bot actions (clicks, scrolls) with fractional coordinates
CREATE TABLE IF NOT EXISTS actions (
    time    TIMESTAMPTZ      NOT NULL,
    fx      DOUBLE PRECISION NOT NULL,
    fy      DOUBLE PRECISION NOT NULL,
    reason  TEXT             NOT NULL DEFAULT ''
);
SELECT create_hypertable('actions', 'time', if_not_exists => TRUE);

-- Optional: 30-day retention policies (uncomment if desired)
-- SELECT add_retention_policy('wave_events',    INTERVAL '30 days');
-- SELECT add_retention_policy('resource_rates', INTERVAL '30 days');
-- SELECT add_retention_policy('spend_rates',    INTERVAL '30 days');
-- SELECT add_retention_policy('wave_rates',     INTERVAL '30 days');
-- SELECT add_retention_policy('actions',        INTERVAL '30 days');
