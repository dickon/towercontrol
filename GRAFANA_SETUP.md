# Grafana + PostgreSQL Setup Instructions

## 1. Install PostgreSQL 16+

Download from https://www.postgresql.org/download/windows/ and run the installer.
Keep default port (5432). Set a password for the `postgres` superuser.

After install, open **pgAdmin** or the **SQL Shell (psql)** and run:

```sql
CREATE ROLE towercontrol WITH LOGIN PASSWORD 'towercontrol';
CREATE DATABASE towercontrol OWNER towercontrol;
```

## 2. Install TimescaleDB (optional but recommended)

Download the Windows TimescaleDB extension from https://docs.timescale.com/self-hosted/latest/install/installation-windows/

Follow the install guide — it patches the PostgreSQL binary via an MSI installer.
After install, restart the PostgreSQL service, then connect as superuser and run:

```sql
\c towercontrol
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
```

If TimescaleDB is unavailable, the migration still works — the `create_hypertable` calls
will fail but the plain tables and indexes remain functional.

## 3. Apply the database schema

Connect as the `towercontrol` user and run the migration:

```
psql -U towercontrol -d towercontrol -f db/migrations/001_initial.sql
```

Or paste the contents of `db/migrations/001_initial.sql` into pgAdmin.

## 4. Set the connection string (optional)

The default connection string is:  
`postgresql://towercontrol:towercontrol@localhost/towercontrol`

Override by setting the environment variable before running:

```powershell
$env:DATABASE_URL = "postgresql://towercontrol:yourpassword@localhost/towercontrol"
python run.py
```

## 5. Install Grafana OSS

Download from https://grafana.com/grafana/download?platform=windows and run the installer.
Default install dir: `C:\Program Files\GrafanaLabs\grafana\`

## 6. Configure Grafana for iframe embedding

Edit `C:\Program Files\GrafanaLabs\grafana\conf\custom.ini`
(create it if it doesn't exist — it overrides defaults):

```ini
[security]
allow_embedding = true
cookie_samesite = none
cookie_secure = false

[auth.anonymous]
enabled = true
org_name = Main Org.
org_role = Viewer
```

Restart the Grafana service:

```powershell
Restart-Service Grafana
```

## 7. Provision data source and dashboard

Copy (or symlink) the provisioning files to Grafana's provisioning directory:

```powershell
$g = "C:\Program Files\GrafanaLabs\grafana\conf\provisioning"
Copy-Item grafana\provisioning\datasources\*.yml "$g\datasources\" -Force
Copy-Item grafana\provisioning\dashboards\*.yml "$g\dashboards\" -Force
# The dashboard JSON must be in the path referenced by dashboards/default.yml ("dashboards")
Copy-Item grafana\dashboards\tower_control.json "C:\Program Files\GrafanaLabs\grafana\conf\provisioning\dashboards\" -Force
```

Then update `grafana\provisioning\dashboards\default.yml`'s `path:` to the full absolute path
of the folder where you placed `tower_control.json`, or restart Grafana so it re-reads provisioning.

Restart the service:

```powershell
Restart-Service Grafana
```

Open http://localhost:3000 — the "Tower Control" dashboard should appear under Dashboards.

## 8. Install the Python dependency

```powershell
pip install psycopg2-binary
```

Or with the venv:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 9. Verify

1. Start the bot: `python run.py`
2. After a few waves, query the DB:
   ```sql
   SELECT COUNT(*) FROM wave_events;
   SELECT COUNT(*) FROM resource_rates;
   ```
3. Open http://localhost:3000 — the Tower Control dashboard should show live data
4. Open the web UI at http://localhost:7700 — the Grafana iframe should appear in the Timeline card

## Notes

- The Grafana iframe URL is hardcoded in `web/templates/index.html` as:
  `http://localhost:3000/d/towercontrol/tower-control?orgId=1&kiosk&from=now-1h&to=now&refresh=5s`
  Adjust if Grafana runs on a different port or the dashboard UID changes.

- The overview minimap (below the Grafana iframe) still drives video seek navigation.

- DB writes are fire-and-forget — if PostgreSQL is down, errors are logged and the bot keeps running.
