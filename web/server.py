"""
TowerControl web server.

Endpoints:
  GET  /                   — main dashboard
  GET  /captures           — capture file browser
  GET  /debug/<path>       — serve a file from the debug dir
  WS   /ws                 — push state + JPEG image every second
  GET  /log/stream         — SSE stream of INFO+ log lines
  GET  /api/captures       — JSON list of archived capture sets
  POST /api/start|stop|pause|resume|scan  — bot control
  POST /api/click          — inject a fractional click  {fx, fy}  or pixel {x, y}
  GET  /api/params/schema  — strategy param schema
  POST /api/params         — update strategy params

Usage (from run.py):
    from web.server import start_server
    start_server(mod, mod.ctx.config.debug_dir, host, port)
"""

import asyncio
import base64
import datetime
import io
import json
import logging
import queue
import re
import threading
import time as _time
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

_HERE = Path(__file__).parent
_TEMPLATE_DIR = _HERE / "templates"
_STATIC_DIR   = _HERE / "static"

_templates = Jinja2Templates(directory=str(_TEMPLATE_DIR))

fapp = FastAPI(docs_url=None, redoc_url=None)
fapp.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

# ---------------------------------------------------------------------------
# Shared state (set by start_server)
# ---------------------------------------------------------------------------
_mod        = None          # reference to towercontrol_functional module
_debug_dir: Optional[Path] = None

# ---------------------------------------------------------------------------
# SSE log queue  (thread-safe; asyncio consumer polls it)
# ---------------------------------------------------------------------------
_sse_queue: queue.Queue = queue.Queue(maxsize=4000)

class _SSELogHandler(logging.Handler):
    """Puts formatted records into _sse_queue for the /log/stream endpoint."""
    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = self.format(record)
            try:
                _sse_queue.put_nowait(line)
            except queue.Full:
                try:
                    _sse_queue.get_nowait()
                except queue.Empty:
                    pass
                _sse_queue.put_nowait(line)
        except Exception:
            pass

_sse_handler = _SSELogHandler()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ctx():
    """Return the live RuntimeContext, or None if not yet available."""
    if _mod is None:
        return None
    return getattr(_mod, "ctx", None)


def _build_state() -> dict:
    c = _ctx()
    if c is None:
        return {"bot_status": "offline"}

    state: dict = {"bot_status": c.status}

    # JPEG-compressed screenshot
    img = c.latest_image
    if img is not None:
        try:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=75)
            state["image"] = base64.b64encode(buf.getvalue()).decode()
        except Exception:
            pass

    # Window geometry
    wr = c.window_rect
    if wr:
        state["window_rect"] = {
            "left": wr.left, "top": wr.top,
            "width": wr.width, "height": wr.height,
        }

    gs = c.game_state
    state["wave"]          = gs.wave or "—"
    state["error_count"]   = gs.error_count
    state["resources"]     = dict(gs.resources)
    state["tier"]          = gs.tier

    sc = gs.current_screen
    state["current_screen"] = sc.screen.name if sc else "UNKNOWN"

    frame = c.frame
    if frame:
        w_img, h_img = frame.image_size
        state["element_count"] = len(frame.results)
        state["ocr_results"] = [
            {
                "text": r.text,
                "x0":   round(r.bbox[0] / w_img, 4) if w_img else 0,
                "y0":   round(r.bbox[1] / h_img, 4) if h_img else 0,
                "w":    round(r.bbox[2] / w_img, 4) if w_img else 0,
                "h":    round(r.bbox[3] / h_img, 4) if h_img else 0,
                "conf": round(r.confidence, 1),
            }
            for r in frame.results
        ]

    state["recent_actions"] = [
        {
            "type":   a.get("type", "?"),
            "reason": a.get("reason", ""),
            "time":   a.get("time", 0),
            "x":      a.get("x"),
            "y":      a.get("y"),
            "ax":     a.get("ax"),
            "ay":     a.get("ay"),
            "fx":     a.get("fx"),
            "fy":     a.get("fy"),
        }
        for a in gs.action_history[-20:]
    ]

    state["perk_selection_history"] = [
        {"timestamp": p.get("timestamp", 0), "wave": p.get("wave", "?"),
         "selected": p.get("selected", "?"), "text": p.get("text", "")}
        for p in gs.perk_selection_history[-50:]
    ]

    state["upgrade_purchase_history"] = [
        {"timestamp": p.get("timestamp", 0), "wave": p.get("wave"),
         "upgrade_name": p.get("upgrade_name", "?"), "cost": p.get("cost"),
         "current_value": p.get("current_value")}
        for p in gs.upgrade_purchase_history[-200:]
    ]

    state["upgrade_advance_history"] = [
        {"timestamp": p.get("timestamp", 0), "wave": p.get("wave"),
         "from_upgrade": p.get("from_upgrade", "?"), "to_upgrade": p.get("to_upgrade", "?"),
         "reason": p.get("reason", "?"),
         "from_button_img": p.get("from_button_img")}
        for p in gs.upgrade_advance_history[-50:]
    ]

    try:
        now = _time.time()
        state["upgrade_seen"] = {
            label: {
                "seen_ago": round(now - d["timestamp"], 0),
                "current_value": d.get("current_value"),
                "cost": d.get("cost"),
                "is_max": d.get("is_max", False),
                "is_affordable": d.get("is_affordable"),
                "upgrades_to_purchase": d.get("upgrades_to_purchase"),
                "cell_color_name": d.get("cell_color_name", ""),
                "crop_b64": d.get("crop_b64"),
            }
            for label, d in c.upgrade_seen.items()
        }
    except Exception:
        pass

    try:
        prio = _mod._active_upgrade_priority()
        state["upgrade_priority"] = [
            {"page": p[0], "label": p[1], "cost_threshold": p[2], "needs_scroll": p[3]}
            for p in prio
        ]
        state["upgrade_state"] = c.upgrade_state
    except Exception:
        pass

    state["strategy_params"] = {
        "input_enabled": c.input_enabled,
        "cloud_grab_enabled": c.config.cloud_grab_enabled,
        "loop_tick": c.config.loop_tick,
    }

    # Watchdog state
    try:
        now = _time.time()
        state["watchdog"] = {
            "enabled":               c.config.watchdog_enabled,
            "game_launch_enabled":   c.config.game_launch_enabled,
            "last_bs_restart_ago":   round(now - c.last_bs_restart, 0) if c.last_bs_restart else None,
            "last_game_launch_ago":  round(now - c.last_game_launch, 0) if c.last_game_launch else None,
            "last_game_ui_seen_ago": round(now - c.last_game_ui_seen, 0) if c.last_game_ui_seen else None,
            "last_wave_advance_ago": round(now - c.last_wave_advance, 0) if c.last_wave_advance else None,
            "wave_stall_timeout":    c.config.wave_stall_timeout,
            "hard_restart_running":  c.hard_restart_running,
        }
    except Exception:
        pass

    # Full context dump
    try:
        now = _time.time()
        bst = gs.battle_start_time
        state["ctx_full"] = {
            "status":                  c.status,
            "running":                 c.running,
            "input_enabled":           c.input_enabled,
            "ocr_time_s":              round(c.ocr_time, 3),
            "upgrade_state":           c.upgrade_state,
            "upgrade_mode_seen":       c.upgrade_mode_seen or "—",
            "upgrade_scroll_dir":      c.upgrade_scroll_direction,
            "upgrade_scroll_start":    round(now - c.upgrade_scroll_start, 1) if c.upgrade_scroll_start else "—",
            "recover_stage":           c.recover_stage,
            "no_perk_until":           round(c.no_perk_until - now, 1) if c.no_perk_until > now else "—",
            "upgrades_finished":       c.upgrades_finished_time or "—",
            "last_upgrade_action_ago": round(now - c.last_upgrade_action, 1) if c.last_upgrade_action else "—",
            "last_seen_upgrades_ago":  round(now - c.last_seen_upgrades, 1) if c.last_seen_upgrades else "—",
            "tier":                    gs.tier,
            "wave":                    gs.wave or "—",
            "screen":                  gs.current_screen.screen.name if gs.current_screen else "UNKNOWN",
            "error_count":             gs.error_count,
            "battle_start_ago":        round(now - bst, 0) if bst else "—",
            "wave_history_len":        len(gs.wave_history),
            "action_history_len":      len(gs.action_history),
            "window":                  (f"{c.window_rect.width}×{c.window_rect.height}"
                                        f" @({c.window_rect.left},{c.window_rect.top})")
                                       if c.window_rect else "—",
            "ocr_engine":              c.config.ocr_engine,
            "loop_tick":               c.config.loop_tick,
        }
    except Exception as exc:
        logging.getLogger(__name__).warning("ctx_full build error: %s", exc)

    # Lightweight sequence value: increments whenever timeline data changes.
    # The frontend watches this via WebSocket and refreshes the chart automatically.
    try:
        _tl_dir_mt = int((_debug_dir.stat().st_mtime if _debug_dir and _debug_dir.exists() else 0) * 2)
        state["timeline_seq"] = len(gs.wave_history) + len(gs.action_history) + _tl_dir_mt
    except Exception:
        pass

    return state


def _list_capture_sets() -> list:
    """Return up to 200 most-recent archived capture sets from debug_dir."""
    if _debug_dir is None:
        return []
    # Plain (non-annotated) archived captures
    plain = sorted(
        (f for f in _debug_dir.glob("capture_*.png")
         if not f.stem.startswith("capture_annotated_")),
        reverse=True,
    )[:200]
    result = []
    for png in plain:
        stem = png.stem                                        # capture_20260223_...
        anno_stem = "capture_annotated_" + stem[len("capture_"):]
        txt_path  = _debug_dir / f"{stem}.txt"
        anno_path = _debug_dir / f"{anno_stem}.png"
        result.append({
            "name":           png.name,
            "stem":           stem,
            "txt":            txt_path.name  if txt_path.exists()  else None,
            "annotated":      anno_path.name if anno_path.exists() else None,
        })
    # Always include the "live" pair at the top if they exist
    live = []
    for base_stem, label in [("capture", "LIVE"), ("capture_annotated", "LIVE annotated")]:
        p = _debug_dir / f"{base_stem}.png"
        if p.exists():
            t = _debug_dir / f"{base_stem}.txt"
            live.append({
                "name":      p.name,
                "stem":      base_stem,
                "label":     label,
                "txt":       t.name if t.exists() else None,
                "annotated": None,
            })
    return live + result


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@fapp.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return _templates.TemplateResponse("index.html", {"request": request})


@fapp.get("/captures", response_class=HTMLResponse)
async def captures_page(request: Request):
    return _templates.TemplateResponse("captures.html", {"request": request})


@fapp.get("/debug/{filename:path}")
async def debug_file(filename: str):
    if _debug_dir is None:
        return Response("Server not initialised", status_code=503)
    path = (_debug_dir / filename).resolve()
    try:
        path.relative_to(_debug_dir.resolve())
    except ValueError:
        return Response("Forbidden", status_code=403)
    if not path.exists():
        return Response("Not found", status_code=404)
    return FileResponse(str(path))


# ── WebSocket ────────────────────────────────────────────────────────────────

@fapp.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            try:
                state = _build_state()
                await websocket.send_text(json.dumps(state))
            except WebSocketDisconnect:
                raise
            except Exception as exc:
                logging.getLogger(__name__).warning("WS send error: %s", exc)
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        pass


# ── SSE log stream ───────────────────────────────────────────────────────────

@fapp.get("/log/stream")
async def log_stream():
    async def generate():
        # Drain anything already queued (backlog)
        backlog: list[str] = []
        try:
            while True:
                backlog.append(_sse_queue.get_nowait())
        except queue.Empty:
            pass
        for line in backlog[-500:]:
            yield f"data: {json.dumps(line)}\n\n"

        # Ongoing stream
        while True:
            lines: list[str] = []
            try:
                while True:
                    lines.append(_sse_queue.get_nowait())
            except queue.Empty:
                pass
            for line in lines:
                yield f"data: {json.dumps(line)}\n\n"
            if not lines:
                yield ": keepalive\n\n"
            await asyncio.sleep(0.4)

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


# ── REST API ─────────────────────────────────────────────────────────────────

@fapp.get("/api/captures")
async def api_captures():
    return _list_capture_sets()


@fapp.post("/api/start")
async def api_start():
    c = _ctx()
    if c:
        c.running = True
        c.status  = "running"
    return {"ok": True}


@fapp.post("/api/stop")
async def api_stop():
    c = _ctx()
    if c:
        c.running = False
        c.status  = "stopped"
    return {"ok": True}


@fapp.post("/api/pause")
async def api_pause():
    c = _ctx()
    if c:
        c.input_enabled = False
        c.status        = "paused"
    return {"ok": True}


@fapp.post("/api/resume")
async def api_resume():
    c = _ctx()
    if c:
        c.input_enabled = True
        c.status        = "running"
    return {"ok": True}


@fapp.post("/api/scan")
async def api_scan():
    # Trigger a fresh OCR tick on next loop iteration — nothing special needed
    return {"ok": True}


@fapp.post("/api/restart")
async def api_restart():
    """Reset all mutable game state (wave, tier, upgrade progress, etc.)
    without stopping the automation loop."""
    import time as _time
    c = _ctx()
    if c is None:
        return {"ok": False, "reason": "no ctx"}
    # Reset frozen GameState
    c.game_state           = _mod.GameState()
    # Reset mutable loop state
    c.upgrade_state        = 0
    c.recover_stage        = 0
    c.upgrade_scroll_start = 0.0
    c.upgrade_scroll_direction = "down"
    c.no_perk_until        = 0.0
    c.upgrades_finished_time = None
    c.last_upgrade_action  = _time.time()
    c.last_seen_upgrades   = _time.time()
    c.upgrade_mode_seen    = None
    c.status               = "running"
    logging.getLogger(__name__).info("Context reset via web UI")
    return {"ok": True}


@fapp.post("/api/watchdog")
async def api_watchdog(request: Request):
    """Enable or disable watchdog features.  Body: {watchdog?: bool, game_launch?: bool}"""
    from dataclasses import replace as _replace
    c = _ctx()
    if c is None:
        return {"ok": False, "reason": "no ctx"}
    try:
        body = await request.json()
    except Exception:
        body = {}
    cfg = c.config
    if "watchdog" in body:
        cfg = _replace(cfg, watchdog_enabled=bool(body["watchdog"]))
    if "game_launch" in body:
        cfg = _replace(cfg, game_launch_enabled=bool(body["game_launch"]))
    c.config = cfg
    return {"ok": True,
            "watchdog_enabled": cfg.watchdog_enabled,
            "game_launch_enabled": cfg.game_launch_enabled}


@fapp.post("/api/hard_restart")
async def api_hard_restart():
    """Kill BlueStacks, restart it, press Home after 60 s, then launch The Tower.

    Identical to the wave-stall watchdog recovery sequence.
    """
    c = _ctx()
    if c is None:
        return {"ok": False, "reason": "no ctx"}
    if c.hard_restart_running:
        return {"ok": False, "reason": "hard restart already in progress"}
    logging.getLogger(__name__).info("Hard restart triggered via web UI")
    _mod.do_hard_restart()
    return {"ok": True}


@fapp.post("/api/click")
async def api_click(request: Request):
    body = await request.json()
    c = _ctx()
    if c is None or c.window_rect is None:
        return {"ok": False, "reason": "no window"}
    wr = c.window_rect
    # Accept either fractional {fx, fy} or pixel {x, y}
    if "fx" in body:
        fx, fy = float(body["fx"]), float(body["fy"])
    else:
        fx = float(body["x"]) / wr.width
        fy = float(body["y"]) / wr.height
    try:
        _mod.do_click("web-injected click", fx, fy)
        return {"ok": True, "fx": round(fx, 4), "fy": round(fy, 4)}
    except Exception as exc:
        return {"ok": False, "reason": str(exc)}


@fapp.get("/api/params/schema")
async def api_params_schema():
    return {
        "input_enabled": {
            "label": "Input enabled", "type": "bool",
        },
        "cloud_grab_enabled": {
            "label": "Cloud Grab", "type": "bool",
        },
        "loop_tick": {
            "label": "Loop tick (s)", "type": "float", "min": 1.0, "max": 120.0, "step": 0.5,
        },
    }


@fapp.post("/api/upgrade_state")
async def api_upgrade_state(request: Request):
    body = await request.json()
    c = _ctx()
    if c is None:
        return {"ok": False, "reason": "no ctx"}
    try:
        idx = int(body["index"])
        prio = _mod._active_upgrade_priority()
        if 0 <= idx <= len(prio):
            c.upgrade_state = idx
            logging.getLogger(__name__).info("Upgrade state set to %d via web UI", idx)
            return {"ok": True, "index": idx}
        return {"ok": False, "reason": "index out of range"}
    except Exception as exc:
        return {"ok": False, "reason": str(exc)}


@fapp.post("/api/params")
async def api_params(request: Request):
    body  = await request.json()
    c     = _ctx()
    if c is None:
        return {"ok": False}
    if "input_enabled" in body:
        c.input_enabled = bool(body["input_enabled"])
    if "cloud_grab_enabled" in body:
        c.config = _mod.replace(c.config, cloud_grab_enabled=bool(body["cloud_grab_enabled"]))
    if "loop_tick" in body:
        val = float(body["loop_tick"])
        if 1.0 <= val <= 120.0:
            c.config = _mod.replace(c.config, loop_tick=val)
            logging.getLogger(__name__).info("loop_tick set to %.1f via web UI", val)
    return {"ok": True}


# ── Timeline ─────────────────────────────────────────────────────────────────

_CAPTURE_TICK_RE = re.compile(
    r"capture_"
    r"(\d{8})_(\d{6})"          # date YYYYMMDD, time HHMMSS
    r"(?:_\d+)?"                 # optional microseconds field
    r"(?:_t(\d+))?"              # optional tier  _t6
    r"(?:_w(\d+))?"              # optional wave  _w3077
    r"(?:_(ATTACK|DEFENSE|UTILITY))?"  # optional tab
    r"\.txt$"
)

_CAPTURE_PNG_RE = re.compile(
    r"capture_"
    r"(\d{8})_(\d{6})"          # date YYYYMMDD, time HHMMSS
    r"(?:_\d+)?"                 # optional microseconds field
    r"(?:_t(\d+))?"              # optional tier
    r"(?:_w(\d+))?"              # optional wave
    r"(?:_(ATTACK|DEFENSE|UTILITY))?"  # optional tab
    r"\.png$"
)


def _build_png_index(debug_dir: Path) -> tuple[dict, dict]:
    """Return (plain_index, annotated_index) mapping (date,time,tier,wave,tab) -> filename."""
    plain: dict = {}
    annotated: dict = {}
    for f in debug_dir.glob("capture_*.png"):
        if f.stem.startswith("capture_annotated_"):
            # Strip "capture_annotated_" prefix; re-prepend "capture_" for the regex
            stripped = "capture_" + f.name[len("capture_annotated_"):]
            m = _CAPTURE_PNG_RE.match(stripped)
            target = annotated
        else:
            m = _CAPTURE_PNG_RE.match(f.name)
            target = plain
        if m:
            key = (m.group(1), m.group(2), m.group(3), m.group(4), m.group(5))
            target[key] = f.name
    return plain, annotated


@fapp.get("/api/timeline")
async def api_timeline():
    """Return wave history, rate history, and debug capture ticks for the timeline chart."""
    import time as _time
    c = _ctx()
    _two_days_ago = _time.time() - 2 * 86400

    # ── wave history (bot-recorded)
    wave_history = []
    if c is not None:
        for wave_num, ts in (c.game_state.wave_history if c.game_state else []):
            wave_history.append({"t": ts, "wave": wave_num})

    # ── wave progression rate (waves/hour) — rolling 10-minute window
    wave_rate_history: list = []
    _RATE_WINDOW = 600  # seconds; look back up to 10 minutes to compute local rate
    if len(wave_history) >= 2:
        for i, pt in enumerate(wave_history):
            t_now    = pt["t"]
            w_now    = pt["wave"]
            cutoff_t = t_now - _RATE_WINDOW
            # Find the furthest-back point within the window
            anchor = None
            for j in range(i - 1, -1, -1):
                if wave_history[j]["t"] >= cutoff_t:
                    anchor = wave_history[j]
                else:
                    break
            if anchor is None and i > 0:
                anchor = wave_history[0]
            if anchor is not None:
                dt = t_now - anchor["t"]
                dw = w_now - anchor["wave"]
                if dt > 0 and dw >= 0:
                    wave_rate_history.append({"t": t_now, "waves_ph": round(dw / dt * 3600, 2)})

    # ── cash/coin rate history
    rate_history = list(getattr(c, "rate_history", [])) if c is not None else []

    # ── capture ticks: parse debug dir filenames (last 2 days only)
    ticks = []
    if _debug_dir is not None:
        plain_idx, anno_idx = _build_png_index(_debug_dir)
        for f in _debug_dir.glob("capture_*.txt"):
            if f.stem.startswith("capture_annotated_"):
                continue
            m = _CAPTURE_TICK_RE.match(f.name)
            if not m:
                continue
            try:
                dt = datetime.datetime.strptime(
                    m.group(1) + m.group(2), "%Y%m%d%H%M%S"
                )
                # Treat as local time; convert to UTC epoch
                ts = dt.timestamp()
            except ValueError:
                continue
            if ts < _two_days_ago:
                continue
            key = (m.group(1), m.group(2), m.group(3), m.group(4), m.group(5))
            ticks.append({
                "t":         ts,
                "wave":      int(m.group(4)) if m.group(4) else None,
                "tier":      int(m.group(3)) if m.group(3) else None,
                "tab":       m.group(5),
                "file":      f.name,
                "png":       plain_idx.get(key),
                "png_anno":  anno_idx.get(key),
            })
        ticks.sort(key=lambda x: x["t"])

    # ── action history: clicks with fractional coords
    action_history = []
    if c is not None and c.game_state:
        for a in c.game_state.action_history:
            fx = a.get("fx")
            fy = a.get("fy")
            t  = a.get("time")
            if fx is None or fy is None or t is None:
                continue
            action_history.append({
                "t":      t,
                "fx":     fx,
                "fy":     fy,
                "reason": a.get("reason", ""),
            })

    return {
        "wave_history":      wave_history,
        "wave_rate_history": wave_rate_history,
        "rate_history":      rate_history,
        "capture_ticks":     ticks,
        "action_history":    action_history,
    }


# ---------------------------------------------------------------------------
# Server bootstrap
# ---------------------------------------------------------------------------

def start_server(mod, debug_dir: Path, host: str = "127.0.0.1", port: int = 7700) -> threading.Thread:
    """
    Start the uvicorn server in a background daemon thread.

    Parameters
    ----------
    mod        : the towercontrol_functional module (for live ctx access)
    debug_dir  : Path to the debug/ directory
    host, port : bind address
    """
    global _mod, _debug_dir
    _mod       = mod
    _debug_dir = debug_dir

    # Register SSE log handler on the root logger
    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
    _sse_handler.setLevel(logging.INFO)
    _sse_handler.setFormatter(fmt)
    logging.getLogger().addHandler(_sse_handler)

    config = uvicorn.Config(
        fapp,
        host=host,
        port=port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)

    t = threading.Thread(target=server.run, daemon=True, name="web-server")
    t.start()
    logging.getLogger(__name__).info("Web server started at http://%s:%d", host, port)
    return t
