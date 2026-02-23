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
import io
import json
import logging
import queue
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

    state["strategy_params"] = {
        "input_enabled": c.input_enabled,
        "loop_tick":     c.config.loop_tick,
    }

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
        "loop_tick": {
            "label": "Loop interval (s)", "type": "number", "min": 1, "max": 60,
        },
        "input_enabled": {
            "label": "Input enabled", "type": "bool",
        },
    }


@fapp.post("/api/params")
async def api_params(request: Request):
    body  = await request.json()
    c     = _ctx()
    if c is None:
        return {"ok": False}
    if "loop_tick" in body:
        try:
            c.config = _mod.replace(c.config, loop_tick=float(body["loop_tick"]))
        except Exception:
            pass
    if "input_enabled" in body:
        c.input_enabled = bool(body["input_enabled"])
    return {"ok": True}


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
