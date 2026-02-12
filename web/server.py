"""
FastAPI web server – provides a dashboard for monitoring and controlling
the automation loop.

Endpoints:
    GET  /                  – Dashboard page
    GET  /api/state         – Current game state JSON
    GET  /api/image         – Latest annotated frame (base64 JPEG)
    POST /api/start         – Start automation
    POST /api/stop          – Stop automation
    POST /api/pause         – Pause automation
    POST /api/resume        – Resume automation
    POST /api/click         – Inject a manual click  {x, y}
    POST /api/scan          – Trigger a full scan
    POST /api/params        – Update strategy parameters
    WS   /ws                – WebSocket for live state + image push
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import TYPE_CHECKING

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import config
from strategy.base import Action, ActionType

if TYPE_CHECKING:
    from automation.loop import AutomationLoop

log = logging.getLogger(__name__)

app = FastAPI(title="TowerControl", version="0.1.0")

# Mount static files
app.mount("/static", StaticFiles(directory=str(config.STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(config.TEMPLATES_DIR))

# Will be set by run.py before starting uvicorn
_loop: "AutomationLoop | None" = None


def set_automation_loop(loop: "AutomationLoop"):
    global _loop
    _loop = loop


# ── Pages ───────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "title": "TowerControl",
    })


# ── REST API ────────────────────────────────────────────────────────────

@app.get("/api/state")
async def get_state():
    if _loop is None:
        return JSONResponse({"error": "not initialized"}, status_code=503)
    return _loop.get_state_dict()


@app.get("/api/image")
async def get_image():
    if _loop is None:
        return JSONResponse({"error": "not initialized"}, status_code=503)
    b64 = _loop.get_latest_image_b64()
    if b64 is None:
        return JSONResponse({"image": None})
    return JSONResponse({"image": b64})


@app.post("/api/start")
async def start():
    if _loop is None:
        return JSONResponse({"error": "not initialized"}, status_code=503)
    _loop.start()
    return {"status": "started"}


@app.post("/api/stop")
async def stop():
    if _loop is None:
        return JSONResponse({"error": "not initialized"}, status_code=503)
    _loop.stop()
    return {"status": "stopped"}


@app.post("/api/pause")
async def pause():
    if _loop is None:
        return JSONResponse({"error": "not initialized"}, status_code=503)
    _loop.pause()
    return {"status": "paused"}


@app.post("/api/resume")
async def resume():
    if _loop is None:
        return JSONResponse({"error": "not initialized"}, status_code=503)
    _loop.resume()
    return {"status": "resumed"}


@app.post("/api/click")
async def inject_click(request: Request):
    if _loop is None:
        return JSONResponse({"error": "not initialized"}, status_code=503)
    body = await request.json()
    x = int(body.get("x", 0))
    y = int(body.get("y", 0))
    _loop.inject_click(x, y, reason="web UI click")
    return {"status": "click queued", "x": x, "y": y}


@app.post("/api/scan")
async def trigger_scan():
    if _loop is None:
        return JSONResponse({"error": "not initialized"}, status_code=503)
    _loop.inject_action(Action(action_type=ActionType.FULL_SCAN,
                               reason="web UI triggered scan", priority=99))
    return {"status": "scan queued"}


@app.post("/api/params")
async def update_params(request: Request):
    if _loop is None:
        return JSONResponse({"error": "not initialized"}, status_code=503)
    body = await request.json()
    for key, value in body.items():
        _loop.strategy.set_parameter(key, value)
    return {"status": "updated", "params": _loop.strategy.parameters}


@app.get("/api/params/schema")
async def param_schema():
    if _loop is None:
        return JSONResponse({"error": "not initialized"}, status_code=503)
    return _loop.strategy.get_parameter_schema()


# ── WebSocket for live updates ──────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    log.info("WebSocket client connected")
    try:
        while True:
            if _loop is None:
                await ws.send_json({"error": "not initialized"})
                await asyncio.sleep(2)
                continue

            payload = _loop.get_state_dict()
            b64 = _loop.get_latest_image_b64()
            payload["image"] = b64
            await ws.send_json(payload)
            await asyncio.sleep(0.8)
    except WebSocketDisconnect:
        log.info("WebSocket client disconnected")
    except Exception as exc:
        log.warning("WebSocket error: %s", exc)
