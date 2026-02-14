"""
Main automation loop – ties capture, OCR, state, scanner, strategy,
and input together into a continuous cycle.

Runs in its own thread so the web server stays responsive.
"""

from __future__ import annotations

import base64
import datetime
import io
import logging
import threading
import time
from typing import Any, Dict, Optional

import config as cfg

import cv2
import numpy as np
from PIL import Image

from capture.window import WindowCapture
from input.controller import InputController
from ocr.engine import OCREngine
from scanner.ui_scanner import UIScanner
from state.machine import GameState, StateMachine
from strategy.base import Action, ActionType, Strategy

log = logging.getLogger(__name__)


class AutomationLoop:
    """Continuous game-playing loop."""

    def __init__(self, capture: WindowCapture, ocr: OCREngine,
                 controller: InputController, state_machine: StateMachine,
                 scanner: UIScanner, strategy: Strategy,
                 tick: float = 1.0):
        self.capture = capture
        self.ocr = ocr
        self.controller = controller
        self.sm = state_machine
        self.scanner = scanner
        self.strategy = strategy
        self.tick = tick

        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._paused = False
        self._lock = threading.Lock()

        # Latest frame data for the web dashboard
        self._latest_image: Optional[Image.Image] = None
        self._latest_annotated: Optional[np.ndarray] = None
        self._override_action: Optional[Action] = None
        self._status: str = "stopped"
        self.save_screenshots: bool = False   # disabled by default

    # ── Thread control ──────────────────────────────────────────────────

    def start(self):
        if self._running:
            return
        self._running = True
        self._paused = False
        self._status = "running"
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="AutomationLoop")
        self._thread.start()
        log.info("Automation loop started")

    def stop(self):
        self._running = False
        self._status = "stopped"
        if self._thread:
            self._thread.join(timeout=5)
        log.info("Automation loop stopped")

    def pause(self):
        self._paused = True
        self._status = "paused"

    def resume(self):
        self._paused = False
        self._status = "running"

    @property
    def running(self) -> bool:
        return self._running

    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def status(self) -> str:
        return self._status

    # ── Override from web UI ────────────────────────────────────────────

    def inject_action(self, action: Action):
        """Queue an action from the web UI (runs on next tick)."""
        with self._lock:
            self._override_action = action

    def inject_click(self, rel_x: int, rel_y: int, reason: str = "manual"):
        self.inject_action(Action(
            action_type=ActionType.CLICK, x=rel_x, y=rel_y,
            reason=reason, priority=99,
        ))

    # ── Image access for web dashboard ──────────────────────────────────

    def get_latest_image_b64(self) -> Optional[str]:
        """Return the latest annotated frame as a base64 JPEG."""
        img = self._latest_annotated
        if img is None:
            return None
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return base64.b64encode(buf.tobytes()).decode("ascii")

    def get_state_dict(self) -> Dict[str, Any]:
        """Return current state for the web dashboard."""
        d = self.sm.state.to_dict()
        d["bot_status"] = self._status
        d["strategy"] = self.strategy.name
        d["strategy_params"] = self.strategy.parameters
        d["window_found"] = self.capture.rect is not None
        if self.capture.rect:
            r = self.capture.rect
            d["window_rect"] = {"left": r.left, "top": r.top,
                                "width": r.width, "height": r.height}
        return d

    # ── Main loop ───────────────────────────────────────────────────────

    def _loop(self):
        while self._running:
            t0 = time.time()
            try:
                if self._paused:
                    time.sleep(0.5)
                    continue
                self._tick()
            except Exception as exc:
                log.error("Loop tick error: %s", exc, exc_info=True)
                self.sm.state.error_count += 1
                time.sleep(2)

            elapsed = time.time() - t0
            sleep_time = max(0, self.tick - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _tick(self):
        # 1. Capture
        img = self.capture.capture()
        if img is None:
            self._status = "no_window"
            time.sleep(2)
            return
        self._latest_image = img

        # 2. OCR
        try:
            frame = self.ocr.process(img)
        except Exception as exc:
            log.error("OCR failed in main loop: %s", exc)
            self.sm.state.error_count += 1
            time.sleep(1)
            return
        
        screen_state = self.sm.update(frame, self.scanner._classify_element)
        resources = self.scanner._extract_resources(frame.results)
        self.sm.state.resources.update(resources)

        # 3. Build annotated image for dashboard
        try:
            self._latest_annotated = self._annotate(img, frame)
        except Exception:
            self._latest_annotated = None

        # 3b. Save screenshot to disk
        if self.save_screenshots:
            self._save_screenshot(img, self._latest_annotated)

        # 4. Check for override action
        action: Optional[Action] = None
        with self._lock:
            if self._override_action:
                action = self._override_action
                self._override_action = None

        # 5. Strategy decides
        if action is None:
            action = self.strategy.decide(self.sm.state)

        # 6. Execute action
        if action and action.action_type != ActionType.NONE:
            self._execute(action)
            self.sm.state.record_action(action.action_type.name, {
                "reason": action.reason,
                "x": action.x, "y": action.y,
            })

    def _execute(self, action: Action):
        log.info("Action: %s – %s", action.action_type.name, action.reason)

        if action.action_type == ActionType.CLICK:
            self.controller.click(action.x, action.y)

        elif action.action_type == ActionType.SCROLL:
            self.controller.scroll(action.x, action.y, action.amount)

        elif action.action_type == ActionType.SWIPE_UP:
            self.controller.swipe_up(action.x, action.y,
                                     distance=action.amount or 300)

        elif action.action_type == ActionType.SWIPE_DOWN:
            self.controller.swipe_down(action.x, action.y,
                                       distance=action.amount or 300)

        elif action.action_type == ActionType.TAP_KEY:
            self.controller.tap_key(action.key)

        elif action.action_type == ActionType.WAIT:
            time.sleep(action.duration)

        elif action.action_type == ActionType.SCAN_CURRENT:
            self.scanner.scan_current()

        elif action.action_type == ActionType.FULL_SCAN:
            self.scanner.full_scan()

    def _annotate(self, img: Image.Image, frame) -> np.ndarray:
        """Draw OCR bounding boxes on the image for the dashboard."""
        arr = np.array(img)
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        for r in frame.results:
            x, y, w, h = r.bbox
            color = (0, 255, 0)  # green
            cv2.rectangle(arr, (x, y), (x + w, y + h), color, 1)
            cv2.putText(arr, r.text[:30], (x, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        return arr

    def _save_screenshot(self, raw: Image.Image, annotated: np.ndarray):
        """Save raw and annotated screenshots to the screenshots directory."""
        try:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            raw_path = cfg.SCREENSHOTS_DIR / f"{ts}_raw.png"
            ann_path = cfg.SCREENSHOTS_DIR / f"{ts}_ocr.jpg"
            raw.save(str(raw_path))
            cv2.imwrite(str(ann_path), annotated,
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
            log.debug("Screenshots saved: %s", ts)
        except Exception as exc:
            log.warning("Failed to save screenshot: %s", exc)
