#!/usr/bin/env python3
"""
TowerControl – entry point.

Wires up all components and starts the web server + automation loop.

Usage:
    python run.py                    # start with defaults
    python run.py --game tower_idle  # specify game
    python run.py --port 7700        # web UI port
    python run.py --ocr easyocr      # use easyocr backend
    python run.py --no-auto          # start web UI only, don't auto-start bot
"""

from __future__ import annotations

import argparse
import logging
import sys

import config

# ── Logging ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-24s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("towercontrol")


def build_components(args):
    """Construct the full object graph."""
    from capture.window import WindowCapture
    from ocr.engine import OCREngine
    from input.controller import InputController
    from state.machine import StateMachine, Screen
    from scanner.ui_scanner import UIScanner
    from automation.loop import AutomationLoop

    # ── Strategy ────────────────────────────────────────────────────────
    if args.game == "tower_idle":
        from strategy.tower_idle import TowerIdleStrategy
        strategy = TowerIdleStrategy()
    else:
        from strategy.base import Strategy, Action, ActionType
        class NullStrategy(Strategy):
            name = "null"
            def decide(self, state):
                return Action(action_type=ActionType.WAIT, duration=1)
            def get_parameter_schema(self):
                return {}
        strategy = NullStrategy()

    # ── Core components ─────────────────────────────────────────────────
    capture = WindowCapture(title_pattern=args.window_title)
    ocr = OCREngine(engine=args.ocr, lang=args.lang,
                    confidence_threshold=args.confidence)
    controller = InputController(capture)
    sm = StateMachine()
    
    # Register Tower Idle screen keywords
    sm.register_screen(Screen.MAIN, ["wave", "damage", "dps", "floor"])
    sm.register_screen(Screen.UPGRADES, ["upgrade", "level", "cost"])
    sm.register_screen(Screen.SHOP, ["shop", "buy", "purchase", "gems", "offer"])
    sm.register_screen(Screen.DIALOG, ["ok", "cancel", "close", "confirm", "collect"])
    sm.register_screen(Screen.IDLE_REWARDS, ["idle", "rewards", "offline", "collect"])
    sm.register_screen(Screen.SETTINGS, ["settings", "options", "sound", "music"])
    sm.register_screen(Screen.LOADING, ["loading", "connecting"])
    
    scanner = UIScanner(capture, ocr, controller, sm)

    loop = AutomationLoop(
        capture=capture, ocr=ocr, controller=controller,
        state_machine=sm, scanner=scanner, strategy=strategy,
        tick=args.tick,
    )

    return loop


def main():
    parser = argparse.ArgumentParser(description="TowerControl – game automation")
    parser.add_argument("--game", default=config.ACTIVE_GAME,
                        help="Game definition to use (default: %(default)s)")
    parser.add_argument("--window-title", default=config.WINDOW_TITLE_PATTERN,
                        help="Window title substring to capture (default: %(default)s)")
    parser.add_argument("--ocr", default=config.OCR_ENGINE,
                        choices=["pytesseract", "easyocr"],
                        help="OCR backend (default: %(default)s)")
    parser.add_argument("--lang", default=config.OCR_LANG,
                        help="OCR language (default: %(default)s)")
    parser.add_argument("--confidence", type=float,
                        default=config.OCR_CONFIDENCE_THRESHOLD,
                        help="OCR confidence threshold (default: %(default)s)")
    parser.add_argument("--tick", type=float, default=config.LOOP_TICK,
                        help="Automation loop interval in seconds (default: %(default)s)")
    parser.add_argument("--host", default=config.WEB_HOST,
                        help="Web server host (default: %(default)s)")
    parser.add_argument("--port", type=int, default=config.WEB_PORT,
                        help="Web server port (default: %(default)s)")
    parser.add_argument("--no-auto", action="store_true",
                        help="Don't auto-start the bot; just open Web UI")
    args = parser.parse_args()

    log.info("TowerControl starting  (game=%s, ocr=%s, port=%d)",
             args.game, args.ocr, args.port)

    loop = build_components(args)

    # Register with web server
    from web.server import app, set_automation_loop
    set_automation_loop(loop)

    # Auto-start unless suppressed
    if not args.no_auto:
        loop.start()

    # Run uvicorn (blocking)
    import uvicorn
    log.info("Dashboard: http://%s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port,
                log_level="warning")


if __name__ == "__main__":
    main()
