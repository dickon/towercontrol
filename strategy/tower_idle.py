"""
Tower Idle strategy – currently observe-only.

Watches tier/wave progress and logs state.  All input to the game is
disabled by default; this strategy only decides to scan and observe.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional

from state.machine import GameState, Screen, UIElement
from .base import Action, ActionType, Strategy

log = logging.getLogger(__name__)


class TowerIdleStrategy(Strategy):
    name = "tower_idle"

    def __init__(self):
        super().__init__()
        self.parameters = {
            "auto_collect_rewards": True,
            "scan_interval": 30,           # full scan every N seconds
        }

    def decide(self, state: GameState) -> Action:
        if self.paused or not self.enabled:
            return Action(action_type=ActionType.WAIT, duration=1.0,
                          reason="Strategy paused")

        screen = state.current_screen

        # Log wave when available
        if state.wave:
            pos_str = ""
            if state.wave_pos:
                pos_str = f" @ ({state.wave_pos[0]:.3f},{state.wave_pos[1]:.3f})"
            log.info("Wave: %s%s", state.wave, pos_str)

        # ── 1. Check if we need a full scan ─────────────────────────────
        if (time.time() - state.last_full_scan
                > self.parameters["scan_interval"]):
            return Action(action_type=ActionType.FULL_SCAN,
                          reason="Periodic full scan")

        # ── 2. Default: just observe ────────────────────────────────────
        return Action(action_type=ActionType.SCAN_CURRENT, duration=1.0,
                      reason="Observing")

    # ── Web UI parameter schema ─────────────────────────────────────────

    def get_parameter_schema(self) -> Dict[str, Any]:
        return {
            "auto_collect_rewards": {"type": "bool", "label": "Auto-collect rewards"},
            "scan_interval": {
                "type": "int", "label": "Full scan interval (sec)",
                "min": 10, "max": 300,
            },
        }
