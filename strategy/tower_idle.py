"""
Tower Idle strategy – decides what to do based on the current game state.

Priority order:
1. Dismiss dialogs / collect idle rewards
2. Upgrade heroes (cheapest first for efficiency, or strongest first)
3. Activate skills when available
4. Prestige when stuck
5. Visit shop for free offers
6. Idle / wait
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
            "auto_upgrade_heroes": True,
            "auto_activate_skills": True,
            "auto_collect_rewards": True,
            "auto_prestige": False,        # dangerous, off by default
            "prestige_stuck_minutes": 10,  # prestige if no progress for N min
            "upgrade_mode": "cheapest",    # "cheapest" or "strongest"
            "scan_interval": 30,           # full scan every N seconds
            "skill_check_interval": 15,    # check skills tab every N seconds
        }
        self._last_skill_check: float = 0
        self._last_progress_stage: str = ""
        self._last_progress_time: float = time.time()

    def decide(self, state: GameState) -> Action:
        if self.paused or not self.enabled:
            return Action(action_type=ActionType.WAIT, duration=1.0,
                          reason="Strategy paused")

        screen = state.current_screen

        # ── 1. Handle dialogs / popups first ────────────────────────────
        action = self._handle_dialogs(screen)
        if action:
            return action

        # ── 2. Collect idle rewards if visible ──────────────────────────
        if self.parameters["auto_collect_rewards"]:
            action = self._collect_rewards(screen)
            if action:
                return action

        # ── 3. Check if we need a full scan ─────────────────────────────
        if (time.time() - state.last_full_scan
                > self.parameters["scan_interval"]):
            return Action(action_type=ActionType.FULL_SCAN,
                          reason="Periodic full scan")

        # ── 4. Upgrade heroes ───────────────────────────────────────────
        if self.parameters["auto_upgrade_heroes"]:
            action = self._upgrade_heroes(state)
            if action:
                return action

        # ── 5. Activate skills ──────────────────────────────────────────
        if self.parameters["auto_activate_skills"]:
            if time.time() - self._last_skill_check > self.parameters["skill_check_interval"]:
                self._last_skill_check = time.time()
                action = self._activate_skills(state)
                if action:
                    return action

        # ── 6. Track progress for prestige decision ─────────────────────
        current_stage = state.resources.get("stage", "")
        if current_stage and current_stage != self._last_progress_stage:
            self._last_progress_stage = current_stage
            self._last_progress_time = time.time()

        # ── 7. Prestige if stuck ────────────────────────────────────────
        if self.parameters["auto_prestige"]:
            stuck_minutes = (time.time() - self._last_progress_time) / 60
            if stuck_minutes > self.parameters["prestige_stuck_minutes"]:
                return Action(action_type=ActionType.CLICK,
                              reason="Prestige (stuck for %.0f min)" % stuck_minutes,
                              priority=5)

        # ── 8. Default: just observe ────────────────────────────────────
        return Action(action_type=ActionType.SCAN_CURRENT, duration=1.0,
                      reason="Observing")

    # ── Tactic implementations ──────────────────────────────────────────

    def _handle_dialogs(self, screen) -> Optional[Action]:
        """Click 'OK', 'Close', 'Collect', 'Confirm' buttons on dialogs."""
        if screen.screen == Screen.DIALOG or screen.has_text(
                r"\b(ok|close|confirm|collect|claim|got it|tap to continue)\b"):
            for elem in screen.elements:
                if elem.element_type == "button":
                    action_word = elem.metadata.get("action", "")
                    if action_word in ("ok", "close", "confirm", "collect",
                                       "claim", "skip"):
                        cx, cy = elem.center
                        return Action(
                            action_type=ActionType.CLICK,
                            x=cx, y=cy, priority=10,
                            reason=f"Dismiss dialog: {elem.text}",
                        )
        return None

    def _collect_rewards(self, screen) -> Optional[Action]:
        """Collect idle/offline rewards."""
        if screen.screen == Screen.IDLE_REWARDS or screen.has_text(
                r"idle.*reward|offline.*reward"):
            for elem in screen.elements:
                if elem.element_type == "button":
                    action_word = elem.metadata.get("action", "")
                    if action_word in ("collect", "claim"):
                        cx, cy = elem.center
                        return Action(
                            action_type=ActionType.CLICK,
                            x=cx, y=cy, priority=9,
                            reason=f"Collect rewards: {elem.text}",
                        )
        return None

    def _upgrade_heroes(self, state: GameState) -> Optional[Action]:
        """Find upgrade buttons on the heroes tab and click them."""
        # Look in current screen OR cached heroes tab
        candidates: List[UIElement] = []

        for src in [state.current_screen,
                    state.tab_states.get("heroes")]:
            if src is None:
                continue
            for elem in src.elements:
                if (elem.element_type == "button"
                        and elem.metadata.get("action") == "upgrade"):
                    candidates.append(elem)

        if not candidates:
            return None

        # Pick based on mode
        if self.parameters["upgrade_mode"] == "cheapest":
            # The one nearest the bottom is typically cheapest in these games
            target = max(candidates, key=lambda e: e.center[1])
        else:
            # Strongest = nearest the top
            target = min(candidates, key=lambda e: e.center[1])

        cx, cy = target.center
        return Action(
            action_type=ActionType.CLICK,
            x=cx, y=cy, priority=5,
            reason=f"Upgrade: {target.text}",
        )

    def _activate_skills(self, state: GameState) -> Optional[Action]:
        """Check skills tab for activatable skills."""
        skills_state = state.tab_states.get("skills")
        if skills_state is None:
            # Need to scan skills tab first
            return Action(action_type=ActionType.FULL_SCAN,
                          reason="Need skills tab data")

        for elem in skills_state.elements:
            if (elem.element_type == "button"
                    and elem.metadata.get("action") == "activate"):
                cx, cy = elem.center
                return Action(
                    action_type=ActionType.CLICK,
                    x=cx, y=cy, priority=7,
                    reason=f"Activate skill: {elem.text}",
                )
        return None

    # ── Web UI parameter schema ─────────────────────────────────────────

    def get_parameter_schema(self) -> Dict[str, Any]:
        return {
            "auto_upgrade_heroes": {"type": "bool", "label": "Auto-upgrade heroes"},
            "auto_activate_skills": {"type": "bool", "label": "Auto-activate skills"},
            "auto_collect_rewards": {"type": "bool", "label": "Auto-collect rewards"},
            "auto_prestige": {"type": "bool", "label": "Auto-prestige when stuck"},
            "prestige_stuck_minutes": {
                "type": "int", "label": "Prestige after stuck (minutes)",
                "min": 1, "max": 120,
            },
            "upgrade_mode": {
                "type": "choice", "label": "Upgrade mode",
                "choices": ["cheapest", "strongest"],
            },
            "scan_interval": {
                "type": "int", "label": "Full scan interval (sec)",
                "min": 10, "max": 300,
            },
            "skill_check_interval": {
                "type": "int", "label": "Skill check interval (sec)",
                "min": 5, "max": 120,
            },
        }
