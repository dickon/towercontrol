"""
Game definition for "The Tower" idle game (a.k.a. Tower Idle).

This is a starting-point mapping.  Coordinates and patterns should be
calibrated against the actual BlueStacks window size.  The classify_element
rules are intentionally broad – they'll pick up most interactive text and
can be fine-tuned later.

Assumes a portrait-mode phone layout inside BlueStacks, roughly 540×960
or similar.  All coordinates are relative to the captured window region.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from ocr.engine import OCRResult
from state.machine import Screen, UIElement

from .base import GameDefinition, ResourcePattern, TabInfo


class TowerIdleGame(GameDefinition):
    name = "tower_idle"

    # ── Rough layout constants (will be auto-scaled) ────────────────────
    # These assume the game is rendered at approximately 540×960 inside
    # BlueStacks.  The scanner can adapt if the window is a different size.

    # Tab bar usually sits at the bottom
    TAB_Y = 920
    TAB_POSITIONS = {
        "battle":  (70,  TAB_Y),
        "heroes":  (170, TAB_Y),
        "skills":  (270, TAB_Y),
        "shop":    (370, TAB_Y),
        "more":    (470, TAB_Y),
    }

    SCROLL_REGION = (20, 200, 500, 650)   # main scrollable content area

    # ── interface ───────────────────────────────────────────────────────

    def get_tabs(self) -> List[TabInfo]:
        tabs = []
        for name, pos in self.TAB_POSITIONS.items():
            tabs.append(TabInfo(
                name=name,
                keywords=[name],
                click_position=pos,
                scroll_region=self.SCROLL_REGION,
                scroll_steps=6,
            ))
        return tabs

    def get_screen_keywords(self) -> Dict[Screen, List[str]]:
        return {
            Screen.MAIN:         ["wave", "stage", "floor", "damage", "dps"],
            Screen.HEROES:       ["heroes", "hero", "level", "upgrade"],
            Screen.SKILLS:       ["skills", "skill", "cooldown", "activate"],
            Screen.SHOP:         ["shop", "buy", "purchase", "gems", "offer"],
            Screen.DIALOG:       ["ok", "cancel", "close", "confirm", "collect"],
            Screen.IDLE_REWARDS: ["idle", "rewards", "offline", "collect"],
            Screen.SETTINGS:     ["settings", "options", "sound", "music"],
            Screen.LOADING:      ["loading", "connecting"],
        }

    def get_resource_patterns(self) -> List[ResourcePattern]:
        return [
            ResourcePattern("gold",  r"(?:gold|coins?)[:\s]*([0-9,.]+[kmbtKMBT]?)"),
            ResourcePattern("gems",  r"(?:gems?|diamonds?)[:\s]*([0-9,.]+)"),
            ResourcePattern("stage", r"(?:stage|floor|wave)[:\s]*([0-9,]+)"),
            ResourcePattern("dps",   r"(?:dps|damage)[:\s]*([0-9,.]+[kmbtKMBT]?)"),
        ]

    def classify_element(self, ocr: OCRResult, screen: Screen) -> Optional[UIElement]:
        text = ocr.text.strip()
        lower = text.lower()

        # ── Buttons ─────────────────────────────────────────────────────
        button_patterns = [
            r"\b(upgrade|buy|collect|claim|merge|prestige|start|ok|cancel"
            r"|close|confirm|watch|skip|equip|unlock|activate)\b"
        ]
        for pat in button_patterns:
            if re.search(pat, lower):
                return UIElement(
                    name=text, element_type="button", text=text,
                    bbox=ocr.bbox, confidence=ocr.confidence,
                    metadata={"action": re.search(pat, lower).group(1)},
                )

        # ── Tab labels ──────────────────────────────────────────────────
        tab_words = {"battle", "heroes", "skills", "shop", "more",
                     "settings", "home", "inventory"}
        if lower in tab_words:
            return UIElement(
                name=text, element_type="tab", text=text,
                bbox=ocr.bbox, confidence=ocr.confidence,
            )

        # ── Resource values ─────────────────────────────────────────────
        for rp in self.get_resource_patterns():
            if re.search(rp.pattern, text, re.IGNORECASE):
                return UIElement(
                    name=rp.name, element_type="resource", text=text,
                    bbox=ocr.bbox, confidence=ocr.confidence,
                )

        # ── Level / numeric indicators ──────────────────────────────────
        if re.match(r"^(lv\.?\s*\d+|level\s*\d+|\d+/\d+)$", lower):
            return UIElement(
                name="level_indicator", element_type="value", text=text,
                bbox=ocr.bbox, confidence=ocr.confidence,
            )

        # ── Costs (e.g. "1.2K") next to buttons ────────────────────────
        if re.match(r"^[0-9,.]+\s*[kmbtKMBT]?$", text):
            return UIElement(
                name="cost_or_value", element_type="value", text=text,
                bbox=ocr.bbox, confidence=ocr.confidence,
            )

        # ── Generic label (keep everything for state awareness) ─────────
        if len(text) >= 2:
            return UIElement(
                name="label", element_type="label", text=text,
                bbox=ocr.bbox, confidence=ocr.confidence,
            )

        return None
