"""
Input controller – sends mouse / keyboard events to the game window.

Uses pyautogui to drive the real mouse cursor.  The target window must be
visible on screen.  All coordinates are **relative to the capture region**
and get translated to absolute screen coordinates internally.
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Tuple

import pyautogui

from capture.window import WindowCapture, WindowRect

log = logging.getLogger(__name__)

# Safety settings
pyautogui.FAILSAFE = True          # move mouse to top-left corner to abort
pyautogui.PAUSE = 0.02             # tiny pause between pyautogui calls


class InputController:
    """Translate game-relative coordinates into real mouse actions."""

    def __init__(self, capture: WindowCapture,
                 click_pause: float = 0.08,
                 input_delay: float = 0.15):
        self.capture = capture
        self.click_pause = click_pause
        self.input_delay = input_delay
        self._enabled = True

    # ── enable / disable (web override) ─────────────────────────────────

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, val: bool):
        self._enabled = val
        log.info("InputController %s", "enabled" if val else "disabled")

    # ── coordinate helpers ──────────────────────────────────────────────

    def _abs(self, rel_x: int, rel_y: int) -> Optional[Tuple[int, int]]:
        """Convert capture-relative coords to absolute screen coords."""
        rect = self.capture.rect
        if rect is None:
            log.warning("No window rect – cannot convert coordinates")
            return None
        ax = rect.left + rel_x
        ay = rect.top + rel_y
        return (ax, ay)

    # ── actions ─────────────────────────────────────────────────────────

    def click(self, rel_x: int, rel_y: int, button: str = "left",
              clicks: int = 1) -> bool:
        """Click at a position relative to the game window."""
        if not self._enabled:
            log.debug("click suppressed (disabled)")
            return False
        pos = self._abs(rel_x, rel_y)
        if pos is None:
            return False
        ax, ay = pos
        log.info("click(%d, %d) → abs(%d, %d)", rel_x, rel_y, ax, ay)
        pyautogui.click(ax, ay, clicks=clicks, button=button,
                        interval=self.click_pause)
        time.sleep(self.click_pause)
        return True

    def double_click(self, rel_x: int, rel_y: int) -> bool:
        return self.click(rel_x, rel_y, clicks=2)

    def long_press(self, rel_x: int, rel_y: int,
                   duration: float = 1.0) -> bool:
        """Press and hold at a position."""
        if not self._enabled:
            return False
        pos = self._abs(rel_x, rel_y)
        if pos is None:
            return False
        ax, ay = pos
        log.info("long_press(%d, %d) for %.1fs", rel_x, rel_y, duration)
        pyautogui.moveTo(ax, ay)
        pyautogui.mouseDown()
        time.sleep(duration)
        pyautogui.mouseUp()
        time.sleep(self.click_pause)
        return True

    def drag(self, from_x: int, from_y: int,
             to_x: int, to_y: int, duration: float = 0.3) -> bool:
        """Drag from one position to another (relative coords)."""
        if not self._enabled:
            return False
        src = self._abs(from_x, from_y)
        dst = self._abs(to_x, to_y)
        if src is None or dst is None:
            return False
        log.info("drag (%d,%d)→(%d,%d)", from_x, from_y, to_x, to_y)
        pyautogui.moveTo(src[0], src[1])
        pyautogui.drag(dst[0] - src[0], dst[1] - src[1],
                       duration=duration)
        time.sleep(self.click_pause)
        return True

    def scroll(self, rel_x: int, rel_y: int, amount: int = -3) -> bool:
        """Scroll at position.  Negative = down, positive = up."""
        if not self._enabled:
            return False
        pos = self._abs(rel_x, rel_y)
        if pos is None:
            return False
        log.info("scroll(%d, %d, amount=%d)", rel_x, rel_y, amount)
        pyautogui.moveTo(pos[0], pos[1])
        pyautogui.scroll(amount)
        time.sleep(self.input_delay)
        return True

    def swipe_up(self, rel_x: int, rel_y: int,
                 distance: int = 300, duration: float = 0.3) -> bool:
        """Swipe up (like scrolling down in a list)."""
        return self.drag(rel_x, rel_y, rel_x, rel_y - distance, duration)

    def swipe_down(self, rel_x: int, rel_y: int,
                   distance: int = 300, duration: float = 0.3) -> bool:
        """Swipe down (like scrolling up in a list)."""
        return self.drag(rel_x, rel_y, rel_x, rel_y + distance, duration)

    def tap_key(self, key: str) -> bool:
        """Send a keyboard key press."""
        if not self._enabled:
            return False
        log.info("key press: %s", key)
        pyautogui.press(key)
        time.sleep(self.click_pause)
        return True

    def wait(self, seconds: float = None):
        """Pause for a bit."""
        if seconds is None:
            seconds = self.input_delay
        time.sleep(seconds)
