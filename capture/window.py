"""
Window capture for BlueStacks (or any named window).

Uses win32gui to locate the target window and mss for fast screen-region
capture.  Falls back to full-screen capture if the window can't be found.
"""

from __future__ import annotations

import ctypes
import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

try:
    import win32gui
    import win32con
    _HAS_WIN32 = True
except ImportError:
    _HAS_WIN32 = False

import mss

log = logging.getLogger(__name__)


@dataclass
class WindowRect:
    """Absolute screen coordinates of a window."""
    left: int
    top: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height

    def to_mss_monitor(self) -> dict:
        return {
            "left": self.left,
            "top": self.top,
            "width": self.width,
            "height": self.height,
        }


class WindowCapture:
    """Finds a window by (sub)title and captures its content."""

    def __init__(self, title_pattern: str = "BlueStacks"):
        self.title_pattern = title_pattern
        self._rect: Optional[WindowRect] = None
        self._last_find_attempt: float = 0
        self._find_cooldown: float = 2.0  # seconds between re-scans
        # Make this process DPI-aware so coordinates are physical pixels
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass

    # ── public API ──────────────────────────────────────────────────────

    def find_window(self) -> Optional[WindowRect]:
        """Locate the target window.  Returns None if not found."""
        now = time.time()
        if self._rect and now - self._last_find_attempt < self._find_cooldown:
            return self._rect
        self._last_find_attempt = now

        if _HAS_WIN32:
            self._rect = self._find_with_win32()
        else:
            log.warning("pywin32 not installed – cannot auto-find window")
            self._rect = None
        return self._rect

    def set_rect(self, left: int, top: int, width: int, height: int):
        """Manually set the capture region (for testing or override)."""
        self._rect = WindowRect(left, top, width, height)

    def capture(self) -> Optional[Image.Image]:
        """Grab a screenshot of the target window region.

        Returns a PIL Image in RGB mode, or None if the window isn't found.
        """
        rect = self.find_window()
        if rect is None:
            return None
        try:
            with mss.mss() as sct:
                raw = sct.grab(rect.to_mss_monitor())
                img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
            return img
        except Exception as exc:
            log.error("capture failed: %s", exc)
            return None

    def capture_np(self) -> Optional[np.ndarray]:
        """Same as capture() but returns a numpy BGR array (for OpenCV)."""
        img = self.capture()
        if img is None:
            return None
        arr = np.array(img)
        return arr[:, :, ::-1].copy()  # RGB → BGR

    @property
    def rect(self) -> Optional[WindowRect]:
        return self._rect

    # ── internal ────────────────────────────────────────────────────────

    def _find_with_win32(self) -> Optional[WindowRect]:
        """Enumerate top-level windows and match by title substring."""
        result: Optional[WindowRect] = None

        def _callback(hwnd, _extra):
            nonlocal result
            if not win32gui.IsWindowVisible(hwnd):
                return True
            title = win32gui.GetWindowText(hwnd)
            if self.title_pattern.lower() in title.lower():
                # Get the client area (the actual rendered content)
                try:
                    rect = win32gui.GetWindowRect(hwnd)
                    left, top, right, bottom = rect
                    w = right - left
                    h = bottom - top
                    if w > 50 and h > 50:  # sanity check
                        result = WindowRect(left, top, w, h)
                        log.info("Found window '%s' at %s", title, result)
                        return False  # stop enumeration
                except Exception:
                    pass
            return True

        try:
            win32gui.EnumWindows(_callback, None)
        except Exception:
            pass  # EnumWindows raises when callback returns False
        return result

    def bring_to_front(self):
        """Try to bring the target window to the foreground."""
        if not _HAS_WIN32:
            return
        def _callback(hwnd, _extra):
            if not win32gui.IsWindowVisible(hwnd):
                return True
            title = win32gui.GetWindowText(hwnd)
            if self.title_pattern.lower() in title.lower():
                try:
                    win32gui.SetForegroundWindow(hwnd)
                except Exception:
                    pass
                return False
            return True
        try:
            win32gui.EnumWindows(_callback, None)
        except Exception:
            pass
