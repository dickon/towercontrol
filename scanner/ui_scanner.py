"""
UI Scanner – navigates through all tabs and scrolls to build a complete
picture of the game state.

Runs a full scan periodically, clicking each tab, scrolling through its
content, and running OCR on every visible frame.  The results are
aggregated into the StateMachine's GameState.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, List

from ocr.engine import OCREngine, OCRFrame
from state.machine import ScreenState, StateMachine
from capture.window import WindowCapture
from input.controller import InputController

if TYPE_CHECKING:
    from games.base import GameDefinition, TabInfo

log = logging.getLogger(__name__)


class UIScanner:
    """Scan the full game UI by tabbing and scrolling."""

    def __init__(self, capture: WindowCapture, ocr: OCREngine,
                 controller: InputController, state_machine: StateMachine,
                 game: "GameDefinition"):
        self.capture = capture
        self.ocr = ocr
        self.controller = controller
        self.sm = state_machine
        self.game = game
        self._scanning = False

    @property
    def is_scanning(self) -> bool:
        return self._scanning

    # ── Single-frame scan ───────────────────────────────────────────────

    def scan_current(self) -> ScreenState:
        """Capture + OCR the current view and update state."""
        img = self.capture.capture()
        if img is None:
            log.warning("scan_current: capture returned None")
            return self.sm.state.current_screen

        frame = self.ocr.process(img)
        screen_state = self.sm.update(frame, self.game.classify_element)

        # Also extract resources
        resources = self.game.extract_resources(frame.results)
        self.sm.state.resources.update(resources)

        return screen_state

    # ── Full scan ───────────────────────────────────────────────────────

    def full_scan(self) -> None:
        """Navigate every tab and scroll through it, building full state."""
        if self._scanning:
            log.warning("full_scan already in progress")
            return

        self._scanning = True
        log.info("── Full UI scan starting ──")

        try:
            tabs = self.game.get_tabs()
            for tab in tabs:
                self._scan_tab(tab)
            self.sm.state.last_full_scan = time.time()
            log.info("── Full UI scan complete (%d tabs) ──", len(tabs))
        except Exception as exc:
            log.error("Full scan failed: %s", exc, exc_info=True)
            self.sm.state.error_count += 1
        finally:
            self._scanning = False

    def _scan_tab(self, tab: "TabInfo") -> None:
        """Click a tab, then scroll through its content."""
        log.info("Scanning tab: %s", tab.name)

        # Click the tab
        tx, ty = tab.click_position
        self.controller.click(tx, ty)
        self.controller.wait(0.6)  # wait for tab transition animation

        # Capture the initial view
        combined_state = self.scan_current()

        # Scroll through the tab's content if it has a scroll region
        if tab.scroll_region:
            rx, ry, rw, rh = tab.scroll_region
            center_x = rx + rw // 2
            center_y = ry + rh // 2

            for step in range(tab.scroll_steps):
                # Swipe up to scroll down
                self.controller.swipe_up(center_x, center_y,
                                         distance=rh // 2,
                                         duration=0.25)
                self.controller.wait(0.4)

                # Capture after scroll
                ss = self.scan_current()

                # Merge elements (avoid duplicates by position)
                self._merge_elements(combined_state, ss)

            # Scroll back to top (swipe down several times)
            for _ in range(tab.scroll_steps + 1):
                self.controller.swipe_down(center_x, center_y,
                                           distance=rh // 2,
                                           duration=0.15)
            self.controller.wait(0.3)

        # Store the aggregated state for this tab
        self.sm.store_tab_state(tab.name, combined_state)

    def _merge_elements(self, target: ScreenState, source: ScreenState):
        """Merge elements from source into target, skipping near-duplicates."""
        for new_elem in source.elements:
            is_dup = False
            for existing in target.elements:
                # Same type + text near the same position → duplicate
                if (existing.element_type == new_elem.element_type
                        and existing.text == new_elem.text
                        and abs(existing.center[0] - new_elem.center[0]) < 30
                        and abs(existing.center[1] - new_elem.center[1]) < 30):
                    is_dup = True
                    break
            if not is_dup:
                target.elements.append(new_elem)

        # Merge raw texts (unique)
        existing_texts = set(target.raw_texts)
        for t in source.raw_texts:
            if t not in existing_texts:
                target.raw_texts.append(t)
                existing_texts.add(t)
