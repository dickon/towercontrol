"""
UI Scanner – navigates through all tabs and scrolls to build a complete
picture of the game state.

Runs a full scan periodically, clicking each tab, scrolling through its
content, and running OCR on every visible frame.  The results are
aggregated into the StateMachine's GameState.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import config
from ocr.engine import OCREngine, OCRFrame, OCRResult
from state.machine import Screen, ScreenState, StateMachine, UIElement
from capture.window import WindowCapture
from input.controller import InputController

log = logging.getLogger(__name__)


@dataclass
class TabInfo:
    """Describes a navigable tab in the game UI."""
    name: str
    keywords: List[str]
    click_position: Tuple[int, int]
    scroll_region: Optional[Tuple[int, int, int, int]] = None
    scroll_steps: int = 5


@dataclass
class ResourcePattern:
    """Pattern to extract a named resource value from OCR text."""
    name: str
    pattern: str
    region: Optional[Tuple[int, int, int, int]] = None


class UIScanner:
    """Scan the full game UI by tabbing and scrolling."""

    def __init__(self, capture: WindowCapture, ocr: OCREngine,
                 controller: InputController, state_machine: StateMachine):
        self.capture = capture
        self.ocr = ocr
        self.controller = controller
        self.sm = state_machine
        self._scanning = False
        
        # Tower Idle game constants
        self._tabs = self._get_tabs()
        self._resource_patterns = self._get_resource_patterns()

    @property
    def is_scanning(self) -> bool:
        return self._scanning
    
    # ── Tower Idle Game Definition ─────────────────────────────────────
    
    def _get_tabs(self) -> List[TabInfo]:
        """Return all navigable tabs for Tower Idle."""
        tabs = []
        for name, pos in config.TAB_POSITIONS.items():
            tabs.append(TabInfo(
                name=name,
                keywords=[name],
                click_position=pos,
                scroll_region=config.SCROLL_REGION,
                scroll_steps=6,
            ))
        return tabs
    
    def _get_resource_patterns(self) -> List[ResourcePattern]:
        """Patterns to extract resources from OCR text."""
        return [
            ResourcePattern("gold",  r"(?:gold|coins?)[:\s]*([0-9,.]+[kmbtKMBT]?)"),
            ResourcePattern("gems",  r"(?:gems?|diamonds?)[:\s]*([0-9,.]+)"),
            ResourcePattern("wave",  r"(?:wave|w)[:\s]*([0-9,]+)"),
            ResourcePattern("dps",   r"(?:dps|damage)[:\s]*([0-9,.]+[kmbtKMBT]?)"),
        ]
    
    def _classify_element(self, ocr: OCRResult, screen: Screen) -> Optional[UIElement]:
        """Classify an OCR result into a UIElement."""
        text = ocr.text.strip()
        lower = text.lower()

        # Buttons
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

        # Tab labels
        tab_words = {"battle", "upgrades", "shop", "more",
                     "settings", "home", "inventory"}
        if lower in tab_words:
            return UIElement(
                name=text, element_type="tab", text=text,
                bbox=ocr.bbox, confidence=ocr.confidence,
            )

        # Resource values
        for rp in self._resource_patterns:
            if re.search(rp.pattern, text, re.IGNORECASE):
                return UIElement(
                    name=rp.name, element_type="resource", text=text,
                    bbox=ocr.bbox, confidence=ocr.confidence,
                )

        # Level / numeric indicators
        if re.match(r"^(lv\.?\s*\d+|level\s*\d+|\d+/\d+)$", lower):
            return UIElement(
                name="level_indicator", element_type="value", text=text,
                bbox=ocr.bbox, confidence=ocr.confidence,
            )

        # Costs (e.g. "1.2K") next to buttons
        if re.match(r"^[0-9,.]+\s*[kmbtKMBT]?$", text):
            return UIElement(
                name="cost_or_value", element_type="value", text=text,
                bbox=ocr.bbox, confidence=ocr.confidence,
            )

        # Generic label
        if len(text) >= 2:
            return UIElement(
                name="label", element_type="label", text=text,
                bbox=ocr.bbox, confidence=ocr.confidence,
            )

        return None
    
    def _extract_resources(self, results: List[OCRResult]) -> Dict[str, str]:
        """Pull resource values out of OCR results."""
        resources: Dict[str, str] = {}
        all_text = " ".join(r.text for r in results)
        for rp in self._resource_patterns:
            m = re.search(rp.pattern, all_text, re.IGNORECASE)
            if m:
                resources[rp.name] = m.group(1) if m.lastindex else m.group(0)
        return resources

    # ── Single-frame scan ───────────────────────────────────────────────

    def scan_current(self) -> ScreenState:
        """Capture + OCR the current view and update state."""
        img = self.capture.capture()
        if img is None:
            log.warning("scan_current: capture returned None")
            return self.sm.state.current_screen

        try:
            frame = self.ocr.process(img)
        except Exception as exc:
            log.error("OCR failed: %s", exc)
            self.sm.state.error_count += 1
            return self.sm.state.current_screen

        screen_state = self.sm.update(frame, self._classify_element)

        # Also extract resources
        resources = self._extract_resources(frame.results)
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
            for tab in self._tabs:
                self._scan_tab(tab)
            self.sm.state.last_full_scan = time.time()
            log.info("── Full UI scan complete (%d tabs) ──", len(self._tabs))
        except Exception as exc:
            log.error("Full scan failed: %s", exc, exc_info=True)
            self.sm.state.error_count += 1
        finally:
            self._scanning = False

    def _scan_tab(self, tab: TabInfo) -> None:
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
                self.controller.wait(0.5)  # extra time for Tesseract to clean up

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
