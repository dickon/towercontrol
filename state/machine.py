"""
State machine – tracks which screen the game is on, what resources
are visible, and what actions are available.

The state is rebuilt every tick from fresh OCR data so it never goes stale.
Historical state is kept for the strategy engine to detect changes.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

from ocr.engine import OCRFrame, OCRResult

log = logging.getLogger(__name__)


# ── Enumerations ────────────────────────────────────────────────────────

class Screen(Enum):
    """High-level screen the game is showing."""
    UNKNOWN = auto()
    MAIN = auto()
    SHOP = auto()
    UPGRADES = auto()
    SETTINGS = auto()
    DIALOG = auto()         # modal popup
    LOADING = auto()
    IDLE_REWARDS = auto()
    # Games can extend this through the GameDefinition


# ── Data models ─────────────────────────────────────────────────────────

@dataclass
class UIElement:
    """A recognized interactive element on screen."""
    name: str
    element_type: str       # "button", "tab", "label", "value", "resource"
    text: str
    bbox: Tuple[int, int, int, int]
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def center(self) -> Tuple[int, int]:
        x, y, w, h = self.bbox
        return (x + w // 2, y + h // 2)


@dataclass
class ScreenState:
    """Snapshot of the UI at one moment in time."""
    screen: Screen = Screen.UNKNOWN
    elements: List[UIElement] = field(default_factory=list)
    resources: Dict[str, str] = field(default_factory=dict)
    ocr_frame: Optional[OCRFrame] = None
    timestamp: float = 0.0
    raw_texts: List[str] = field(default_factory=list)

    def find_elements(self, name_pattern: str = "",
                      element_type: str = "") -> List[UIElement]:
        out = []
        for e in self.elements:
            if name_pattern and name_pattern.lower() not in e.name.lower():
                continue
            if element_type and e.element_type != element_type:
                continue
            out.append(e)
        return out

    def has_text(self, pattern: str) -> bool:
        for t in self.raw_texts:
            if re.search(pattern, t, re.IGNORECASE):
                return True
        return False


@dataclass
class GameState:
    """Aggregate state built from scanning multiple screens/tabs."""
    current_screen: ScreenState = field(default_factory=ScreenState)
    tab_states: Dict[str, ScreenState] = field(default_factory=dict)
    resources: Dict[str, str] = field(default_factory=dict)
    wave: Optional[str] = None
    wave_pos: Optional[Tuple[float, float]] = None  # (x_frac, y_frac)
    last_full_scan: float = 0.0
    action_history: List[Dict[str, Any]] = field(default_factory=list)
    error_count: int = 0

    def record_action(self, action_type: str, details: Dict[str, Any] = None):
        entry = {
            "time": time.time(),
            "type": action_type,
            **(details or {}),
        }
        self.action_history.append(entry)
        # Keep last 200 actions
        if len(self.action_history) > 200:
            self.action_history = self.action_history[-200:]

    def to_dict(self) -> dict:
        """Serialisable summary for the web dashboard."""
        return {
            "current_screen": self.current_screen.screen.name,
            "wave": self.wave,
            "resources": self.resources,
            "element_count": len(self.current_screen.elements),
            "elements": [
                {"name": e.name, "type": e.element_type,
                 "text": e.text, "bbox": e.bbox}
                for e in self.current_screen.elements
            ],
            "raw_texts": self.current_screen.raw_texts[:60],
            "tabs_scanned": list(self.tab_states.keys()),
            "last_full_scan": self.last_full_scan,
            "recent_actions": self.action_history[-20:],
            "error_count": self.error_count,
        }


# ── State machine ───────────────────────────────────────────────────────

class StateMachine:
    """Maintains game state and updates it from OCR frames."""

    def __init__(self):
        self.state = GameState()
        self._screen_identifiers: Dict[Screen, List[str]] = {}

    def register_screen(self, screen: Screen, keywords: List[str]):
        """Register OCR keywords that identify a screen."""
        self._screen_identifiers[screen] = [k.lower() for k in keywords]

    def identify_screen(self, frame: OCRFrame) -> Screen:
        """Determine which screen is being shown based on OCR results."""
        texts_lower = [r.text.lower() for r in frame.results]
        all_text = " ".join(texts_lower)

        best_screen = Screen.UNKNOWN
        best_score = 0

        for screen, keywords in self._screen_identifiers.items():
            score = sum(1 for kw in keywords if kw in all_text)
            if score > best_score:
                best_score = score
                best_screen = screen

        return best_screen

    def update(self, frame: OCRFrame,
               classify_element=None) -> ScreenState:
        """Build a new ScreenState from an OCR frame.

        Args:
            frame: OCR results for the current capture.
            classify_element: optional callable(OCRResult, Screen) -> UIElement
                              provided by the game definition.
        """
        screen = self.identify_screen(frame)
        elements: List[UIElement] = []
        resources: Dict[str, str] = {}

        for r in frame.results:
            if classify_element:
                elem = classify_element(r, screen)
                if elem:
                    elements.append(elem)
                    if elem.element_type == "resource":
                        resources[elem.name] = elem.text

        ss = ScreenState(
            screen=screen,
            elements=elements,
            resources=resources,
            ocr_frame=frame,
            timestamp=time.time(),
            raw_texts=[r.text for r in frame.results],
        )
        self.state.current_screen = ss
        self.state.resources.update(resources)

        # Extract wave from OCR results positioned immediately to the right
        self._extract_tier_wave(frame)

        return ss

    def _extract_tier_wave(self, frame: OCRFrame):
        """Extract wave by finding numeric text to the right of labels."""
        w, h = frame.image_size
        if w == 0 or h == 0:
            log.warning("_extract_tier_wave: invalid image size (%d, %d)", w, h)
            return
            
        for r in frame.results:
            text_lower = r.text.lower().strip()
            
            # Found "wave" label
            if text_lower in ("wave", "w"):
                log.debug("Found wave label: '%s' at bbox %s", r.text, r.bbox)
                label_x = r.bbox[0] + r.bbox[2]
                label_y = r.bbox[1] + r.bbox[3] // 2
                
                best_num = None
                best_candidate = None
                best_dist = 999999
                for candidate in frame.results:
                    if re.match(r'^[0-9]+$', candidate.text.strip()):
                        cand_x = candidate.bbox[0]
                        cand_y = candidate.bbox[1] + candidate.bbox[3] // 2
                        if cand_x > label_x and abs(cand_y - label_y) < 20:
                            dist = cand_x - label_x
                            if dist < best_dist:
                                best_dist = dist
                                best_num = candidate.text.strip()
                                best_candidate = candidate
                if best_num and best_candidate:
                    self.state.wave = best_num
                    cx, cy = best_candidate.center
                    self.state.wave_pos = (cx / w if w else 0, cy / h if h else 0)
                    log.debug("Wave number found: %s at position (%.3f, %.3f)", 
                             best_num, self.state.wave_pos[0], self.state.wave_pos[1])
                elif text_lower in ("wave", "w"):
                    log.debug("Wave label found but no valid number to the right (checked %d candidates)", 
                             len([c for c in frame.results if re.match(r'^[0-9]+$', c.text.strip())]))

    def store_tab_state(self, tab_name: str, screen_state: ScreenState):
        """Save a scanned tab's state."""
        self.state.tab_states[tab_name] = screen_state
