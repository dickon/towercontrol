"""
State machine – tracks which screen the game is on, what resources
are visible, and what actions are available.

The state is rebuilt every tick from fresh OCR data so it never goes stale.
Historical state is kept for the strategy engine to detect changes.
"""

from __future__ import annotations

import logging
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
    HEROES = auto()
    SKILLS = auto()
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
        import re
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
        return ss

    def store_tab_state(self, tab_name: str, screen_state: ScreenState):
        """Save a scanned tab's state."""
        self.state.tab_states[tab_name] = screen_state
