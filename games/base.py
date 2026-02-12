"""
Base class for game definitions.

Each supported game subclasses GameDefinition to describe its UI layout,
tab structure, screen identification keywords, and how to classify OCR
results into semantic UIElements.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from ocr.engine import OCRResult
from state.machine import Screen, StateMachine, UIElement


@dataclass
class TabInfo:
    """Describes a navigable tab in the game UI."""
    name: str
    keywords: List[str]              # OCR text that identifies this tab is active
    click_position: Tuple[int, int]  # relative coords to click to open this tab
    scroll_region: Optional[Tuple[int, int, int, int]] = None  # scrollable area (x,y,w,h)
    scroll_steps: int = 5            # how many scroll-downs to see everything


@dataclass
class ResourcePattern:
    """Pattern to extract a named resource value from OCR text."""
    name: str
    pattern: str          # regex with a capture group for the value
    region: Optional[Tuple[int, int, int, int]] = None  # screen region to look in


class GameDefinition(ABC):
    """Interface that each supported game must implement."""

    name: str = "unknown"

    @abstractmethod
    def get_tabs(self) -> List[TabInfo]:
        """Return all navigable tabs."""
        ...

    @abstractmethod
    def get_screen_keywords(self) -> Dict[Screen, List[str]]:
        """Map screens to identifying keywords."""
        ...

    @abstractmethod
    def get_resource_patterns(self) -> List[ResourcePattern]:
        """Patterns to extract resources from OCR text."""
        ...

    @abstractmethod
    def classify_element(self, ocr: OCRResult, screen: Screen) -> Optional[UIElement]:
        """Classify an OCR result into a UIElement (or None to skip)."""
        ...

    def register_with(self, sm: StateMachine):
        """Register screen keywords with the state machine."""
        for screen, kws in self.get_screen_keywords().items():
            sm.register_screen(screen, kws)

    def extract_resources(self, results: List[OCRResult]) -> Dict[str, str]:
        """Pull resource values out of OCR results."""
        resources: Dict[str, str] = {}
        all_text = " ".join(r.text for r in results)
        for rp in self.get_resource_patterns():
            m = re.search(rp.pattern, all_text, re.IGNORECASE)
            if m:
                resources[rp.name] = m.group(1) if m.lastindex else m.group(0)
        return resources
