"""
Base class for game-playing strategies.

A strategy inspects the current GameState and decides what action to take.
Actions are returned as Action dataclass instances that the automation loop
executes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

from state.machine import GameState


class ActionType(Enum):
    CLICK = auto()
    SCROLL = auto()
    SWIPE_UP = auto()
    SWIPE_DOWN = auto()
    DRAG = auto()
    TAP_KEY = auto()
    WAIT = auto()
    SCAN_CURRENT = auto()
    FULL_SCAN = auto()
    NONE = auto()


@dataclass
class Action:
    """An action the bot wants to take."""
    action_type: ActionType
    x: int = 0
    y: int = 0
    x2: int = 0
    y2: int = 0
    amount: int = 0
    key: str = ""
    duration: float = 0.3
    reason: str = ""
    priority: int = 0        # higher = more urgent
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other: "Action"):
        return self.priority < other.priority


class Strategy(ABC):
    """Abstract strategy – subclass per game."""

    name: str = "base"

    def __init__(self):
        self.enabled = True
        self.paused = False
        self.parameters: Dict[str, Any] = {}

    @abstractmethod
    def decide(self, state: GameState) -> Action:
        """Given the current game state, decide what to do next."""
        ...

    @abstractmethod
    def get_parameter_schema(self) -> Dict[str, Any]:
        """Describe tuneable parameters for the web UI."""
        ...

    def set_parameter(self, key: str, value: Any):
        self.parameters[key] = value
