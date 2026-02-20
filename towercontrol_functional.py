#!/usr/bin/env python3
"""
TowerControl - Functional Style Single File Edition

Game automation for BlueStacks using screen capture + OCR.
Rewritten as pure functional code without classes.

Usage:
    python towercontrol_functional.py
    # Dashboard at http://127.0.0.1:7700
"""

import argparse
import datetime
import io
import os
import json
import logging
import re
import time
import pprint
from dataclasses import dataclass, field, replace
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, NamedTuple, TypedDict

import cv2
import math
import numpy as np
from PIL import Image, ImageDraw

# Third-party dependencies
import mss
import pyautogui

PERK_ROWS = [ (0, 0.273), (1, 0.373), (2, 0.473), (3, 0.573), (4, 0.673) ]

# Upgrade button positions (6 buttons in 3 rows × 2 columns)
# Format: (index, y_fraction, x_range)
# X ranges extended to capture prices displayed on the right side of buttons
UPGRADE_BUTTON_ROWS = [
    (0, 0.69, (0.20, 0.55)),  # Row 1, Left column (extended right to capture prices)
    (1, 0.69, (0.55, 0.95)),  # Row 1, Right column (extended right for prices)
    (2, 0.80, (0.20, 0.55)),  # Row 2, Left column
    (3, 0.80, (0.55, 0.95)),  # Row 2, Right column
    (4, 0.90, (0.20, 0.55)),  # Row 3, Left column
    (5, 0.90, (0.55, 0.95)),  # Row 3, Right column
]

# Attack upgrade labels
ATTACK_UPGRADES = [
    'Damage', 'Attack Speed', 'Critical Chance', 'Critical Factor', 'Range',
    'Damage Per Meter', 'Multishot Chance', 'Multishot Targets',
    'Rapid Fire Chance', 'Rapid Fire Duration', 'Bounce Shot Chance',
    'Bounce Shot Targets', 'Bounce Shot Range', 'Super Crit Chance',
    'Super Crit Mult', 'Rend Armor Chance', 'Rend Armor Mult'
]

# Defense upgrade labels
DEFENSE_UPGRADES = [
    'Health', 'Health Regen', 'Defense %', 'Defense Absolute', 'Thorn Damage',
    'Lifesteal', 'Knockback Chance', 'Knockback Force', 'Orb Speed', 'Orbs',
    'Shockwave Size', 'Shockwave Frequency', 'Land Mine Chance',
    'Land Mine Damage', 'Land Mine Radius', 'Death Defy', 'Wall Health',
    'Wall Rebuild'
]

# Utility upgrade labels
UTILITY_UPGRADES = [
    'Cash Bonus', 'Cash / Wave', 'Coins / Kill Bonus', 'Coins/Wave',
    'Free Attack Upgrade', 'Free Defense Upgrade', 'Free Utility Upgrade',
    'Interest / Wave', 'Recovery Amount', 'Max Recovery', 'Package Chance',
    'Enemy Attack Level Skip', 'Enemy Health Level Skip'
]

# Combined list for matching
ALL_UPGRADE_LABELS = ATTACK_UPGRADES + DEFENSE_UPGRADES + UTILITY_UPGRADES

PERK_CHOICES = [
    r'Perk Wave Requirement( -[\d\.]+%)?',
    r'(Increase )?Max Game Speed( by \-[\d\.]+)?',
    r'(x[\d\.]+ )?All Coins Bonuses',
    r'[\d\.]+ Damage',
    r'Chain Lightning Damage( x[\d\.]+)?',
    r'Golden Tower Bonus( x[\d\.]+)?',
    r'(\d*\s*)?More Smart Missiles',
    r'(\d*\s*)?(Wave )?On Death Wave',
    r'Bounce Shot( \+\d+)?',
    r'Black Hole Duration( \+[\d\.]+s)?',
    r'Unlock Chrono Field',
    r'(x[\d\.]+ )?Max Health',
    r'Upgrade Chance for all'
    r'Unlock poison swamp',
    r'(x[\d\.]+ )?Cash Bonus',
    r'Spotlight Damage Bonus( x[\d\.]+)?',
    r'Defense percent( \+[\d\.]+%)?',
    r'(x?[\d\.]+ )?Health Regen',
    r'Chrono Field Duration( \+[\d\.]+s)?',
    r'Swamp Radius( x[\d\.]+)?',
    r'Extra Set Of Inner Mines',
    r'Orbs( \+\d+)?',
    r'(x[\d\.]*\s*)?Defense Absolute',
    r'Land Mine Damage( x[\d\.]+)?',
    r'Unlock Spotlight',
    r'Interest( x[\d\.]+)?',    
    r'.*but boss.*',
    r'x1.89 coins, but -10.0% tower max',
    ]

UPGRADE_PRIORITY = [
    ('UTILITY', 'Enemy Attack level Skip', 1e8, True),
    ('UTILITY', 'Enemy Health level Skip', 1e8, True),
    ('ATTACK', 'Damage', None, False),
    ('DEFENSE', 'Health', None, False),
    ('DEFENSE', 'Shockwave Size', None, True),
    ('DEFENSE', 'Shockwave Frequency', None, True),
    ('DEFENSE', 'Land Mine Chance', None, True),
    ('DEFENSE', 'Land Mine Damage', None, True),
    ('DEFENSE', 'Land Mine Radius', None, True),
    ('DEFENSE', 'Death Defy', None, True),
    ('DEFENSE', 'Wall Health', 1e8, True),
    ('DEFENSE', 'Wall Rebuild', 1e8, True),
    ('DEFENSE', 'Health Regen', 1e8, False),
    ('UTILITY', 'Enemy Attack level Skip', 1e9, True),
    ('UTILITY', 'Enemy Health level Skip', 1e9, True),
    ('DEFENSE', 'Wall Health', 1e9, True),
    ('DEFENSE', 'Wall Rebuild', 1e9, True),
    ('DEFENSE', 'Defense Absoslute', None, False)
]

FLOATER_POSITIONS = [
    (0.6694, 0.3016)
]

# OCR
try:
    import pytesseract
except ImportError:
    pytesseract = None

try:
    import easyocr
except ImportError:
    easyocr = None

try:
    import win32gui
except ImportError:
    win32gui = None


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass(frozen=True)
class Config:
    """Immutable configuration"""
    base_dir: Path = Path(__file__).resolve().parent
    window_title: str = "BlueStacks"
    ocr_engine: str = "pytesseract"
    ocr_lang: str = "eng"
    ocr_confidence: float = 40.0
    tesseract_cmd: str = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    click_pause: float = 5.0
    swipe_pause: float = 1.0
    input_delay: float = 0.15
    loop_tick: float = 20.0
    web_host: str = "127.0.0.1"
    web_port: int = 7700
    tab_y: int = 920
    tab_positions: Dict[str, Tuple[int, int]] = field(default_factory=lambda: {
        "battle": (70, 920),
        "upgrades": (170, 920),
        "shop": (370, 920),
        "more": (470, 920),
    })
    scroll_region: Tuple[int, int, int, int] = (20, 200, 500, 650)
    upgrade_interval: float = 2.0            # seconds between upgrade purchase attempts
    upgrade_scroll_timeout: float = 120.0     # seconds scrolling down before switching to up
    free_upgrade_cycle_seconds: float = 120.0  # seconds per category in free-upgrade mode

    @property
    def screenshots_dir(self) -> Path:
        d = self.base_dir / "screenshots"
        d.mkdir(exist_ok=True)
        return d

    @property
    def debug_dir(self) -> Path:
        d = self.base_dir / "debug"
        d.mkdir(exist_ok=True)
        return d


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class Screen(Enum):
    UNKNOWN = auto()
    MAIN = auto()
    SHOP = auto()
    UPGRADES = auto()
    SETTINGS = auto()
    DIALOG = auto()
    LOADING = auto()
    IDLE_REWARDS = auto()


class ActionType(Enum):
    CLICK = auto()
    SCROLL = auto()
    SWIPE_UP = auto()
    SWIPE_DOWN = auto()
    TAP_KEY = auto()
    WAIT = auto()
    NONE = auto()


@dataclass(frozen=True)
class WindowRect:
    left: int
    top: int
    width: int
    height: int
    hwnd: int = 0  # Window handle for focus control

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height


@dataclass(frozen=True)
class OCRResult:
    text: str
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    confidence: float
    image_size: Tuple[int, int] = (0, 0)  # (width, height) of the frame

    @property
    def center(self) -> Tuple[int, int]:
        x, y, w, h = self.bbox
        return (x + w // 2, y + h // 2)
    
    @property
    def fx(self) -> float:
        """Fractional X coordinate (0.0 to 1.0)"""
        cx, _ = self.center
        return cx / self.image_size[0] if self.image_size[0] > 0 else 0.0
    
    @property
    def fy(self) -> float:
        """Fractional Y coordinate (0.0 to 1.0)"""
        _, cy = self.center
        return cy / self.image_size[1] if self.image_size[1] > 0 else 0.0
    
    def is_near(self, target_fx: float, target_fy: float, tolerance: float = 0.05) -> bool:
        """Check if center is within tolerance of target fractional coordinates."""
        return abs(self.fx - target_fx) < tolerance and abs(self.fy - target_fy) < tolerance
    def click(self, message: str = "Clicking OCR result"):
        do_click(message, self.fx, self.fy)
    def __repr__(self):
        return f'%.3f,%.3f "%s"' % (self.fx, self.fy, self.text)

@dataclass(frozen=True)
class OCRFrame:
    results: Tuple[OCRResult, ...] = ()
    image_size: Tuple[int, int] = (0, 0)

    @property
    def texts(self) -> List[str]:
        return [r.text for r in self.results]


@dataclass(frozen=True)
class UIElement:
    name: str
    element_type: str  # "button", "tab", "label", "value", "resource"
    text: str
    bbox: Tuple[int, int, int, int]
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def center(self) -> Tuple[int, int]:
        x, y, w, h = self.bbox
        return (x + w // 2, y + h // 2)


@dataclass(frozen=True)
class ScreenState:
    screen: Screen = Screen.UNKNOWN
    elements: Tuple[UIElement, ...] = ()
    resources: Dict[str, str] = field(default_factory=dict)
    raw_texts: Tuple[str, ...] = ()
    timestamp: float = 0.0


@dataclass(frozen=True)
class GameState:
    current_screen: ScreenState = field(default_factory=ScreenState)
    tab_states: Dict[str, ScreenState] = field(default_factory=dict)
    resources: Dict[str, str] = field(default_factory=dict)
    wave: Optional[str] = None
    wave_pos: Optional[Tuple[float, float]] = None
    wave_history: Tuple[Tuple[int, float], ...] = ()  # (wave_number, timestamp)
    action_history: Tuple[Dict[str, Any], ...] = ()
    error_count: int = 0
    battle_start_time: Optional[float] = None
    tier: Optional[int] = None


@dataclass(frozen=True)
class Action:
    action_type: ActionType
    x: int = 0
    y: int = 0
    x2: int = 0
    y2: int = 0
    amount: int = 0
    key: str = ""
    duration: float = 0.3
    reason: str = ""
    priority: int = 0


class UpgradeInfo(TypedDict):
    """Information about a detected upgrade button."""
    current_value: Optional[float]  # Current numerical value of the stat
    cost: Optional[float]  # Cost to upgrade (None if MAX)
    upgrades_to_purchase: int  # Number of upgrades to buy (default 1)
    cell_color: Tuple[int, int, int]  # RGB color of the button cell
    cell_color_name: str  # English name for the cell color (e.g. "dark red", "dark blue")
    label_position: Tuple[float, float]  # Fractional position (fx, fy) of the label
    button_position: Tuple[float, float]  # Fractional position (fx, fy) of the button center


# ============================================================================
# PURE FUNCTIONS - WINDOW CAPTURE
# ============================================================================

def find_window(title_pattern: str) -> Optional[WindowRect]:
    """Locate window by title substring. Pure function (reads system state)."""
    if not win32gui:
        return None

    result = None

    def callback(hwnd, _):
        nonlocal result
        if not win32gui.IsWindowVisible(hwnd):
            return True
        title = win32gui.GetWindowText(hwnd)
        if title_pattern.lower() in title.lower():
            try:
                left, top, right, bottom = win32gui.GetWindowRect(hwnd)
                w, h = right - left, bottom - top
                if w > 50 and h > 50:
                    result = WindowRect(left, top, w, h, hwnd=hwnd)
                    return False
            except Exception:
                pass
        return True

    try:
        win32gui.EnumWindows(callback, None)
    except Exception:
        pass

    return result


def capture_window(rect: WindowRect) -> Optional[Image.Image]:
    """Capture screenshot of window region."""
    try:
        monitor = {
            "left": rect.left,
            "top": rect.top,
            "width": rect.width,
            "height": rect.height,
        }
        with mss.mss() as sct:
            raw = sct.grab(monitor)
            img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
        return img
    except Exception:
        return None


def image_to_bgr_array(img: Image.Image) -> np.ndarray:
    """Convert PIL Image to OpenCV BGR array."""
    arr = np.array(img)
    return arr[:, :, ::-1].copy()


def save_debug_files(img: Image.Image, frame: OCRFrame, config: Config, prefix: str = "capture") -> None:
    """Save debug image and text file with fractional positions of OCR elements."""
    log = logging.getLogger(__name__)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    
    try:
        # Fixed filenames that overwrite
        img_path = config.debug_dir / f"{prefix}.png"
        txt_path = config.debug_dir / f"{prefix}.txt"
        annotated_path = config.debug_dir / f"{prefix}_annotated.png"
        
        # Save original image
        img.save(img_path)
        log.debug(f"Saved debug image: {img_path}")
        
        # Create annotated image with boxes and coordinates
        annotated_img = img.copy()
        draw = ImageDraw.Draw(annotated_img)
        img_width, img_height = frame.image_size
        
        for result in frame.results:
            x, y, w, h = result.bbox
            cx, cy = result.center
            
            # Draw bounding box
            box_color = (0, 255, 0) if result.confidence >= 80 else (255, 165, 0)  # Green for high conf, orange for lower
            draw.rectangle([x, y, x + w, y + h], outline=box_color, width=2)
            
            # Draw text and fractional coordinates
            label = f"{result.text}\n({result.fx:.3f}, {result.fy:.3f})"
            # Position label above the box if there's room, otherwise below
            label_y = max(0, y - 25) if y > 30 else y + h + 2
            draw.text((x, label_y), label, fill=box_color)
        
        annotated_img.save(annotated_path)
        log.debug(f"Saved annotated debug image: {annotated_path}")
        
        # Create text table
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"Debug Capture: {timestamp}\n")
            f.write(f"Image Size: {img_width} x {img_height}\n")
            f.write(f"Total OCR Elements: {len(frame.results)}\n")
            f.write("\n" + "="*100 + "\n")
            f.write(f"{'Text':<30} | {'X_Frac':<8} | {'Y_Frac':<8} | {'W_Frac':<8} | {'H_Frac':<8} | {'Conf':<6}\n")
            f.write("="*100 + "\n")
            
            for result in frame.results:
                x, y, w, h = result.bbox
                w_frac = w / img_width if img_width > 0 else 0.0
                h_frac = h / img_height if img_height > 0 else 0.0
                
                text_display = result.text[:28] + ".." if len(result.text) > 30 else result.text
                f.write(f"r.text == '{text_display}' and r.is_near({result.fx:>8.4f}, {result.fy:>8.4f})\n")
        
        log.debug(f"Saved debug text: {txt_path}")
    except Exception as e:
        log.error(f"Failed to save debug files: {e}")


# ============================================================================
# PURE FUNCTIONS - OCR
# ============================================================================

def preprocess_image(img: np.ndarray) -> List[np.ndarray]:
    """Generate optimized preprocessed variants for OCR. Returns list of images to try."""
    # Upscale 1.5x instead of 2x - good balance between quality and speed
    h, w = img.shape[:2]
    img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    variants = []

    # Strategy 1: Adaptive threshold - best all-around performer for game UI
    # Faster than OTSU and handles uneven lighting well
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 10
    )
    variants.append(adaptive)

    # Strategy 2: Simple sharpening with high-contrast threshold
    # Good for crisp game text, faster than OTSU
    sharp_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])  # Lighter sharpening kernel
    sharpened = cv2.filter2D(gray, -1, sharp_kernel)
    _, sharp_thresh = cv2.threshold(sharpened, 127, 255, cv2.THRESH_BINARY)
    variants.append(sharp_thresh)

    return variants


def _bbox_overlap(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> float:
    """Compute IoU between two (x,y,w,h) bounding boxes."""
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _merge_ocr_results(all_results: List[List[OCRResult]]) -> List[OCRResult]:
    """Merge OCR results from multiple preprocessing runs, deduplicating by bbox overlap."""
    merged = []
    
    for results in all_results:
        for r in results:
            # Check if we already have a result overlapping this bbox
            duplicate = False
            for i, existing in enumerate(merged):
                if _bbox_overlap(r.bbox, existing.bbox) > 0.4:
                    # Keep the one with higher confidence
                    if r.confidence > existing.confidence:
                        merged[i] = r
                    duplicate = True
                    break
            if not duplicate:
                merged.append(r)
    
    return merged


def run_pytesseract_ocr(img: np.ndarray, config: Config) -> List[OCRResult]:
    """Run pytesseract OCR with optimized single PSM mode. Returns list of OCR results."""
    if not pytesseract:
        return []

    pytesseract.pytesseract.tesseract_cmd = config.tesseract_cmd

    # Use PSM 11 (sparse text) - best for game UI elements
    # Single mode is much faster than trying multiple modes
    try:
        data = pytesseract.image_to_data(
            img, lang=config.ocr_lang,
            output_type=pytesseract.Output.DICT,
            config="--psm 11 --oem 3"  # OEM 3 = default (LSTM + legacy)
        )

        results = []
        n = len(data["text"])
        for i in range(n):
            txt = data["text"][i].strip()
            conf = float(data["conf"][i])
            if conf >= config.ocr_confidence and len(txt) >= 2:
                bbox = (data["left"][i], data["top"][i],
                       data["width"][i], data["height"][i])
                results.append(OCRResult(text=txt, bbox=bbox, confidence=conf))
        return results
    except Exception:
        return []


def run_easyocr_ocr(img: np.ndarray, reader, config: Config) -> List[OCRResult]:
    """Run EasyOCR. Returns list of OCR results."""
    if not reader:
        return []

    raw = reader.readtext(img)
    results = []
    for bbox_pts, text, conf in raw:
        xs = [p[0] for p in bbox_pts]
        ys = [p[1] for p in bbox_pts]
        x, y = int(min(xs)), int(min(ys))
        w, h = int(max(xs) - x), int(max(ys) - y)
        if conf * 100 >= config.ocr_confidence and text.strip():
            results.append(OCRResult(
                text=text.strip(),
                bbox=(x, y, w, h),
                confidence=conf * 100
            ))
    return results


def _ocr_perk_row_recovery(gray_upscaled: np.ndarray, config: Config,
                           scale_factor: float, orig_w: int, orig_h: int,
                           missing_rows: set = None) -> List[OCRResult]:
    """Run targeted per-row OCR recovery for light-on-dark perk text.

    Standard full-image OCR often fails to detect white/light text on
    dark coloured backgrounds (e.g. dark red, dark green).  This function
    crops each PERK_ROW region, inverts the grayscale and applies OTSU
    thresholding so that the light text becomes dark-on-white — which
    Tesseract handles reliably.

    Only rows listed in *missing_rows* are processed.  If *missing_rows*
    is ``None``, all rows with a dark background (mean gray < 100) are
    attempted.

    Args:
        gray_upscaled: Grayscale image at 1.5× scale
        config: Application configuration
        scale_factor: The upscale factor used (1.5)
        orig_w: Original image width
        orig_h: Original image height
        missing_rows: Set of row indices to attempt recovery for.
                      If None, all dark rows are attempted.

    Returns:
        List of OCRResult objects with bboxes in original-image coordinates.
    """
    if not pytesseract:
        return []

    h2, w2 = gray_upscaled.shape[:2]
    recovery_results: List[OCRResult] = []

    for row_idx, fy in PERK_ROWS:
        if missing_rows is not None and row_idx not in missing_rows:
            continue
        y_lo = max(0, int((fy - 0.05) * h2))
        y_hi = min(h2, int((fy + 0.05) * h2))
        x_lo = int(0.40 * w2)
        x_hi = min(w2, int(0.82 * w2))
        roi = gray_upscaled[y_lo:y_hi, x_lo:x_hi]

        # Only attempt recovery on dark-background rows
        if roi.mean() > 100:
            continue

        inv_roi = cv2.bitwise_not(roi)
        _, inv_thresh = cv2.threshold(inv_roi, 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        try:
            pytesseract.pytesseract.tesseract_cmd = config.tesseract_cmd
            data = pytesseract.image_to_data(
                inv_thresh, lang=config.ocr_lang,
                output_type=pytesseract.Output.DICT,
                config="--psm 7 --oem 3",
                timeout=5,
            )
        except Exception:
            continue

        n = len(data["text"])
        for i in range(n):
            txt = data["text"][i].strip()
            conf = float(data["conf"][i])
            if conf >= config.ocr_confidence and len(txt) >= 2:
                # Map ROI coordinates back to full upscaled image
                bx = data["left"][i] + x_lo
                by = data["top"][i] + y_lo
                bw, bh = data["width"][i], data["height"][i]
                # Scale back to original image coordinates
                recovery_results.append(OCRResult(
                    text=txt,
                    bbox=(int(bx / scale_factor), int(by / scale_factor),
                          int(bw / scale_factor), int(bh / scale_factor)),
                    confidence=conf,
                    image_size=(orig_w, orig_h),
                ))

    return recovery_results


def _ocr_upgrade_header_recovery(arr: np.ndarray, config: Config) -> List[OCRResult]:
    """Recover the upgrade section header (UTILITY / ATTACK / DEFENSE UPGRADES).

    The coloured banner text is frequently garbled by standard OCR (e.g. 'WN'
    instead of 'UTILITY').  This crops the header band at y≈[0.60, 0.67],
    upscales 2×, tries both inverted and straight adaptive thresholding, and
    runs Tesseract in single-line mode.

    On success it returns a single synthetic OCRResult with text equal to the
    detected category word ('UTILITY', 'ATTACK', or 'DEFENSE'), centred in the
    header band at approximately (fx=0.33, fy=0.635) so that the existing
    automation_loop_tick detection logic matches it unchanged.

    Returns an empty list if no category word is found.
    """
    if not pytesseract:
        return []

    try:
        h, w = arr.shape[:2]

        # Header band position (fractional)
        y_lo = max(0, int(0.60 * h))
        y_hi = min(h, int(0.67 * h))
        x_lo = int(0.08 * w)
        x_hi = min(w, int(0.58 * w))

        roi = arr[y_lo:y_hi, x_lo:x_hi]
        if roi.size == 0:
            return []

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        upscaled = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

        pytesseract.pytesseract.tesseract_cmd = config.tesseract_cmd

        # Try inverted first (light text on dark banner), then straight.
        # Block size 31 and PSM 6 (uniform text block) give the best results
        # for the coloured game-UI banner font.
        for proc in (cv2.bitwise_not(upscaled), upscaled):
            thresh = cv2.adaptiveThreshold(
                proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 31, 10,
            )
            try:
                text = pytesseract.image_to_string(
                    thresh, lang=config.ocr_lang,
                    config="--psm 6 --oem 3",
                    timeout=5,
                ).strip().upper()
            except Exception:
                continue

            for category in ('UTILITY', 'ATTACK', 'DEFENSE'):
                if category in text:
                    # Emit a synthetic token centred in the header band
                    cx = (x_lo + x_hi) // 2
                    cy = (y_lo + y_hi) // 2
                    bw = (x_hi - x_lo) // 2
                    bh = (y_hi - y_lo)
                    return [OCRResult(
                        text=category,
                        bbox=(cx - bw // 2, cy - bh // 2, bw, bh),
                        confidence=70.0,
                        image_size=(w, h),
                    )]

        return []

    except Exception:
        return []


def process_ocr(img: Image.Image, config: Config, ocr_reader=None) -> OCRFrame:
    """Run OCR pipeline on image with optimized preprocessing strategies."""
    arr = image_to_bgr_array(img)
    h, w = arr.shape[:2]
    variants = preprocess_image(arr)

    if config.ocr_engine == "easyocr" and ocr_reader:
        # EasyOCR: just run on first variant
        results = run_easyocr_ocr(variants[0], ocr_reader, config)
    else:
        # Pytesseract: run on both variants and merge
        all_variant_results = []
        for variant in variants:
            variant_results = run_pytesseract_ocr(variant, config)
            all_variant_results.append(variant_results)
        results = _merge_ocr_results(all_variant_results)

    # Scale bboxes back to original image coords (we upscaled 1.5x)
    scale_factor = 1.5
    scaled_results = []
    for r in results:
        x, y, bw, bh = r.bbox
        scaled_results.append(OCRResult(
            text=r.text,
            bbox=(int(x / scale_factor), int(y / scale_factor), 
                  int(bw / scale_factor), int(bh / scale_factor)),
            confidence=r.confidence,
            image_size=(w, h)
        ))

    # Recovery pass: per-row OCR for light-on-dark perk text
    # Only attempt recovery for perk rows that had no results from the
    # standard OCR pass, to avoid redundant tesseract calls.
    detected_rows = set()
    for r in scaled_results:
        for row_idx, fy in PERK_ROWS:
            if abs(r.fy - fy) < 0.05 and 0.45 < r.fx < 0.82:
                detected_rows.add(row_idx)
                break

    missing_rows = {idx for idx, _ in PERK_ROWS} - detected_rows
    if missing_rows:
        gray_upscaled = cv2.cvtColor(
            cv2.resize(arr, None, fx=scale_factor, fy=scale_factor,
                       interpolation=cv2.INTER_CUBIC),
            cv2.COLOR_BGR2GRAY,
        )
        recovery = _ocr_perk_row_recovery(
            gray_upscaled, config, scale_factor, w, h, missing_rows
        )
        if recovery:
            scaled_results = _merge_ocr_results([scaled_results, recovery])

    # Recovery pass: upgrade section header (UTILITY / ATTACK / DEFENSE UPGRADES)
    # Only run if no category marker is already present near the expected position.
    header_found = any(
        r.text.upper() in ('UTILITY', 'ATTACK', 'DEFENSE') and r.is_near(0.333, 0.632, 0.15)
        for r in scaled_results
    )
    if not header_found:
        header_recovery = _ocr_upgrade_header_recovery(arr, config)
        if header_recovery:
            scaled_results = _merge_ocr_results([scaled_results, header_recovery])

    # Save preprocessed debug image (first variant)
    try:
        debug_path = config.debug_dir / "preprocessed.png"
        cv2.imwrite(str(debug_path), variants[0])
    except Exception:
        pass

    return OCRFrame(
        results=tuple(scaled_results),
        image_size=(w, h)
    )


# ============================================================================
# PURE FUNCTIONS - STATE MACHINE
# ============================================================================

def classify_ocr_result(ocr: OCRResult) -> Optional[UIElement]:
    """Classify an OCR result into a UI element. Pure function."""
    text = ocr.text.strip()
    lower = text.lower()

    # Button patterns
    button_pattern = (
        r"\b(upgrade|buy|collect|claim|merge|prestige|start|ok|cancel"
        r"|close|confirm|watch|skip|equip|unlock|activate)\b"
    )
    match = re.search(button_pattern, lower)
    if match:
        return UIElement(
            name=text, element_type="button", text=text,
            bbox=ocr.bbox, confidence=ocr.confidence,
            metadata={"action": match.group(1)}
        )

    # Tabs
    tab_words = {"battle", "upgrades", "shop", "more", "settings", "home", "inventory"}
    if lower in tab_words:
        return UIElement(
            name=text, element_type="tab", text=text,
            bbox=ocr.bbox, confidence=ocr.confidence
        )

    # Resources
    resource_patterns = [
        (r"(?:gold|coins?)[:\s]*([0-9,.]+[kmbtKMBT]?)", "gold"),
        (r"(?:gems?|diamonds?)[:\s]*([0-9,.]+)", "gems"),
        (r"(?:wave|w)[:\s]*([0-9,]+)", "wave"),
        (r"(?:dps|damage)[:\s]*([0-9,.]+[kmbtKMBT]?)", "dps"),
    ]
    for pattern, name in resource_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return UIElement(
                name=name, element_type="resource", text=text,
                bbox=ocr.bbox, confidence=ocr.confidence
            )

    # Level indicators
    if re.match(r"^(lv\.?\s*\d+|level\s*\d+|\d+/\d+)$", lower):
        return UIElement(
            name="level_indicator", element_type="value", text=text,
            bbox=ocr.bbox, confidence=ocr.confidence
        )

    # Numeric values
    if re.match(r"^[0-9,.]+\s*[kmbtKMBT]?$", text):
        return UIElement(
            name="cost_or_value", element_type="value", text=text,
            bbox=ocr.bbox, confidence=ocr.confidence
        )

    # Generic label
    if len(text) >= 2:
        return UIElement(
            name="label", element_type="label", text=text,
            bbox=ocr.bbox, confidence=ocr.confidence
        )

    return None


def extract_wave_from_frame(frame: OCRFrame) -> Tuple[Optional[str], Optional[Tuple[float, float]]]:
    """Extract wave number and position. Pure function."""
    log = logging.getLogger(__name__)
    w, h = frame.image_size
    if w == 0 or h == 0:
        return None, None

    # Strategy 1: Check if "wave" or "w" appears with a number in the same OCR result
    for r in frame.results:
        text = r.text.strip()
        # Match patterns like "Wave 123", "W 123", "Wave: 123", "w:123", etc.
        match = re.search(r'(?:wave|w)[:\s]*([0-9,]+)', text, re.IGNORECASE)
        if match:
            wave_num = match.group(1).replace(',', '')
            log.debug(f"Wave detected (strategy 1): '{wave_num}' from text '{text}' at ({r.fx:.3f}, {r.fy:.3f})")
            return wave_num, (r.fx, r.fy)

    # Strategy 2: Find standalone "wave"/"w" label and number to the right
    for r in frame.results:
        text_lower = r.text.lower().strip()
        if text_lower in ("wave", "w"):
            label_x = r.bbox[0] + r.bbox[2]
            label_y = r.bbox[1] + r.bbox[3] // 2

            # Find numeric to the right
            best_num = None
            best_candidate = None
            best_dist = 999999

            for candidate in frame.results:
                if re.match(r'^[0-9,]+$', candidate.text.strip()):
                    cand_x = candidate.bbox[0]
                    cand_y = candidate.bbox[1] + candidate.bbox[3] // 2
                    if cand_x > label_x and abs(cand_y - label_y) < 50:
                        dist = cand_x - label_x
                        if dist < best_dist:
                            best_dist = dist
                            best_num = candidate.text.strip().replace(',', '')
                            best_candidate = candidate

            if best_num and best_candidate:
                log.debug(f"Wave detected (strategy 2): '{best_num}' at ({best_candidate.fx:.3f}, {best_candidate.fy:.3f})")
                return best_num, (best_candidate.fx, best_candidate.fy)

    # Strategy 3: Look for any text starting with "wave" or "w" followed by digits
    for r in frame.results:
        text_lower = r.text.lower().strip()
        if text_lower.startswith(('wave', 'w')):
            # Extract any digits from the text
            digits = re.findall(r'[0-9]+', r.text)
            if digits:
                wave_num = digits[0]
                log.debug(f"Wave detected (strategy 3): '{wave_num}' from text '{r.text}' at ({r.fx:.3f}, {r.fy:.3f})")
                return wave_num, (r.fx, r.fy)

    # Debug: log all OCR results if wave not found (but only occasionally to avoid spam)
    if len(frame.results) > 0:
        import random
        if random.random() < 0.05:  # 5% of the time
            log.debug(f"Wave not found. OCR results: {[r.text for r in frame.results]}")

    return None, None


def do_click(message, click_x_frac, click_y_frac):
    global ctx
    log = logging.getLogger(__name__)
    w = ctx.window_rect.width
    h = ctx.window_rect.height
    click_x = int(click_x_frac * w)
    click_y = int(click_y_frac * h)
    log.info(f'{message} at ({click_x_frac:.4f}, {click_y_frac:.4f}) -> pixels ({click_x}, {click_y})')
    if ctx.window_rect and ctx.config:
        execute_click(click_x, click_y, ctx.window_rect, ctx.config, reason=message)

def close_perks():
    do_click("Closing perks mode by clicking 'X'", 0.878, 0.100)


# ============================================================================
# PURE FUNCTIONS - PERK DETECTION
# ============================================================================

def collect_perk_texts(frame: OCRFrame) -> dict:
    """Collect OCR texts grouped by perk row/slot index.
    
    Scans OCR results and groups text fragments that fall within
    the known perk row y-positions and valid x-range.  Texts within
    each row are sorted by x-position so they appear in natural
    left-to-right reading order.
    
    Args:
        frame: OCRFrame with detected text results
    
    Returns:
        dict mapping slot index (int) -> list of text strings
    """
    # Collect (fx, text) pairs per row so we can sort by x
    perk_items: dict = {}
    for r in frame.results:
        for (row, dy) in PERK_ROWS:
            if abs(r.fy - dy) < 0.05 and r.fx > 0.45 and r.fx < 0.82:
                perk_items.setdefault(row, [])
                perk_items[row].append((r.fx, r.text))
    # Sort each row's texts by x-position and extract text only
    return {row: [t for _, t in sorted(items)] for row, items in perk_items.items()}


def filter_selected_perks(perk_text: dict) -> dict:
    """Filter out perk rows that contain 'Selected' and rows below them.
    
    When a row contains the word 'Selected', it means that perk was
    already chosen. All rows at or after the selected row are removed.
    
    Args:
        perk_text: dict mapping slot index -> list of text strings
    
    Returns:
        Filtered dict with selected/below rows removed
    """
    for i in range(5):
        if 'Selected' in ' '.join(perk_text.get(i, [])):
            return {k: v for k, v in perk_text.items() if k < i}
    return perk_text


def join_perk_texts(perk_text: dict) -> dict:
    """Join text fragments for each perk row into single strings.
    
    Args:
        perk_text: dict mapping slot index -> list of text strings
    
    Returns:
        dict mapping slot index (int) -> joined text string
    """
    return {row: " ".join(texts) for row, texts in perk_text.items()}


def match_perk_priorities(perk_text_join: dict) -> Tuple[list, bool]:
    """Match joined perk texts against known PERK_CHOICES patterns.
    
    Args:
        perk_text_join: dict mapping slot index -> joined text string
    
    Returns:
        Tuple of (priority_list, all_matched) where:
        - priority_list: list of (row, pattern, priority_index) tuples
          sorted by priority (lowest index = highest priority)
        - all_matched: True if every row matched a pattern
    """
    log = logging.getLogger(__name__)
    perk_text_priority = []
    all_matched = True
    
    for row, text in perk_text_join.items():
        hit = False
        for idx, choice_pattern in enumerate(PERK_CHOICES):
            if re.search(choice_pattern, text, re.IGNORECASE):
                perk_text_priority.append((row, choice_pattern, idx))
                hit = True
                break
        if not hit:
            log.debug(f"No perk choice pattern matched for row {row} with text '{text}'")
            all_matched = False
    
    perk_text_priority.sort(key=lambda x: x[2])
    return perk_text_priority, all_matched


def detect_perks(frame: OCRFrame) -> dict:
    """Run the full perk detection pipeline on an OCR frame.
    
    This is the main entry point for perk detection, combining:
    1. Collecting perk texts from OCR results by slot position
    2. Filtering out already-selected perks
    3. Joining text fragments per row
    4. Matching against known perk patterns
    
    Args:
        frame: OCRFrame with detected text results
    
    Returns:
        dict with keys:
        - 'perk_text': raw collected text by row (before filtering)
        - 'perk_text_join': joined text per row (after filtering)
        - 'perk_text_priority': list of (row, pattern, priority_index) sorted by priority
        - 'all_matched': True if every row matched a known pattern
    """
    perk_text = collect_perk_texts(frame)
    perk_text = filter_selected_perks(perk_text)
    perk_text_join = join_perk_texts(perk_text)
    perk_text_priority, all_matched = match_perk_priorities(perk_text_join)
    
    return {
        'perk_text': perk_text,
        'perk_text_join': perk_text_join,
        'perk_text_priority': perk_text_priority,
        'all_matched': all_matched,
    }


def record_action_in_state(state: GameState, action: Action) -> GameState:
    """Record action in history. Pure function."""
    entry = {
        "time": time.time(),
        "type": action.action_type.name,
        "reason": action.reason,
        "x": action.x,
        "y": action.y,
    }
    new_history = (*state.action_history, entry)
    if len(new_history) > 200:
        new_history = new_history[-200:]
    
    return replace(state, action_history=new_history)


def parse_number_with_suffix(text: str) -> Optional[float]:
    """Parse a number string with optional K/M/B/T suffix into a float.

    Examples:
        "1.5K" -> 1500.0
        "3.62B" -> 3620000000.0
        "28.10" -> 28.10
        "MAX" -> None

    Args:
        text: String containing number with optional suffix

    Returns:
        Parsed float value, or None if unparseable
    """
    text = text.strip().upper()

    if text in ["MAX", "???"]:
        return None

    # Match number with optional suffix
    match = re.match(r'^([0-9,.]+)\s*([KMBT])?$', text)
    if not match:
        return None

    try:
        value = float(match.group(1).replace(',', ''))
        suffix = match.group(2)

        if suffix:
            multipliers = {'K': 1_000, 'M': 1_000_000, 'B': 1_000_000_000, 'T': 1_000_000_000_000}
            value *= multipliers.get(suffix, 1)

        return value
    except (ValueError, AttributeError):
        return None


def get_cell_color(img: Optional[Image.Image], y_frac: float, x_min: float, x_max: float,
                   y_tolerance: float = 0.08) -> Tuple[int, int, int]:
    """Extract the dominant color from a cell region.

    Args:
        img: PIL Image to analyze
        y_frac: Fractional Y position of cell center (0.0-1.0)
        x_min: Fractional X start of cell region (0.0-1.0)
        x_max: Fractional X end of cell region (0.0-1.0)
        y_tolerance: Vertical tolerance for cell region

    Returns:
        RGB tuple of the dominant color, or (128, 128, 128) if detection fails
    """
    if img is None:
        return (128, 128, 128)  # Default gray

    try:
        img_array = np.array(img)
        h, w = img_array.shape[:2]

        # Calculate pixel coordinates
        y_start = max(0, int((y_frac - y_tolerance) * h))
        y_end = min(h, int((y_frac + y_tolerance) * h))
        x_start = max(0, int(x_min * w))
        x_end = min(w, int(x_max * w))

        # Extract cell region
        region = img_array[y_start:y_end, x_start:x_end]

        if region.size == 0:
            return (128, 128, 128)

        # Calculate average color (simple dominant color approximation)
        avg_color = region.mean(axis=(0, 1))
        return (int(avg_color[0]), int(avg_color[1]), int(avg_color[2]))

    except Exception:
        return (128, 128, 128)


def rgb_to_color_name(r: int, g: int, b: int) -> str:
    """Convert an RGB color to an approximate English name.

    Returns names like "dark red", "dark blue", "gray", etc.
    """
    brightness = (r + g + b) / 3
    dark = brightness < 100
    light = brightness > 180

    # Check for grays (low saturation)
    max_c = max(r, g, b)
    min_c = min(r, g, b)
    saturation = (max_c - min_c) / max_c if max_c > 0 else 0

    if saturation < 0.15:
        if brightness < 50:
            return "black"
        elif brightness > 200:
            return "white"
        elif dark:
            return "dark gray"
        elif light:
            return "light gray"
        else:
            return "gray"

    # Determine dominant hue
    if r >= g and r >= b:
        if g > b * 1.5:
            hue = "yellow" if g > r * 0.7 else "orange"
        else:
            hue = "red"
    elif g >= r and g >= b:
        if b > r * 1.5:
            hue = "cyan"
        else:
            hue = "green"
    else:  # b dominant
        if r > g * 1.5:
            hue = "purple"
        else:
            hue = "blue"

    if dark:
        return f"dark {hue}"
    elif light:
        return f"light {hue}"
    return hue


def is_button_red(img: Optional[Image.Image], y_frac: float, x_min: float, x_max: float,
                  y_tolerance: float = 0.08) -> bool:
    """Check if a button region is predominantly dark red (indicating MAX status).

    Detects dark red buttons using HSV color space to identify MAX upgrade buttons
    that may have clipped or unreadable text.

    Args:
        img: PIL Image to analyze
        y_frac: Fractional Y position of button center (0.0-1.0)
        x_min: Fractional X start of button region (0.0-1.0)
        x_max: Fractional X end of button region (0.0-1.0)
        y_tolerance: Vertical tolerance for button region

    Returns:
        True if button region contains significant dark red color
    """
    if img is None:
        return False
    
    try:
        # Convert to numpy array
        img_array = np.array(img)
        h, w = img_array.shape[:2]
        
        # Focus on the button area (lower-right of cell) where the Max/cost button sits,
        # not the entire cell region. Using the full region causes false positives from
        # nearby red UI elements like the "DEFENSE UPGRADES" header.
        x_mid = x_min + (x_max - x_min) * 0.4  # Right ~60% of cell
        y_start = max(0, int((y_frac - 0.01) * h))
        y_end = min(h, int((y_frac + 0.04) * h))
        x_start = max(0, int(x_mid * w))
        x_end = min(w, int(x_max * w))
        
        # Extract button region
        region = img_array[y_start:y_end, x_start:x_end]
        
        if region.size == 0:
            return False
        
        # Convert to HSV for better color detection
        region_hsv = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
        
        # Define red color range in HSV for dark red MAX buttons
        # Red wraps around in HSV, so we need two ranges
        # Lowered Value (brightness) threshold from 100 to 50 to catch dark red
        lower_red1 = np.array([0, 80, 50])  # Hue 0-10, Saturation 80+, Value 50+
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 80, 50])  # Hue 160-180, Saturation 80+, Value 50+
        upper_red2 = np.array([180, 255, 255])

        # Create masks for red color
        mask1 = cv2.inRange(region_hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(region_hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)
        
        # Calculate percentage of red pixels
        red_pixels = np.count_nonzero(red_mask)
        total_pixels = region.shape[0] * region.shape[1]
        red_percentage = (red_pixels / total_pixels) * 100

        # Consider button red (MAX) if more than 10% of pixels are dark red
        # Lowered from 15% to 10% to better catch dark red MAX buttons
        is_red = red_percentage > 10.0
        
        if is_red:
            logging.getLogger(__name__).debug(
                f"Dark red MAX button detected at y={y_frac:.3f}, x=[{x_min:.3f}-{x_max:.3f}]: {red_percentage:.1f}% red pixels"
            )
        
        return is_red
        
    except Exception as e:
        logging.getLogger(__name__).warning(f"Red detection failed: {e}")
        return False


def _find_upgrade_boxes(img: Image.Image) -> List[Tuple[int, int, int, int]]:
    """Find white-bordered upgrade boxes by scanning for bright separator lines.

    Looks at the lower portion of the image (where the upgrade panel lives),
    projects pixel brightness horizontally and vertically to find separator
    lines, then forms boxes from consecutive pairs of those lines.

    Returns a list of (x, y, w, h) in pixel coordinates,
    sorted top-to-bottom then left-to-right.
    """
    arr = np.array(img)
    h_img, w_img = arr.shape[:2]
    bgr = arr[:, :, ::-1].copy()
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # The upgrade panel occupies roughly y=0.58–0.98 and x=0.15–1.0
    y_top = int(h_img * 0.59)
    x_left = int(w_img * 0.05)
    panel = gray[y_top:, x_left:]
    ph, pw = panel.shape

    # Threshold: near-white pixels mark box border lines
    _, binary = cv2.threshold(panel, 200, 255, cv2.THRESH_BINARY)

    # Project brightness: fraction of white pixels per row / column
    h_density = binary.mean(axis=1) / 255.0
    v_density = binary.mean(axis=0) / 255.0

    def cluster_peaks(indices: List[int], gap: int = 8) -> List[int]:
        """Cluster adjacent indices into representative midpoints."""
        if not indices:
            return []
        groups: List[List[int]] = [[indices[0]]]
        for idx in indices[1:]:
            if idx - groups[-1][-1] <= gap:
                groups[-1].append(idx)
            else:
                groups.append([idx])
        return [int(np.mean(g)) for g in groups]

    # True separator bars span the full width/height of the panel and produce
    # 80-90 % white pixels per row/column.  Noise from in-box text is typically
    # < 35 %.  Use 0.65 / 0.50 to cleanly separate signal from noise.
    # Use a wide cluster gap (25 px) so that adjacent double-border lines
    # (two parallel white bars ~16-18 px apart) merge into a single separator.
    h_peaks = cluster_peaks([y for y, d in enumerate(h_density) if d > 0.65], gap=25)
    v_peaks = cluster_peaks([x for x, d in enumerate(v_density) if d > 0.35], gap=25)

    log = logging.getLogger(__name__)
    log.debug(
        f"_find_upgrade_boxes: h_peaks={[p + y_top for p in h_peaks]}, "
        f"v_peaks={[p + x_left for p in v_peaks]}"
    )

    # Build boxes from consecutive pairs of separator lines
    boxes: List[Tuple[int, int, int, int]] = []
    min_bh = h_img * 0.05   # at least 5 % of image height
    max_bh = h_img * 0.22   # at most 22 % of image height
    min_bw = w_img * 0.10   # at least 10 % of image width
    max_bw = w_img * 0.55   # at most 55 % of image width

    for i in range(len(h_peaks) - 1):
        y1 = h_peaks[i] + y_top
        y2 = h_peaks[i + 1] + y_top
        bh = y2 - y1
        if not (min_bh <= bh <= max_bh):
            continue
        for j in range(len(v_peaks) - 1):
            x1 = v_peaks[j] + x_left
            x2 = v_peaks[j + 1] + x_left
            bw = x2 - x1
            if not (min_bw <= bw <= max_bw):
                continue
            if bh >= bw:   # boxes are landscape (wider than tall)
                continue
            boxes.append((x1, y1, bw, bh))

    return sorted(boxes, key=lambda b: (b[1], b[0]))


def _ocr_box_purchase_count(img: Image.Image, box_x: int, box_y: int,
                             box_w: int, box_h: int, config: Config) -> Optional[int]:
    """Extract the "xNNN" purchase count from the top-right corner of an upgrade box.

    The count (e.g. "x504") appears as small cyan text in the top-right area of
    each box. We crop that corner tightly, upscale heavily, invert, and OCR
    with a digit-only whitelist.

    Returns the count as int (> 1), or None if not found.
    """
    if not pytesseract or img is None:
        return None
    try:
        arr = image_to_bgr_array(img)
        h_arr, w_arr = arr.shape[:2]

        # Top-right ~25 % width × ~35 % height of the box.
        # x_frac=0.75 keeps the crop narrow so that the "x" prefix and digits
        # fit on one line without neighbouring text.  h_frac=0.35 is tall
        # enough to capture the xNNN count even when it sits lower in the box.
        x_start = box_x + int(box_w * 0.75)
        y_start = box_y
        x_end = min(w_arr, box_x + box_w)
        y_end = box_y + int(box_h * 0.35)

        roi = arr[y_start:y_end, x_start:x_end]
        if roi.size == 0:
            return None

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        upscaled = cv2.resize(gray, None, fx=4.0, fy=4.0, interpolation=cv2.INTER_CUBIC)
        inverted = cv2.bitwise_not(upscaled)
        adaptive = cv2.adaptiveThreshold(
            inverted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 10,
        )
        padded = cv2.copyMakeBorder(adaptive, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)

        pytesseract.pytesseract.tesseract_cmd = config.tesseract_cmd
        text = pytesseract.image_to_string(
            padded, lang=config.ocr_lang,
            config="--psm 7 --oem 3 -c tessedit_char_whitelist=x0123456789",
            timeout=5,
        ).strip()

        match = re.match(r'^x(\d+)', text, re.IGNORECASE)
        if match:
            count = int(match.group(1))
            if count > 1:
                return count
        return None
    except Exception:
        return None


def _ocr_box_cost(img: Image.Image, box_x: int, box_y: int,
                  box_w: int, box_h: int, config: Config) -> Optional[str]:
    """Run targeted OCR on the cost-button area (right ~55 % × lower ~45 %) of a box.

    The cost button sits in the lower-right of each upgrade box. On dark
    backgrounds the text (e.g. "$7.86B" or "Max") is light-on-dark and can be
    missed by the global OCR pass; we invert and threshold to recover it.

    Returns the raw detected text, or None if nothing found.
    """
    if not pytesseract or img is None:
        return None
    try:
        arr = image_to_bgr_array(img)
        h_arr, w_arr = arr.shape[:2]

        x_start = box_x + int(box_w * 0.40)
        y_start = box_y + int(box_h * 0.52)
        x_end = min(w_arr, box_x + box_w)
        y_end = min(h_arr, box_y + box_h)

        roi = arr[y_start:y_end, x_start:x_end]
        if roi.size == 0:
            return None

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        if gray.mean() > 120:   # skip bright-background buttons (already readable)
            return None

        upscaled = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        inverted = cv2.bitwise_not(upscaled)
        thresh = cv2.adaptiveThreshold(
            inverted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 10,
        )

        pytesseract.pytesseract.tesseract_cmd = config.tesseract_cmd
        text = pytesseract.image_to_string(
            thresh, lang=config.ocr_lang,
            config="--psm 7 --oem 3",
            timeout=5,
        ).strip()
        return text if text else None
    except Exception:
        return None


def _ocr_box_value(img: Image.Image, box_x: int, box_y: int,
                   box_w: int, box_h: int, config: Config) -> Optional[str]:
    """Run targeted OCR on the stat-value area of an upgrade box.

    The current stat value (e.g. "2.65T", "95.90%", "107.30B/sec") lives
    inside a bordered sub-widget in the upper-right of each upgrade box.
    Strategy:
    1. Crop the right ~60 % × full height of the box.
    2. Detect the golden/orange inner-border sub-widget by HSV masking.
       If found, crop to just the upper ~55 % of its interior (skipping the
       cost/Max button that lives in the lower half of the same sub-widget).
    3. Apply CLAHE for local contrast enhancement, then OTSU threshold to
       produce clean black-on-white text for Tesseract.

    Returns the raw OCR text, or None if nothing found.
    """
    if not pytesseract or img is None:
        return None
    try:
        arr = image_to_bgr_array(img)
        h_arr, w_arr = arr.shape[:2]

        x_start = box_x + int(box_w * 0.40)
        x_end = min(w_arr, box_x + box_w)
        full_roi = arr[box_y:box_y + box_h, x_start:x_end]
        if full_roi.size == 0:
            return None

        # ── Step 1: find the golden/orange inner sub-widget border ─────────
        hsv = cv2.cvtColor(full_roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            np.array([10, 80, 80]),    # lower: orange-ish hue
            np.array([45, 255, 255]),  # upper: yellow-ish hue
        )
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_roi: Optional[np.ndarray] = None
        for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
            cx, cy, cw, ch = cv2.boundingRect(cnt)
            if cw * ch < 800:
                continue
            margin = 5
            # Crop to just the upper ~55 % of the sub-widget interior
            # (the lower portion is occupied by the cost / "Max" button)
            sub = full_roi[
                cy + margin: cy + int(ch * 0.55),
                cx + margin: cx + cw - margin,
            ]
            if sub.size > 0:
                best_roi = sub
                break

        # Fall back to the full right-portion if no sub-widget was detected
        if best_roi is None:
            best_roi = arr[box_y: box_y + int(box_h * 0.60), x_start:x_end]

        gray = cv2.cvtColor(best_roi, cv2.COLOR_BGR2GRAY)
        if gray.mean() > 120:   # skip if area is unexpectedly bright
            return None

        # ── Step 2: CLAHE + OTSU for robust light-on-dark text ────────────
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(gray)
        upscaled = cv2.resize(enhanced, None, fx=3.0, fy=3.0,
                               interpolation=cv2.INTER_CUBIC)
        _, thresh = cv2.threshold(upscaled, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Invert to black-on-white then pad
        padded = cv2.copyMakeBorder(
            cv2.bitwise_not(thresh), 20, 20, 20, 20,
            cv2.BORDER_CONSTANT, value=255,
        )

        pytesseract.pytesseract.tesseract_cmd = config.tesseract_cmd
        text = pytesseract.image_to_string(
            padded, lang=config.ocr_lang,
            config="--psm 7 --oem 3",
            timeout=5,
        ).strip()
        return text if text else None
    except Exception:
        return None


def _match_upgrade_label(texts: List[str]) -> Optional[str]:
    """Match text fragments against the known upgrade label list.

    Sorts known labels by word-count descending so longer / more-specific
    labels (e.g. "Thorn Damage") match before shorter ones (e.g. "Damage").

    Returns the matched label string, or None if no match found.
    """
    texts_lower = [t.lower() for t in texts]
    sorted_labels = sorted(ALL_UPGRADE_LABELS, key=lambda l: len(l.split()), reverse=True)

    for known_label in sorted_labels:
        known_words = known_label.lower().replace('/', ' ').split()
        if all(
            any(word in lt or lt in word for lt in texts_lower)
            for word in known_words
        ):
            return known_label
    return None


def _parse_upgrade_value(texts: List[str]) -> Optional[float]:
    """Parse the current stat value from a list of OCR text fragments.

    Accepts percentages (e.g. "95.90%"), plain numbers with K/M/B/T suffix
    (e.g. "689.52M"), and numbers followed by a unit (e.g. "107.30B/sec").

    Returns the parsed float, or None if nothing parseable is found.
    """
    for text in texts:
        text = text.strip()
        # Percentage (e.g. "95.90%")
        pct = re.match(r'^([0-9,.]+)\s*%', text)
        if pct:
            try:
                return float(pct.group(1).replace(',', ''))
            except ValueError:
                pass
        # Number optionally followed by a unit suffix (e.g. "107.30B/sec")
        num = re.match(r'^([0-9,.]+\s*[KMBT]?)(?:/\w+)?$', text, re.IGNORECASE)
        if num:
            parsed = parse_number_with_suffix(num.group(1).strip())
            if parsed is not None:
                return parsed
    return None


def _parse_upgrade_cost(texts: List[str]) -> Tuple[bool, Optional[float]]:
    """Parse the upgrade cost from button-area text fragments.

    Returns ``(is_max, cost_value)`` where:
    - ``is_max=True`` means the button shows "Max" (no cost to pay).
    - ``cost_value`` is the numeric cost as float, or ``None`` if MAX / unparseable.
    """
    for text in texts:
        stripped = text.strip()
        lower = stripped.lower()
        if lower in ('max', 'max.', 'max!'):
            return True, None
        # Strip leading currency symbol then parse
        cleaned = stripped.lstrip('$').strip()
        parsed = parse_number_with_suffix(cleaned)
        if parsed is not None:
            return False, parsed
    return False, None


def detect_upgrade_buttons(frame: OCRFrame, img: Optional[Image.Image] = None,
                           config: Optional[Config] = None) -> Dict[str, UpgradeInfo]:
    """Detect upgrade buttons and parse their four key fields per box.

    Box-based approach:
    1. Scan the upgrade panel for white-bordered boxes via brightness line
       projection (``_find_upgrade_boxes``).
    2. For each box, collect the OCR results from ``frame.results`` that fall
       within its pixel bounds.
    3. Split the box into sub-regions and classify results:
       - Left ~45 %      → label (upgrade name)
       - Right-upper     → current stat value / upgrade level
       - Right-lower     → cost button ("Max" or "$NNN.NNX")
       - Top-right corner → purchase count ("xNNN", small text)
    4. Match the label against the known upgrade list; fall back to targeted
       Tesseract OCR for any field that the global pass missed.

    Returns a dict mapping label → ``UpgradeInfo``.  Every detected box is
    logged at INFO level with all four fields::

        Upgrade box 'Health': level=2650000000000.0, cost=MAX, x_count=1
        Upgrade box 'Defense Absolute': level=689520000.0, cost=7860000000.0, x_count=504
    """
    log = logging.getLogger(__name__)
    upgrades: Dict[str, UpgradeInfo] = {}

    if img is None:
        log.warning("detect_upgrade_buttons: no image provided")
        return upgrades

    img_w, img_h = img.size

    # ── Step 1: Locate the white-bordered boxes ────────────────────────────
    boxes_px = _find_upgrade_boxes(img)
    log.debug(f"detect_upgrade_buttons: found {len(boxes_px)} candidate boxes")

    # save out a debug image with the box corners marked in yellow
    debug_img = img.copy()
    draw = ImageDraw.Draw(debug_img)
    for box_x, box_y, box_w, box_h in boxes_px:
        draw.rectangle([box_x, box_y, box_x + box_w, box_y + box_h], outline='yellow', width=2) 
    debug_img.save('debug_upgrade_boxes.png')

    if not boxes_px:
        log.warning(
            "detect_upgrade_buttons: no upgrade boxes detected; "
            "upgrade panel may not be visible"
        )
        return upgrades

    # ── Step 2: Process each box ────────────────────────────────────────────
    for box_x, box_y, box_w, box_h in boxes_px:
        fx_min = box_x / img_w
        fx_max = (box_x + box_w) / img_w
        fy_min = box_y / img_h
        fy_max = (box_y + box_h) / img_h
        fy_center = (fy_min + fy_max) / 2

        # Dividers between sub-regions (fractional within the image)
        fx_mid = fx_min + (fx_max - fx_min) * 0.45  # label | value+button
        fy_mid = fy_min + (fy_max - fy_min) * 0.55  # value | button

        # Collect OCR results whose centre falls inside this box
        box_results = [
            r for r in frame.results
            if fx_min <= r.fx <= fx_max and fy_min <= r.fy <= fy_max
        ]
        # ── Label (left portion of box) ────────────────────────────────────
        label_texts = [
            r.text.strip() for r in box_results
            if r.fx < fx_mid and len(r.text.strip()) >= 2
        ]
        label = _match_upgrade_label(label_texts)

        # ── Current value (upper-right portion) ────────────────────────────
        value_texts = [
            r.text.strip() for r in box_results
            if r.fx >= fx_mid and r.fy < fy_mid
        ]
        current_value = _parse_upgrade_value(value_texts)

        # ── Cost (lower-right / button portion) ────────────────────────────
        cost_texts = [
            r.text.strip() for r in box_results
            if r.fx >= fx_mid and r.fy >= fy_mid
        ]
        is_max, cost_value = _parse_upgrade_cost(cost_texts)

        # Color-based MAX check (dark-red button → MAX even if text was missed)
        if not is_max and is_button_red(img, fy_center, fx_min, fx_max):
            is_max = True
            cost_value = None

        # ── Purchase count (xNNN to right of box)
        purchase_count = None
        for r in box_results:            
            m = re.match(r'^x(\d+)$', r.text.strip(), re.IGNORECASE)

            if m and r.fx > 0.5:
                log.info(f'purhcase count match: "{r.text.strip()}" at fx={r.fx:.3f}, fy={r.fy:.3f}')
                n = int(m.group(1))
                if n > 1:
                    purchase_count = n
                    break

        # ── Targeted OCR fallbacks ─────────────────────────────────────────
        if config is not None:
            if current_value is None and label is not None:
                ocr_text = _ocr_box_value(img, box_x, box_y, box_w, box_h, config)
                if ocr_text:
                    recovered = _parse_upgrade_value([ocr_text])
                    if recovered is not None:
                        current_value = recovered
                        log.debug(
                            f"Recovered value for '{label}' via targeted OCR: "
                            f"'{ocr_text}' → {current_value}"
                        )

            if not is_max and cost_value is None and label is not None:
                ocr_text = _ocr_box_cost(img, box_x, box_y, box_w, box_h, config)
                if ocr_text:
                    if 'max' in ocr_text.lower():
                        is_max = True
                        cost_value = None
                        log.debug(
                            f"Recovered MAX for '{label}' via targeted OCR: '{ocr_text}'"
                        )
                    else:
                        cost_m = re.search(
                            r'\$?\s*([0-9,.]+\s*[KMBT])(?!\s*/)',
                            ocr_text, re.IGNORECASE,
                        )
                        if cost_m:
                            parsed = parse_number_with_suffix(cost_m.group(1).strip())
                            if parsed is not None:
                                cost_value = parsed
                                log.debug(
                                    f"Recovered cost for '{label}' via targeted OCR: "
                                    f"'{ocr_text}' → {parsed}"
                                )

            if purchase_count == 1:
                count = _ocr_box_purchase_count(img, box_x, box_y, box_w, box_h, config)
                if count is not None:
                    purchase_count = count
                    log.debug(f"Recovered purchase count x{count} via targeted OCR")

        # ── Skip boxes with no matched label ───────────────────────────────
        if label is None:
            log.debug(
                f"Box ({box_x},{box_y},{box_w},{box_h}): no known label matched "
                f"from texts {label_texts!r}"
            )
            continue

        # ── Assemble UpgradeInfo ────────────────────────────────────────────
        cell_color = get_cell_color(img, fy_center, fx_min, fx_max)
        label_r = next(
            (r for r in box_results
             if any(w.lower() in r.text.lower() for w in label.split())),
            None,
        )
        label_pos = (label_r.fx, label_r.fy) if label_r else (fx_min + 0.05, fy_center)
        button_pos = ((fx_min + fx_max) / 2, fy_center)

        info: UpgradeInfo = {
            'current_value': current_value,
            'cost': cost_value,
            'upgrades_to_purchase': purchase_count,
            'cell_color': cell_color,
            'cell_color_name': rgb_to_color_name(*cell_color),
            'label_position': label_pos,
            'button_position': button_pos,
        }
        upgrades[label] = info

        log.debug(
            f"Upgrade box '{label}': "
            f"level={current_value!r}, "
            f"cost={'MAX' if is_max else repr(cost_value)}, "
            f"x_count={purchase_count}"
        )

    return upgrades


# ============================================================================
# PURE FUNCTIONS - STRATEGY
# ============================================================================

def load_gem_template(config: Config) -> Optional[np.ndarray]:
    """Load gem template image for detection."""
    gem_path = config.base_dir / "gem.png"
    if not gem_path.exists():
        logging.getLogger(__name__).warning(f"Gem template not found: {gem_path}")
        return None
    
    try:
        template = cv2.imread(str(gem_path), cv2.IMREAD_UNCHANGED)
        if template is None:
            return None
        # Convert to grayscale for template matching
        if len(template.shape) == 3:
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        return template
    except Exception as e:
        logging.getLogger(__name__).warning(f"Failed to load gem template: {e}")
        return None


def load_template(config: Config, filename: str, label: str) -> Optional[np.ndarray]:
    """Load a grayscale template image for detection."""
    path = config.base_dir / filename
    if not path.exists():
        logging.getLogger(__name__).debug(f"{label} template not found: {path}")
        return None

    try:
        template = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if template is None:
            return None
        if len(template.shape) == 3:
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        return template
    except Exception as e:
        logging.getLogger(__name__).debug(f"Failed to load {label} template: {e}")
        return None


def load_claim_template(config: Config) -> Optional[np.ndarray]:
    """Load CLAIM button template image for detection."""
    return load_template(config, "claim_button.png", "CLAIM")


def load_battle_template(config: Config) -> Optional[np.ndarray]:
    """Load BATTLE button template image for detection."""
    return load_template(config, "battle.png", "BATTLE")


# Gems orbit a fixed centre point.  The tip of the gem always points toward that
# centre, and all gems orbit at the same radius.
_GEM_ORBIT_CENTER_FX: float = 0.5989   # fractional x of orbit centre
_GEM_ORBIT_CENTER_FY: float = 0.2630   # fractional y of orbit centre
_GEM_ORBIT_RADIUS_FW: float = 0.2956   # orbital radius as a fraction of image width
_GEM_SEARCH_MARGIN_PX: int  = 100      # extra pixels beyond orbit radius to search


def detect_floating_gem(
    img: Optional[Image.Image],
    gem_template: Optional[np.ndarray],
    config: Config,
) -> Optional[Tuple[int, int, float]]:
    """Detect the floating gem that orbits _GEM_ORBIT_CENTER_*.

    Searches a bounding box centred on the orbit centre (radius + margin) at all
    template rotations in 10-degree steps.

    Returns (x, y, angle_from_north_deg) where *angle_from_north_deg* is the
    clockwise angle of the vector (orbit_centre → gem) measured from straight
    up (0 = gem directly above centre, 90 = gem to the right, etc.).  Returns
    None when no gem is found above the confidence threshold.
    """
    if img is None or gem_template is None:
        return None

    log = logging.getLogger(__name__)

    try:
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        h, w = img_cv.shape
        th, tw = gem_template.shape[:2]

        # Orbit centre in pixel coordinates
        cx = _GEM_ORBIT_CENTER_FX * w
        cy = _GEM_ORBIT_CENTER_FY * h

        # Square search region centred on the orbit centre
        search_r = int(_GEM_ORBIT_RADIUS_FW * w) + _GEM_SEARCH_MARGIN_PX
        sx1 = max(tw // 2, int(cx - search_r))
        sx2 = min(w - tw // 2, int(cx + search_r))
        sy1 = max(th // 2, int(cy - search_r))
        sy2 = min(h - th // 2, int(cy + search_r))
        search_region = img_cv[sy1:sy2, sx1:sx2]

        if search_region.shape[0] < th or search_region.shape[1] < tw:
            return None

        best_val = 0.0
        best_match_cx: Optional[int] = None
        best_match_cy: Optional[int] = None
        threshold = 0.70

        center_pt = (tw // 2, th // 2)
        for rot_deg in range(0, 360, 10):
            matrix = cv2.getRotationMatrix2D(center_pt, rot_deg, 1.0)
            rotated = cv2.warpAffine(gem_template, matrix, (tw, th))

            result = cv2.matchTemplate(search_region, rotated, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if max_val > best_val:
                best_val = max_val
                best_match_cx = sx1 + max_loc[0] + tw // 2
                best_match_cy = sy1 + max_loc[1] + th // 2

        if best_val >= threshold and best_match_cx is not None:
            dx = best_match_cx - cx
            dy = best_match_cy - cy
            # Clockwise angle from north: 0° = above centre, 90° = right, etc.
            angle_from_north = math.degrees(math.atan2(dx, -dy))
            log.info(
                f"Floating gem at ({best_match_cx}, {best_match_cy}) "
                f"angle={angle_from_north:.1f}° confidence={best_val:.3f}"
            )
            return (int(best_match_cx), int(best_match_cy), angle_from_north)

        return None

    except Exception as e:
        log.warning(f"Gem detection failed: {e}")
        return None


def process_claim_button(img: Optional[Image.Image], claim_template: Optional[np.ndarray],
                        config: Config) -> None:
    """Detect CLAIM button in middle-left region and click it if found.

    This supplements OCR-based detection for improved reliability without OCR cost.
    Focuses on middle-left area where CLAIM buttons typically appear.
    """
    global ctx
    if img is None or claim_template is None:
        return

    log = logging.getLogger(__name__)

    try:
        # Convert PIL to cv2
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        h, w = img_cv.shape

        # Define middle-left region (left 40%, middle 60% vertically)
        # This focuses on where CLAIM buttons commonly appear
        x_start = int(w * 0.05)
        x_end = int(w * 0.45)
        y_start = int(h * 0.2)
        y_end = int(h * 0.8)
        search_region = img_cv[y_start:y_end, x_start:x_end]

        # Skip if template is larger than search region
        if claim_template.shape[0] > search_region.shape[0] or claim_template.shape[1] > search_region.shape[1]:
            return

        # Template matching with normalized cross-correlation
        result = cv2.matchTemplate(search_region, claim_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        threshold = 0.65  # Slightly higher threshold for button matching

        if max_val >= threshold:
            # Convert back to full image coordinates
            claim_pos = (
                max_loc[0] + x_start + claim_template.shape[1] // 2,
                max_loc[1] + y_start + claim_template.shape[0] // 2,
            )
            log.info(f"CLAIM button detected at {claim_pos} with confidence {max_val:.2f}")
            # Add cooldown to avoid clicking too frequently
            last_claim_time = get_last_action_time(ctx.game_state, ActionType.CLICK)
            if time.time() - last_claim_time > 2.0:
                log.info(f"CLAIM button detected via template at {claim_pos} - clicking!")
                execute_click(claim_pos[0], claim_pos[1], ctx.window_rect, config, reason="claim")
                # Update action history
                action = Action(
                    action_type=ActionType.CLICK,
                    x=claim_pos[0],
                    y=claim_pos[1],
                    reason="Clicking CLAIM button (template match)",
                    priority=90
                )
                ctx.game_state = record_action_in_state(ctx.game_state, action)

    except Exception as e:
        log.debug(f"CLAIM button detection failed: {e}")


def process_battle_button(img: Optional[Image.Image]) -> None:
    """Detect BATTLE button via template matching and click it if found."""
    global ctx
    log = logging.getLogger(__name__)

    battle_button_pos = None
    if ctx.battle_template is not None:
        battle_button_pos = detect_template_in_region(img, ctx.battle_template, "BATTLE button", 0.28, 0.71, 0.88, 0.91, threshold=0.98)
    else:
        log.error('no battle button template')
    if battle_button_pos:
        log.info(f"BATTLE button detected via template at {battle_button_pos} - clicking to start game!")
        do_click("Clicking BATTLE button", battle_button_pos[0], battle_button_pos[1])
        mark_battle_start()


def detect_template_in_region(
    img: Optional[Image.Image],
    template: Optional[np.ndarray],
    label: str,
    x0: float, y0: float, x1: float, y1: float,
    threshold: float,
) -> Optional[Tuple[float, float]]:
    """Match *template* inside the normalised sub-region [x0,x1] x [y0,y1] of *img*.

    Returns the normalised (fx, fy) centre of the best match when its score
    meets *threshold*, otherwise None.
    """
    if img is None or template is None:
        return None

    log = logging.getLogger(__name__)

    try:
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        h, w = img_cv.shape

        x_start, x_end = int(w * x0), int(w * x1)
        y_start, y_end = int(h * y0), int(h * y1)
        search_region = img_cv[y_start:y_end, x_start:x_end]

        if template.shape[0] > search_region.shape[0] or template.shape[1] > search_region.shape[1]:
            scale = min(search_region.shape[0] / template.shape[0],
                        search_region.shape[1] / template.shape[1])
            new_w = max(1, int(template.shape[1] * scale))
            new_h = max(1, int(template.shape[0] * scale))
            template = cv2.resize(template, (new_w, new_h), interpolation=cv2.INTER_AREA)

        result = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        log.debug('%s template check: max_val=%.3f at location %s', label, max_val, max_loc)
        if max_val >= threshold:
            match_x = max_loc[0] + x_start + template.shape[1] // 2
            match_y = max_loc[1] + y_start + template.shape[0] // 2
            fx, fy = match_x / w, match_y / h
            log.debug(f"{label} detected at ({fx:.4f}, {fy:.4f}) with confidence {max_val:.2f}")
            return (fx, fy)

        return None

    except Exception as e:
        log.debug(f"{label} detection failed: {e}")
        return None



def update_tier_from_frame(frame: OCRFrame) -> None:
    """Detect the tier number from OCR results and update ctx.game_state.tier.

    Searches for 'Tier N' in OCR output, either as a single result or as
    'Tier' followed by a nearby digit result.  When no tier is found the
    state is left unchanged.
    """
    global ctx
    log = logging.getLogger(__name__)

    tier_num: Optional[int] = None

    for r in frame.results:
        # Single token: "Tier 4", "Tier 5", etc.
        m = re.fullmatch(r'Tier\s+(\d+)', r.text, re.IGNORECASE)
        if m:
            tier_num = int(m.group(1))
            break

        # Two adjacent tokens: "Tier" then a standalone digit nearby
        if r.text.lower() == 'tier':
            for other in frame.results:
                if other is r:
                    continue
                if re.fullmatch(r'\d+', other.text):
                    if abs(other.fx - r.fx) < 0.12 and abs(other.fy - r.fy) < 0.05:
                        tier_num = int(other.text)
                        break
            if tier_num is not None:
                break

    if tier_num is not None:
        log.info("Tier detected: %d", tier_num)
        ctx.game_state = replace(ctx.game_state, tier=tier_num)
    else:
        log.info("No tier found in frame; state unchanged")


def get_last_action_time(state: GameState, action_type: ActionType) -> float:
    """Get timestamp of last action of given type. Returns 0 if not found."""
    for entry in reversed(state.action_history):
        if entry.get("type") == action_type.name:
            return entry.get("time", 0.0)
    return 0.0


def find_claimable_buttons(state: GameState) -> List[UIElement]:
    """Find all claim/collect buttons on current screen."""
    log = logging.getLogger(__name__)
    claimable = []
    
    # Log all buttons found for debugging
    all_buttons = [elem for elem in state.current_screen.elements if elem.element_type == "button"]
    if all_buttons:
        log.debug(f"Found {len(all_buttons)} buttons: {[b.text for b in all_buttons]}")
    
    for elem in state.current_screen.elements:
        if elem.element_type == "button":
            text_lower = elem.text.lower()
            # Match claim, collect, gather, etc.
            if any(word in text_lower for word in ["claim", "collect", "gather"]):
                claimable.append(elem)
                log.info('Claimable button found: "%s" at (%d, %d)', elem.text, elem.center[0], elem.center[1])
    
    if not claimable and all_buttons:
        log.debug('No claimable buttons found among: %s', [b.text for b in all_buttons])
    


# ============================================================================
# UPGRADE AUTOMATION
# ============================================================================

FREE_UPGRADE_CATEGORIES = ['ATTACK', 'DEFENSE', 'UTILITY']

CATEGORY_UPGRADES: Dict[str, set] = {
    'ATTACK': set(ATTACK_UPGRADES),
    'DEFENSE': set(DEFENSE_UPGRADES),
    'UTILITY': set(UTILITY_UPGRADES),
}

# Fractional (fx, fy) coordinates for clicking each upgrade sub-tab
UPGRADE_TAB_CLICK = {
    'ATTACK':  (0.321,  0.985),
    'DEFENSE': (0.5105, 0.985),
    'UTILITY': (0.7,    0.985),
}


def _click_upgrade_tab(want_page: str) -> None:
    """Click the upgrade sub-tab for the given category."""
    if want_page in UPGRADE_TAB_CLICK:
        fx, fy = UPGRADE_TAB_CLICK[want_page]
        do_click(f"Switching to {want_page} upgrade tab", fx, fy)


def _do_upgrade_scroll(direction: str, w: int, h: int) -> None:
    """Drag the upgrade list up or down by ~200 px."""
    global ctx
    log = logging.getLogger(__name__)
    if not ctx.window_rect:
        return
    scroll_x = int(0.595 * w)
    scroll_y = int(0.8325 * h)
    execute_swipe(scroll_x, scroll_y, 100, direction, ctx.window_rect, ctx.config, True)


def _advance_upgrade_state() -> None:
    """Move to the next item in UPGRADE_PRIORITY and reset scroll tracking."""
    global ctx
    log = logging.getLogger(__name__)
    ctx.upgrade_state += 1
    ctx.upgrade_scroll_start = 0.0
    ctx.upgrade_scroll_direction = 'down'
    if ctx.upgrade_state < len(UPGRADE_PRIORITY):
        log.info(f"Advanced to upgrade state {ctx.upgrade_state}: "
                 f"'{UPGRADE_PRIORITY[ctx.upgrade_state][1]}'")
    else:
        log.info("All priority upgrades complete - entering free upgrade mode")


def check_wave_restart(game_state: GameState) -> bool:
    """Return True if the wave number has dropped sharply, indicating a game restart."""
    history = game_state.wave_history
    if len(history) < 4:
        return False
    latest_wave = history[-1][0]
    # Peak of the 4 entries before the latest one
    recent_peak = max(w for w, _ in history[-5:-1])
    return latest_wave < recent_peak - 100


def handle_upgrade_action(seen_page: Optional[str],
                          upgrade_buttons: Dict[str, Any],
                          w: int, h: int) -> None:
    """Timed upgrade purchasing driven by UPGRADE_PRIORITY.

    Called every tick; only acts when upgrade_interval seconds have elapsed.
    Steps:
      1. If upgrade_state is past the end of UPGRADE_PRIORITY, hand off to
         handle_free_upgrade_mode.
      2. Ensure the correct upgrade sub-tab (ATTACK/DEFENSE/UTILITY) is open.
      3. Look for the target upgrade label among visible buttons.
         - Not found → scroll (down first, switch to up after upgrade_scroll_timeout).
         - Found, cost is None (MAX) → advance to next priority item.
         - Found, cost exceeds threshold → advance to next priority item.
         - Found, affordable → click to purchase.
    """
    global ctx
    log = logging.getLogger(__name__)
    now = time.time()

    if now - ctx.last_upgrade_action < ctx.config.upgrade_interval:
        return

    if ctx.upgrade_state >= len(UPGRADE_PRIORITY):
        handle_free_upgrade_mode(seen_page, upgrade_buttons)
        return

    want_page, want_label, cost_threshold, needs_scroll = UPGRADE_PRIORITY[ctx.upgrade_state]
    log.info(f"Upgrade state {ctx.upgrade_state}: targeting '{want_label}' on {want_page} tab "
             f"(threshold={cost_threshold}, needs_scroll={needs_scroll})")

    if seen_page is None:
        log.info("No upgrade tab visible - skipping upgrade action this tick")
        return

    if seen_page != want_page:
        log.info(f"Currently on '{seen_page}', need '{want_page}' - switching tab")
        _click_upgrade_tab(want_page)
        ctx.last_upgrade_action = now
        return

    # Search for the target among currently visible buttons
    target_info: Optional[Any] = None
    for label, info in upgrade_buttons.items():
        if label.lower() == want_label.lower():
            target_info = info
            break

    if target_info is None:
        # Not visible - scroll to find it
        if ctx.upgrade_scroll_start == 0.0:
            ctx.upgrade_scroll_start = now
            ctx.upgrade_scroll_direction = 'down'
            log.info(f"'{want_label}' not on screen - beginning scroll down")

        elapsed = now - ctx.upgrade_scroll_start
        if ctx.upgrade_scroll_direction == 'down' and elapsed > ctx.config.upgrade_scroll_timeout:
            log.info(f"Scroll down timed out after {elapsed:.0f}s - switching to scroll up")
            ctx.upgrade_scroll_direction = 'up'
            ctx.upgrade_scroll_start = now

        _do_upgrade_scroll(ctx.upgrade_scroll_direction, w, h)
        ctx.last_upgrade_action = now
        return

    # Found - reset scroll tracking
    ctx.upgrade_scroll_start = 0.0
    ctx.upgrade_scroll_direction = 'down'

    # Maxed?
    if target_info['cost'] is None:
        log.info(f"'{want_label}' is maxed - advancing to next priority upgrade")
        _advance_upgrade_state()
        ctx.last_upgrade_action = now
        return

    # Exceeds cost threshold?
    if cost_threshold is not None and target_info['cost'] > cost_threshold:
        log.info(f"'{want_label}' cost {target_info['cost']:.0f} > threshold {cost_threshold:.0f} "
                 f"- advancing to next priority upgrade")
        _advance_upgrade_state()
        ctx.last_upgrade_action = now
        return

    # Purchase
    fx, fy = target_info['button_position']
    fx = fx +  0.1
    do_click(f"Buying upgrade '{want_label}'", fx, fy)
    ctx.last_upgrade_action = now


def handle_free_upgrade_mode(seen_page: Optional[str],
                             upgrade_buttons: Dict[str, Any]) -> None:
    """Purchase any available (non-maxed) upgrade, cycling ATTACK→DEFENSE→UTILITY every 2 min."""
    global ctx
    log = logging.getLogger(__name__)
    now = time.time()

    if now - ctx.last_upgrade_action < ctx.config.upgrade_interval:
        return

    # Initialise cycle timer on first entry into this mode
    if ctx.free_upgrade_cycle_start == 0.0:
        ctx.free_upgrade_cycle_start = now

    # Rotate category after free_upgrade_cycle_seconds
    if now - ctx.free_upgrade_cycle_start >= ctx.config.free_upgrade_cycle_seconds:
        ctx.free_upgrade_category = (ctx.free_upgrade_category + 1) % len(FREE_UPGRADE_CATEGORIES)
        ctx.free_upgrade_cycle_start = now
        log.info(f"Free upgrade cycle: now targeting "
                 f"{FREE_UPGRADE_CATEGORIES[ctx.free_upgrade_category]}")

    want_page = FREE_UPGRADE_CATEGORIES[ctx.free_upgrade_category]
    log.info(f"Free upgrade mode: looking for any available {want_page} upgrade")

    if seen_page is None:
        return

    if seen_page != want_page:
        log.info(f"Free upgrade: switching from {seen_page} to {want_page}")
        _click_upgrade_tab(want_page)
        ctx.last_upgrade_action = now
        return

    category_labels = CATEGORY_UPGRADES[want_page]
    for label, info in upgrade_buttons.items():
        if label in category_labels and info['cost'] is not None:
            fx, fy = info['button_position']
            log.info(f"Free upgrade: buying '{label}' (cost={info['cost']}) "
                     f"at ({fx:.3f}, {fy:.3f})")
            do_click(f"Free upgrade: '{label}'", fx, fy)
            ctx.last_upgrade_action = now
            return

    log.info(f"Free upgrade: no available {want_page} upgrades visible this tick")
    ctx.last_upgrade_action = now


# ============================================================================
# SIDE-EFFECT FUNCTIONS - INPUT
# ============================================================================

def to_absolute_coords(rel_x: int, rel_y: int, rect: WindowRect) -> Tuple[int, int]:
    """Convert relative to absolute coordinates."""
    return (rect.left + rel_x, rect.top + rel_y)


def cleanup_old_click_debug_files(config: Config, keep_count: int = 20) -> None:
    """Keep only the last N click_debug files, delete older ones."""
    log = logging.getLogger(__name__)
    try:
        # Find all click_debug files
        pattern = config.debug_dir / "click_debug_*.png"
        files = sorted(config.debug_dir.glob("click_debug_*.png"), key=lambda p: p.stat().st_mtime)
        
        # Delete older files if we have more than keep_count
        if len(files) > keep_count:
            to_delete = files[:-keep_count]
            for f in to_delete:
                f.unlink()
            log.debug(f"Cleaned up {len(to_delete)} old click_debug files")
    except Exception as e:
        log.warning(f"Failed to cleanup old click_debug files: {e}")


def execute_click(x: int, y: int, rect: WindowRect, config: Config, bring_to_front: bool = False, reason: str = "") -> bool:
    """Execute click action. Side effect."""
    log = logging.getLogger(__name__)
    
    # Capture debug screenshot with crosshairs BEFORE clicking
    try:
        debug_img = capture_window(rect)
        if debug_img:
            draw = ImageDraw.Draw(debug_img)
            # Draw crosshairs at click location
            crosshair_size = 40
            crosshair_color = (0, 255, 0)  # Green
            crosshair_width = 3
            
            # Horizontal line
            draw.line([(x - crosshair_size, y), (x + crosshair_size, y)], 
                     fill=crosshair_color, width=crosshair_width)
            # Vertical line
            draw.line([(x, y - crosshair_size), (x, y + crosshair_size)], 
                     fill=crosshair_color, width=crosshair_width)
            # Center circle
            draw.ellipse([(x-5, y-5), (x+5, y+5)], outline=crosshair_color, width=crosshair_width)
            
            # Add text annotation
            draw.text((x + crosshair_size + 10, y), 
                     f"Click: ({x},{y})\nAbs: ({rect.left + x},{rect.top + y})",
                     fill=crosshair_color)
            
            # Save debug image
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            reason_slug = ("_" + re.sub(r"[^\w]+", "_", reason).strip("_")) if reason else ""
            debug_path = config.debug_dir / f"click_debug_{timestamp}{reason_slug}.png"
            debug_img.save(debug_path)
            log.info(f"Debug screenshot saved: {debug_path}")
            
            # Cleanup old files - keep only last 20
            cleanup_old_click_debug_files(config, keep_count=20)
    except Exception as e:
        log.warning(f"Could not create debug screenshot: {e}")
    
    # Bring window to foreground before clicking
    assert not bring_to_front, "Bringing to front is currently disabled to avoid issues. Set bring_to_front=True to enable (use with caution)."
    if bring_to_front:
        if rect.hwnd and win32gui:
            try:
                win32gui.SetForegroundWindow(rect.hwnd)
                time.sleep(0.1)  # Brief delay for window to come to front
                log.debug(f"Brought window to foreground (hwnd={rect.hwnd})")
            except Exception as e:
                log.warning(f"Could not bring window to foreground: {e}")
        
    ax, ay = to_absolute_coords(x, y, rect)
    log.debug(f"click({x}, {y}) → abs({ax}, {ay})")
    pyautogui.click(ax, ay, interval=config.click_pause)
    log.info(f"Clicked at ({ax}, {ay}); sleep for {config.click_pause:.2f} seconds" + (f"  [{reason}]" if reason else ""))
    time.sleep(config.click_pause)

    return True


def execute_swipe(x: int, y: int, distance: int, direction: str,
                 rect: WindowRect, config: Config, enabled: bool) -> bool:
    """Execute swipe action. Side effect."""
    if not enabled:
        return False
    
    log = logging.getLogger(__name__)
    ax, ay = to_absolute_coords(x, y, rect)
    offset = distance if direction == "up" else -distance
    
    log.info(f"swipe_{direction}({x}, {y}, {distance})")
    pyautogui.moveTo(ax, ay)
    pyautogui.drag(0, offset, duration=0.3)
    log.info(f"Swiped {direction} from ({ax}, {ay}) by {offset} pixels")
    log.info('sleep for %.2f seconds', config.click_pause)
    time.sleep(config.swipe_pause)
    return True


def execute_action(action: Action, rect: Optional[WindowRect],
                  config: Config) -> None:
    """Execute an action. Side effect dispatcher."""
    log = logging.getLogger(__name__)
    log.info(f"Action: {action.action_type.name} – {action.reason}")

    if action.action_type == ActionType.CLICK and rect:
        log.info('clicking at (%d, %d) relative to window', action.x, action.y)
        execute_click(action.x, action.y, rect, config, reason=action.reason)

    elif action.action_type == ActionType.SWIPE_UP and rect:
        execute_swipe(action.x, action.y, action.amount or 300,
                     "up", rect, config)

    elif action.action_type == ActionType.SWIPE_DOWN and rect:
        execute_swipe(action.x, action.y, action.amount or 300,
                     "down", rect, config)

    elif action.action_type == ActionType.WAIT:
        log.trace('sleeping for %.2f seconds', action.duration)
        time.sleep(action.duration)


# ============================================================================
# STATEFUL RUNTIME CONTEXT
# ============================================================================

@dataclass
class RuntimeContext:
    """Container for mutable runtime state."""
    config: Config
    ocr_reader: Any = None
    gem_template: Optional[np.ndarray] = None
    claim_template: Optional[np.ndarray] = None
    battle_template: Optional[np.ndarray] = None
    newperk_template: Optional[np.ndarray] = None
    game_state: GameState = field(default_factory=GameState)
    window_rect: Optional[WindowRect] = None
    latest_image: Optional[Image.Image] = None
    frame: Optional[OCRFrame] = None
    running: bool = False
    input_enabled: bool = False
    status: str = "stopped"
    last_window_check: float = 0.0
    last_seen_upgrades: float = 0.0
    upgrade_state: int = 0
    last_upgrade_action: float = 0.0        # timestamp of last upgrade purchase attempt
    last_seen_upgrades: float = 0.0  # timestamp when upgrade buttons were last visible
    recover_stage: int = 0
    upgrade_scroll_start: float = 0.0       # when current scroll direction started (0 = not scrolling)
    upgrade_scroll_direction: str = 'down'  # 'down' or 'up'
    free_upgrade_category: int = 0          # index into FREE_UPGRADE_CATEGORIES
    free_upgrade_cycle_start: float = 0.0   # when the current free-upgrade category started
    no_perk_until: float = 0.0              # skip perk processing until this timestamp (set when no perks selectable)

    def update_window(self):
        """Update window rect if needed."""
        if time.time() - self.last_window_check > 2.0:
            self.window_rect = find_window(self.config.window_title)
            log = logging.getLogger(__name__)
            log.debug('Window check: %s', self.window_rect)
            self.last_window_check = time.time()

def mark_battle_start():
    global ctx
    log = logging.getLogger(__name__)
    log.info("Battle start detected - marking in state")
    ctx.game_state = replace(ctx.game_state, battle_start_time=time.time())
    ctx.upgrade_state = 0
    ctx.upgrade_scroll_start = 0.0
    ctx.upgrade_scroll_direction = 'down'
    ctx.free_upgrade_cycle_start = 0.0
    ctx.no_perk_until = 0.0

def click_if_present(name, condition, callback=None):
    global ctx
    log = logging.getLogger(__name__)
    marks = [r for r in ctx.frame.results if condition(r)]
    log.debug(f'Marks found for condition "{name}"   : {marks}')
    if marks:
        log.info(f"Condition met for '{marks[0].text}' at ({marks[0].fx:.4f}, {marks[0].fy:.4f}) - clicking!")
        execute_click(marks[0].center[0], marks[0].center[1], ctx.window_rect, ctx.config, reason=name)
        if callback:
            callback()

def do_ocr():
    """Capture the window and run OCR. Returns (img, img_capture_time, frame) or None on failure."""
    global ctx
    log = logging.getLogger(__name__)

    # Update window rect
    ctx.update_window()
    if not ctx.window_rect:
        log.warning("Window not found: '%s'", ctx.config.window_title)
        ctx.status = "no_window"
        return None

    # Capture screen
    log.debug(f"Capturing window: {ctx.window_rect.width} x {ctx.window_rect.height} at ({ctx.window_rect.left}, {ctx.window_rect.top})")
    img = capture_window(ctx.window_rect)
    img_capture_time = time.time()  # used for gem dead-reckoning dt
    log.debug('Capture done')
    if not img:
        log.warning("Failed to capture window")
        return None

    # time OCR
    try:
        ocr_t0 = time.time()
        frame = process_ocr(img, ctx.config, ctx.ocr_reader)
        ocr_t1 = time.time()
        log.info(f"------------------ OCR completed in {ocr_t1 - ocr_t0:.2f} seconds with {len(frame.results)} results")
        if frame:
            save_debug_files(img, frame, ctx.config)
        else:
            log.warning("OCR processing failed")
            return None
    except Exception as e:
        log.error(f"OCR failed: {e}")
        return None

    # Check known markers (inlined from check_known_markers)
    w,h= frame.image_size
    if w <= 0 or h <= 0:
        log.warning("Invalid image size from OCR frame: (%d, %d)", w, h)
        return
    return img, img_capture_time, frame


def automation_loop_tick():
    """Single tick of automation loop."""
    global ctx
    log = logging.getLogger(__name__)
    result = do_ocr()
    if result is None:
        return
    img, img_capture_time, frame = result
    w, h = frame.image_size
    ctx.latest_image = img
    ctx.frame = frame
    update_tier_from_frame(frame)

    # Check for floating gem and click it with dead reckoning
    if ctx.gem_template is not None:
        gem_pos = detect_floating_gem(img, ctx.gem_template, ctx.config)
        attempt_floating_gem_click(log, img, img_capture_time, gem_pos)
    
    # Check for CLAIM button via template matching (supplements OCR detection)
    process_claim_button(img, ctx.claim_template, ctx.config)

    # Check for BATTLE button via template matching
    process_battle_button(img)

    # Build screen state (inlined from build_screen_state)
    log.debug('build screen state')
    

    # Extract wave number for comparison
    wave_num_str, _ = extract_wave_from_frame(frame)
    wave_num = None
    if wave_num_str:
        try:
            wave_num = int(wave_num_str.replace(',', ''))
        except ValueError:
            pass
    
    texts: tuple[str, float, float] = [(r.text, r.fx, r.fy) for r in frame.results]
    perks_mode = [t for t in texts if t[0] == 'Perks' and abs(t[1]-0.612) < 0.1 and abs(t[2]-0.098) < 0.1]
    choose = [t for t in texts if t[0] == 'Choose' and abs(t[1]-0.522) < 0.05 and abs(t[2]-0.206) < 0.05]
    log.debug('perks_mode: %s, choose: %s', perks_mode, choose)
    if perks_mode and choose:
        log.info("Perks mode found")
        mode = 'perks'        
    else:
        mode = 'main'
    log.info(f"Detected mode: {mode}")
    perk_text = {}
    click_if_present('claim', lambda r: r.text.lower() == "claim" and (r.is_near(0.6056, 0.035, 0.2) or r.is_near(  0.2224,   0.9882) or r.is_near(  0.3130,   0.8281)))
    click_if_present('battle', lambda r: r.text == 'BATTLE' and r.is_near(  0.5951,   0.8168))
    click_if_present('home', lambda r: r.text == 'HOME' and r.is_near(  0.7644,   0.7429))
    game_stats_mark = [r for r in frame.results if r.text == 'GAME' and r.is_near(  0.5171,   0.2491) or r.text == 'STATS' and r.is_near(  0.6667,   0.2491)]
    defense_marks = [r for r in frame.results if r.text.lower() == "defense" and r.is_near(0.343, 0.632, 0.1)]
    attack_marks = [r for r in frame.results if r.text.lower() == "attack" and r.is_near(0.329, 0.632, 0.1)]
    utility_marks = [r for r in frame.results if r.text.lower() == "utility" and r.is_near(0.333, 0.632, 0.1)]
    if game_stats_mark:
        mode = 'killed by'
        do_click("Seen game stats, clicking to exit", 0.7601, 0.7496)
    seen = None  # which upgrade sub-tab is currently visible; set inside if mode == 'main'
    if mode == 'main':
        seen = 'DEFENSE' if defense_marks else 'ATTACK' if attack_marks else 'UTILITY' if utility_marks else None
        log.info(f'Seen upgrade mode selector: {seen} at {ctx.upgrade_state} {UPGRADE_PRIORITY[ctx.upgrade_state]}')
        want_upgrades = UPGRADE_PRIORITY[ctx.upgrade_state][0] if ctx.upgrade_state < len(UPGRADE_PRIORITY) else None   

        if seen:
            ctx.last_seen_upgrades = time.time()
        else:
            delay = time.time() - ctx.last_seen_upgrades
            if delay > 15.0:
                pos = (0.5105, 0.985) if want_upgrades == 'DEFENSE' else (0.3232, 0.9711) if want_upgrades == 'ATTACK' else (0.7000, 0.9719)
                do_click(f"Seen no upgrades for {delay} so clicking upgrades tab for {want_upgrades}", pos[0], pos[1])
            
        newperk_pos = detect_template_in_region(img, ctx.newperk_template, "new perk icon", 0.42, 0.00, 0.8, 0.10, threshold=0.9)
        if newperk_pos:
            do_click("Clicking new perk icon (template match)", newperk_pos[0], newperk_pos[1])
            mode = 'perks'
    for r in frame.results:
        cx, cy = r.center
        
        lowertext = r.text.lower()

        # Check for perk threshold near position
        if r.is_near(0.6341, 0.0436, 0.05):
            # Try to parse as number
            text_clean = r.text.replace(',', '').strip()
            if re.match(r'^\d+$', text_clean):
                try:
                    threshold_num = int(text_clean)
                    # Check if it's similar to wave number (within reasonable range)
                    if wave_num and abs(threshold_num - wave_num) < 10000:
                        log.debug(f"Perk threshold detected: {r.text} at ({r.fx:.4f}, {r.fy:.4f}) [wave: {wave_num_str}]")
                except ValueError:
                    pass
        
        # Check for coins at position
        if r.is_near(0.3132, 0.0819, 0.02):
            # Parse number with suffix (K, M, B, T)
            match = re.match(r'^([0-9.]+)\s*([KMBT])?$', r.text.strip(), re.IGNORECASE)
            if match:
                value = float(match.group(1))
                suffix = match.group(2).upper() if match.group(2) else ''
                
                multipliers = {'K': 1_000, 'M': 1_000_000, 'B': 1_000_000_000, 'T': 1_000_000_000_000}
                multiplier = multipliers.get(suffix, 1)
                coins = value * multiplier

    # Use refactored perk detection
    if mode == 'perks' and time.time() < ctx.no_perk_until:
        log.info(f"Perk cooldown active - skipping perk check for {ctx.no_perk_until - time.time():.0f}s more, closing window")
        close_perks()
    else:
        perk_result = detect_perks(frame)
        perk_text = perk_result['perk_text']
        perk_text_join = perk_result['perk_text_join']
        perk_text_priority = perk_result['perk_text_priority']
        clean = perk_result['all_matched']
        log.debug('perk text: %s', perk_text)
        if mode == 'perks' and not perk_text:
            log.info("Perks mode active but no selectable perks detected - closing window and suppressing perk checks for 1200s")
            close_perks()
            ctx.no_perk_until = time.time() + 1200.0
        elif perk_text and mode == 'perks':
            log.debug(f"Perk text by row: {perk_text_join}")

            if len(perk_text_join) not in [3,4] or not clean:
                log.warning(f"Unexpected number of meaningful perk rows detected: {len(perk_text_join)}. Expected 3 or 4. Detected rows: {list(perk_text_join.keys())}")
            # create/append to perks.log with a jsonl format containing wave, timestamp in epoch seconds, timestamp as iso string, perk_text_join, perk_text_priority
            with open(ctx.config.debug_dir / "perks.jsonl", "a") as log_file:
                log_entry = {
                    "timestamp": time.time(),
                    "timestamp_iso": datetime.datetime.now().isoformat(),
                    "wave": wave_num_str,
                    "perk_text_join": perk_text_join,
                    "perk_text_priority": perk_text_priority,
                }
                log_file.write(json.dumps(log_entry) + "\n")
            # pick the row with the highest priority (lowest index)
            if perk_text_priority:
                best_row, best_choice, best_idx = perk_text_priority[0]
                log.info(f"Best perk choice: '{best_choice}' in row {best_row} with text '{perk_text_join[best_row]}'")
                # click the perk in that row
                message = f"Clicking perk at row {best_row} (choice: '{best_choice}')"
                do_click(message, 0.671, PERK_ROWS[best_row][1])
                close_perks()

        if perks_mode and not choose:
            close_perks()
    
    # Detect and log upgrade buttons
    upgrade_buttons = detect_upgrade_buttons(frame, img, ctx.config)
    if upgrade_buttons:
        text = [ f'{k} ({v["cost"]})' for k,v in upgrade_buttons.items()]
        log.info(f'upgrades detected : {", ".join(text)}')
        ctx.last_seen_upgrades = time.time()
        ctx.recover_stage = 0
    elif mode == 'main' and time.time() - ctx.last_seen_upgrades > 30.0 and ctx.recover_stage == 0:
        log.info("No upgrade buttons visible for 30s - clicking low moddle to dismiss popup")
        do_click("Dismiss popup blocking upgrades", 0.58, 0.9)
        ctx.recover_stage = 1
    elif mode == 'main' and time.time() - ctx.last_seen_upgrades > 45.0 and ctx.recover_stage == 1:
        log.info("No upgrade buttons visible for 45s - clicking top right to dismiss wave")
        do_click('Dismiss wave info', 0.9129, 0.1411)

    # Game-restart detection: reset upgrade state when wave drops sharply
    if check_wave_restart(ctx.game_state):
        log.info("Wave drop detected - resetting upgrade state (game restarted)")
        mark_battle_start()
    # Timed upgrade purchasing (only when on main game screen with upgrades visible)
    if mode == 'main':
        handle_upgrade_action(seen, upgrade_buttons, w, h)
    
    elements = []
    resources = {}
    
    for ocr in frame.results:
        elem = classify_ocr_result(ocr)
        if elem:
            elements.append(elem)
            if elem.element_type == "resource":
                resources[elem.name] = elem.text
            elif elem.element_type == "button":
                log.debug(f"Button classified: '{elem.text}' at {elem.center}")

    screen_state = ScreenState(
        elements=tuple(elements),
        resources=resources,
        raw_texts=tuple(r.text for r in frame.results),
        timestamp=time.time()
    )
    
    # Update game state (inlined from update_game_state)
    wave, wave_pos = extract_wave_from_frame(frame)
    
    # Also check if wave was classified as a resource
    if not wave and "wave" in screen_state.resources:
        wave_text = screen_state.resources["wave"]
        # Extract just the number from resource text like "Wave 123" or "123"
        digits = re.findall(r'[0-9,]+', wave_text)
        if digits:
            wave = digits[0].replace(',', '')
    
    new_resources = {**ctx.game_state.resources, **screen_state.resources}
    
    # Track wave progress and calculate rate
    new_wave_history = ctx.game_state.wave_history
    if wave and wave != ctx.game_state.wave:
        try:
            wave_num = int(wave.replace(',', ''))
            current_time = time.time()
            new_wave_history = (*ctx.game_state.wave_history, (wave_num, current_time))
            
            # Keep only last 100 waves
            if len(new_wave_history) > 100:
                new_wave_history = new_wave_history[-100:]
            
            # Calculate waves per hour if we have enough history
            if len(new_wave_history) >= 2:
                first_wave, first_time = new_wave_history[0]
                last_wave, last_time = new_wave_history[-1]
                time_diff_hours = (last_time - first_time) / 3600.0
                
                if time_diff_hours > 0:
                    waves_diff = last_wave - first_wave
                    waves_per_hour = waves_diff / time_diff_hours
                    log.info(f"Wave progress: {wave} | Rate: {waves_per_hour:.1f} waves/hour (based on {len(new_wave_history)} samples over {time_diff_hours*60:.1f} min)")
                else:
                    log.info(f"Wave progress: {wave}")
            else:
                log.info(f"Wave progress: {wave} (collecting data...)")
        except (ValueError, TypeError):
            pass
    
    ctx.game_state = replace(
        ctx.game_state,
        current_screen=screen_state,
        resources=new_resources,
        wave=wave or ctx.game_state.wave,
        wave_pos=wave_pos or ctx.game_state.wave_pos,
        wave_history=new_wave_history
    )
    ctx.status = "running"

def attempt_floating_gem_click(log, img, img_capture_time, gem_pos):
    if gem_pos:
        gx, gy, angle_from_north = gem_pos
        img_w, img_h = img.size
        cx_px = _GEM_ORBIT_CENTER_FX * img_w
        cy_px = _GEM_ORBIT_CENTER_FY * img_h
        orbit_r_px = math.hypot(gx - cx_px, gy - cy_px)
            # Advance angle by elapsed time × 30°/s (clockwise)
        dt = time.time() - img_capture_time
        advanced_angle = angle_from_north + 30.0 * dt
        adv_rad = math.radians(advanced_angle)
        click_x = int(cx_px + orbit_r_px * math.sin(adv_rad))
        click_y = int(cy_px - orbit_r_px * math.cos(adv_rad))
        message = f"Floating gem at ({gx},{gy}) angle={angle_from_north:.1f}° dt={dt:.3f}s → clicking ({click_x},{click_y}) adv_angle={advanced_angle:.1f}°"

        log.info(message)
            # append to debug/gems.jsonl
        with (ctx.config.debug_dir / "gems.jsonl").open("a") as f:
            json.dump({
                    "timestamp": time.time(),
                    'localtime': datetime.datetime.now().isoformat(),
                    "message": message,
                    "gem_position": {"x": gx, "y": gy},
                    "click_position": {"x": click_x, "y": click_y},
                    "angle_from_north": angle_from_north,
                    "advanced_angle": advanced_angle,
                    "dt": dt
                }, f)
            f.write("\n")
        execute_click(click_x, click_y, ctx.window_rect, ctx.config, reason="floating gem")
            # Post-click verification: re-capture and re-detect
        time.sleep(0.3)
        post_img = capture_window(ctx.window_rect)
        if post_img is not None:
            post_pos = detect_floating_gem(post_img, ctx.gem_template, ctx.config)
            if post_pos is None:
                log.info("Gem click HIT - gem no longer detected")
            else:
                log.info(
                        f"Gem click MISS - gem still at ({post_pos[0]},{post_pos[1]}) "
                        f"angle={post_pos[2]:.1f}°"
                    )
            with (ctx.config.debug_dir / "gems.jsonl").open("a") as f:
                json.dump({
                        "timestamp": time.time(),
                        'localtime': datetime.datetime.now().isoformat(),
                        "post_click_verification": True,
                        "gem_still_present": post_pos is not None,
                        "post_gem_position": {"x": post_pos[0], "y": post_pos[1]} if post_pos else None,
                        "post_gem_angle": post_pos[2] if post_pos else None
                    }, f)
                f.write("\n")


def automation_loop_run(ctx: RuntimeContext):
    """Main loop runner."""
    log = logging.getLogger(__name__)
    log.info("Automation loop started")

    while ctx.running:
        t0 = time.time()
        try:
            automation_loop_tick()
        except Exception as exc:
            log.error(f"Loop tick error: {exc}", exc_info=True)
            ctx.game_state = replace(ctx.game_state,
                                    error_count=ctx.game_state.error_count + 1)
            log.info('sleeping for 2 seconds after error')
            time.sleep(2)

        elapsed = time.time() - t0
        sleep_time = max(0, ctx.config.loop_tick - elapsed)
        log.debug('sleeping for %.2f seconds', sleep_time)
        time.sleep(sleep_time)

    log.info("Automation loop stopped")


# ============================================================================
# INITIALIZATION
# ============================================================================

def initialize_ocr_backend(config: Config):
    """Initialize OCR backend. Returns reader object or None."""
    log = logging.getLogger(__name__)

    if config.ocr_engine == "easyocr" and easyocr:
        log.info("Initializing EasyOCR...")
        return easyocr.Reader([config.ocr_lang], gpu=False)
    elif pytesseract:
        pytesseract.pytesseract.tesseract_cmd = config.tesseract_cmd
        log.info(f"Using pytesseract: {config.tesseract_cmd}")
        return None
    else:
        log.error("No OCR backend available")
        return None


def setup_logging():
    """Configure logging."""
    # Create logs directory
    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"towercontrol_{timestamp}.log"
    
    # Create formatters
    formatter = logging.Formatter(
        "%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S"
    )
    
    # Setup root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)  # Log more detail to file
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logging.getLogger(__name__).info(f"Logging to file: {log_file}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="TowerControl – Functional Edition")
    parser.add_argument("--window-title", default="BlueStacks",
                       help="Window title pattern (default: BlueStacks)")
    parser.add_argument("--ocr", default="pytesseract",
                       choices=["pytesseract", "easyocr"],
                       help="OCR backend (default: pytesseract)")
    parser.add_argument("--lang", default="eng",
                       help="OCR language (default: eng)")
    return parser.parse_args()


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point."""
    global ctx
    setup_logging()
    log = logging.getLogger(__name__)

    args = parse_args()
    
    config = Config(
        window_title=args.window_title,
        ocr_engine=args.ocr,
        ocr_lang=args.lang,
    )

    log.info(f"TowerControl starting (ocr={config.ocr_engine})")

    # Initialize OCR
    ocr_reader = initialize_ocr_backend(config)

    # Load gem template for floating gem detection
    gem_template = load_gem_template(config)
    if gem_template is not None:
        log.info(f"Loaded gem template: {gem_template.shape}")
    else:
        log.warning("Gem template not loaded - floating gem detection disabled")

    # Load CLAIM button template for improved recognition
    claim_template = load_claim_template(config)
    if claim_template is not None:
        log.info(f"Loaded CLAIM template: {claim_template.shape}")
    else:
        log.info("CLAIM template not found - using OCR-only detection")

    # Load BATTLE button template
    battle_template = load_battle_template(config)
    if battle_template is not None:
        log.info(f"Loaded BATTLE template: {battle_template.shape}")
    else:
        log.info("BATTLE template not found - battle button detection disabled")

    # Load new perk template
    newperk_template = load_template(config, "newperk.png", "newperk")
    if newperk_template is not None:
        log.info(f"Loaded newperk template: {newperk_template.shape}")
    else:
        log.info("newperk template not found - image-based perk detection disabled")

    # Create runtime context
    ctx = RuntimeContext(
        config=config,
        ocr_reader=ocr_reader,
        gem_template=gem_template,
        claim_template=claim_template,
        battle_template=battle_template,
        newperk_template=newperk_template,
        last_upgrade_action=time.time(),
        running=True,
        status="running",
        input_enabled=False,
    )
    ctx.last_seen_upgrades = time.time()

    # Run automation loop on main thread
    try:
        automation_loop_run(ctx)
    except KeyboardInterrupt:
        log.info("Interrupted by user")
        ctx.running = False


if __name__ == "__main__":
    # Set PyAutoGUI safety
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.02
    
    main()
