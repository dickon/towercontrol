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
import logging
import re
import time
from dataclasses import dataclass, field, replace
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, NamedTuple

import cv2
import numpy as np
from PIL import Image, ImageDraw

# Third-party dependencies
import mss
import pyautogui

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
    click_pause: float = 0.08
    input_delay: float = 0.15
    loop_tick: float = 20.0
    full_scan_interval: float = 30.0
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
    SCAN_CURRENT = auto()
    FULL_SCAN = auto()
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

    @property
    def center(self) -> Tuple[int, int]:
        x, y, w, h = self.bbox
        return (x + w // 2, y + h // 2)


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
    last_full_scan: float = 0.0
    action_history: Tuple[Dict[str, Any], ...] = ()
    error_count: int = 0


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
        
        # Save image
        img.save(img_path)
        log.debug(f"Saved debug image: {img_path}")
        
        # Create text table
        img_width, img_height = frame.image_size
        
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"Debug Capture: {timestamp}\n")
            f.write(f"Image Size: {img_width} x {img_height}\n")
            f.write(f"Total OCR Elements: {len(frame.results)}\n")
            f.write("\n" + "="*100 + "\n")
            f.write(f"{'Text':<30} | {'X_Frac':<8} | {'Y_Frac':<8} | {'W_Frac':<8} | {'H_Frac':<8} | {'Conf':<6}\n")
            f.write("="*100 + "\n")
            
            for result in frame.results:
                x, y, w, h = result.bbox
                x_frac = x / img_width if img_width > 0 else 0.0
                y_frac = y / img_height if img_height > 0 else 0.0
                w_frac = w / img_width if img_width > 0 else 0.0
                h_frac = h / img_height if img_height > 0 else 0.0
                
                text_display = result.text[:28] + ".." if len(result.text) > 30 else result.text
                f.write(f"{text_display:<30} | {x_frac:>8.4f} | {y_frac:>8.4f} | {w_frac:>8.4f} | {h_frac:>8.4f} | {result.confidence:>6.1f}\n")
        
        log.debug(f"Saved debug text: {txt_path}")
    except Exception as e:
        log.error(f"Failed to save debug files: {e}")


# ============================================================================
# PURE FUNCTIONS - OCR
# ============================================================================

def preprocess_image(img: np.ndarray) -> List[np.ndarray]:
    """Generate multiple preprocessed variants for OCR. Returns list of images to try."""
    # Always upscale 2x - game text is small and tesseract needs it
    h, w = img.shape[:2]
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    variants = []

    # Strategy 1: OTSU thresholding - good for bimodal histograms
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(otsu)

    # Strategy 2: Adaptive threshold - good for uneven lighting
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 10
    )
    variants.append(adaptive)

    # Strategy 3: Inverted OTSU - catches light text on dark backgrounds
    _, otsu_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    variants.append(otsu_inv)

    # Strategy 4: Sharpened grayscale (no threshold) - sometimes raw works best
    sharp_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(gray, -1, sharp_kernel)
    variants.append(sharpened)

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
    """Run pytesseract OCR with multiple PSM modes. Returns list of OCR results."""
    if not pytesseract:
        return []

    pytesseract.pytesseract.tesseract_cmd = config.tesseract_cmd
    all_results = []

    # Try multiple page segmentation modes
    for psm in [6, 11, 3]:  # 6=uniform block, 11=sparse, 3=auto
        try:
            data = pytesseract.image_to_data(
                img, lang=config.ocr_lang,
                output_type=pytesseract.Output.DICT,
                config=f"--psm {psm}"
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
            all_results.append(results)
        except Exception:
            continue

    return _merge_ocr_results(all_results) if all_results else []


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


def process_ocr(img: Image.Image, config: Config, ocr_reader=None) -> OCRFrame:
    """Run OCR pipeline on image with multiple preprocessing strategies."""
    arr = image_to_bgr_array(img)
    h, w = arr.shape[:2]
    variants = preprocess_image(arr)

    if config.ocr_engine == "easyocr" and ocr_reader:
        # EasyOCR: just run on first variant
        results = run_easyocr_ocr(variants[0], ocr_reader, config)
    else:
        # Pytesseract: run on all variants and merge
        all_variant_results = []
        for variant in variants:
            variant_results = run_pytesseract_ocr(variant, config)
            all_variant_results.append(variant_results)
        results = _merge_ocr_results(all_variant_results)

    # Scale bboxes back to original image coords (we upscaled 2x)
    scaled_results = []
    for r in results:
        x, y, bw, bh = r.bbox
        scaled_results.append(OCRResult(
            text=r.text,
            bbox=(x // 2, y // 2, bw // 2, bh // 2),
            confidence=r.confidence
        ))

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

SCREEN_KEYWORDS = {
    Screen.MAIN: ["wave", "damage", "dps", "floor"],
    Screen.UPGRADES: ["upgrade", "level", "cost"],
    Screen.SHOP: ["shop", "buy", "purchase", "gems", "offer"],
    Screen.DIALOG: ["ok", "cancel", "close", "confirm", "collect"],
    Screen.IDLE_REWARDS: ["idle", "rewards", "offline", "collect"],
    Screen.SETTINGS: ["settings", "options", "sound", "music"],
    Screen.LOADING: ["loading", "connecting"],
}


def identify_screen(frame: OCRFrame) -> Screen:
    """Determine screen from OCR results. Pure function."""
    texts_lower = [r.text.lower() for r in frame.results]
    all_text = " ".join(texts_lower)

    best_screen = Screen.UNKNOWN
    best_score = 0

    for screen, keywords in SCREEN_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in all_text)
        if score > best_score:
            best_score = score
            best_screen = screen

    return best_screen


def classify_ocr_result(ocr: OCRResult, screen: Screen) -> Optional[UIElement]:
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
            cx, cy = r.center
            log.debug(f"Wave detected (strategy 1): '{wave_num}' from text '{text}' at ({cx/w:.3f}, {cy/h:.3f})")
            return wave_num, (cx / w, cy / h)

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
                cx, cy = best_candidate.center
                log.debug(f"Wave detected (strategy 2): '{best_num}' at ({cx/w:.3f}, {cy/h:.3f})")
                return best_num, (cx / w, cy / h)

    # Strategy 3: Look for any text starting with "wave" or "w" followed by digits
    for r in frame.results:
        text_lower = r.text.lower().strip()
        if text_lower.startswith(('wave', 'w')):
            # Extract any digits from the text
            digits = re.findall(r'[0-9]+', r.text)
            if digits:
                wave_num = digits[0]
                cx, cy = r.center
                log.debug(f"Wave detected (strategy 3): '{wave_num}' from text '{r.text}' at ({cx/w:.3f}, {cy/h:.3f})")
                return wave_num, (cx / w, cy / h)

    # Debug: log all OCR results if wave not found (but only occasionally to avoid spam)
    if len(frame.results) > 0:
        import random
        if random.random() < 0.05:  # 5% of the time
            log.debug(f"Wave not found. OCR results: {[r.text for r in frame.results]}")

    return None, None


def check_known_markers(frame: OCRFrame) -> None:
    """Log known UI markers when detected at expected positions."""
    log = logging.getLogger(__name__)
    w, h = frame.image_size
    if w == 0 or h == 0:
        return
    
    # Extract wave number for comparison
    wave_num_str, _ = extract_wave_from_frame(frame)
    wave_num = None
    if wave_num_str:
        try:
            wave_num = int(wave_num_str.replace(',', ''))
        except ValueError:
            pass
    
    for r in frame.results:
        cx, cy = r.center
        fx, fy = cx / w, cy / h
        # Check for Perk button
        if r.text.lower() == "perk" and abs(fx - 0.6194) < 0.05 and abs(fy - 0.0418) < 0.05:
            log.info(f"'Perk' detected at ({fx:.4f}, {fy:.4f})")
        
        # Check for perk threshold near position
        if abs(fx - 0.6341) < 0.05 and abs(fy - 0.0436) < 0.05:
            # Try to parse as number
            text_clean = r.text.replace(',', '').strip()
            if re.match(r'^\d+$', text_clean):
                try:
                    threshold_num = int(text_clean)
                    # Check if it's similar to wave number (within reasonable range)
                    if wave_num and abs(threshold_num - wave_num) < 10000:
                        log.info(f"Perk threshold detected: {r.text} at ({fx:.4f}, {fy:.4f}) [wave: {wave_num_str}]")
                except ValueError:
                    pass
        
        # Check for coins at position
        if abs(fx - 0.3132) < 0.02 and abs(fy - 0.0819) < 0.02:
            # Parse number with suffix (K, M, B, T)
            match = re.match(r'^([0-9.]+)\s*([KMBT])?$', r.text.strip(), re.IGNORECASE)
            if match:
                value = float(match.group(1))
                suffix = match.group(2).upper() if match.group(2) else ''
                
                multipliers = {'K': 1_000, 'M': 1_000_000, 'B': 1_000_000_000, 'T': 1_000_000_000_000}
                multiplier = multipliers.get(suffix, 1)
                coins = value * multiplier
                    
                

def build_screen_state(frame: OCRFrame) -> ScreenState:
    """Build ScreenState from OCR frame. Pure function."""
    log = logging.getLogger(__name__)
    screen = identify_screen(frame)
    check_known_markers(frame)
    
    elements = []
    resources = {}
    
    for ocr in frame.results:
        elem = classify_ocr_result(ocr, screen)
        if elem:
            elements.append(elem)
            if elem.element_type == "resource":
                resources[elem.name] = elem.text
            elif elem.element_type == "button":
                log.debug(f"Button classified: '{elem.text}' at {elem.center}")

    return ScreenState(
        screen=screen,
        elements=tuple(elements),
        resources=resources,
        raw_texts=tuple(r.text for r in frame.results),
        timestamp=time.time()
    )


def update_game_state(state: GameState, screen_state: ScreenState, frame: OCRFrame) -> GameState:
    """Update game state with new screen state. Pure function."""
    log = logging.getLogger(__name__)
    wave, wave_pos = extract_wave_from_frame(frame)
    
    # Also check if wave was classified as a resource
    if not wave and "wave" in screen_state.resources:
        wave_text = screen_state.resources["wave"]
        # Extract just the number from resource text like "Wave 123" or "123"
        digits = re.findall(r'[0-9,]+', wave_text)
        if digits:
            wave = digits[0].replace(',', '')
    
    new_resources = {**state.resources, **screen_state.resources}
    
    # Track wave progress and calculate rate
    new_wave_history = state.wave_history
    if wave and wave != state.wave:
        try:
            wave_num = int(wave.replace(',', ''))
            current_time = time.time()
            new_wave_history = (*state.wave_history, (wave_num, current_time))
            
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
    
    return replace(
        state,
        current_screen=screen_state,
        resources=new_resources,
        wave=wave or state.wave,
        wave_pos=wave_pos or state.wave_pos,
        wave_history=new_wave_history
    )


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


def detect_floating_gem(img: Optional[Image.Image], gem_template: Optional[np.ndarray], 
                       config: Config) -> Optional[Tuple[int, int]]:
    """Detect floating gem at any rotation near middle of screen. Returns (x, y) or None."""
    if img is None or gem_template is None:
        return None
    
    log = logging.getLogger(__name__)
    
    try:
        # Convert PIL to cv2
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        h, w = img_cv.shape
        
        # Define middle region (center 60% of screen)
        middle_margin_x = int(w * 0.2)
        middle_margin_y = int(h * 0.2)
        middle_region = img_cv[middle_margin_y:h-middle_margin_y, middle_margin_x:w-middle_margin_x]
        
        best_match = None
        best_val = 0.0
        threshold = 0.6  # Matching threshold
        
        # Try multiple rotations (0, 90, 180, 270 degrees)
        for angle in [0, 90, 180, 270]:
            # Rotate template
            if angle == 0:
                rotated = gem_template
            else:
                center = (gem_template.shape[1] // 2, gem_template.shape[0] // 2)
                matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(gem_template, matrix, 
                                        (gem_template.shape[1], gem_template.shape[0]))
            
            # Skip if template is larger than search region
            if rotated.shape[0] > middle_region.shape[0] or rotated.shape[1] > middle_region.shape[1]:
                continue
            
            # Template matching
            result = cv2.matchTemplate(middle_region, rotated, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            if max_val > best_val:
                best_val = max_val
                # Convert back to full image coordinates
                best_match = (max_loc[0] + middle_margin_x + rotated.shape[1] // 2,
                            max_loc[1] + middle_margin_y + rotated.shape[0] // 2)
        
        if best_val >= threshold and best_match:
            log.info(f"Floating gem detected at ({best_match[0]}, {best_match[1]}) with confidence {best_val:.2f}")
            return best_match
        
        return None
        
    except Exception as e:
        log.warning(f"Gem detection failed: {e}")
        return None


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
    
    return claimable


def decide_action(state: GameState, config: Config, latest_image: Optional[Image.Image] = None, 
                 gem_template: Optional[np.ndarray] = None, enabled: bool = True) -> Action:
    """Strategy decision function. Pure function."""
    log = logging.getLogger(__name__)
    
    if not enabled:
        return Action(action_type=ActionType.WAIT, duration=1.0, reason="Strategy disabled")

    # Priority 0: Click on floating gem (highest priority, with cooldown)
    if latest_image is not None and gem_template is not None:
        gem_pos = detect_floating_gem(latest_image, gem_template, config)
        if gem_pos:
            last_click_time = get_last_action_time(state, ActionType.CLICK)
            time_since_last_click = time.time() - last_click_time
            
            # Cooldown: at least 1 second between clicks
            if time_since_last_click > 1.0:
                return Action(
                    action_type=ActionType.CLICK,
                    x=gem_pos[0],
                    y=gem_pos[1],
                    reason="Clicking floating gem",
                    priority=100
                )

    # Priority 1: Click on claim/collect buttons (with cooldown)
    claimable = find_claimable_buttons(state)
    if claimable:
        last_claim_time = get_last_action_time(state, ActionType.CLICK)
        time_since_last_claim = time.time() - last_claim_time
        
        # Cooldown: at least 2 seconds between claims
        if time_since_last_claim > 2.0:
            # Click the first claimable button found
            button = claimable[0]
            cx, cy = button.center
            return Action(
                action_type=ActionType.CLICK,
                x=cx,
                y=cy,
                reason=f"Clicking {button.text} button",
                priority=10
            )
        else:
            log.debug(f"Cooldown active: {time_since_last_claim:.1f}s since last click")

    # Priority 2: Check if full scan needed
    if time.time() - state.last_full_scan > config.full_scan_interval:
        return Action(action_type=ActionType.FULL_SCAN, reason="Periodic full scan")

    # Default: observe current screen
    return Action(action_type=ActionType.SCAN_CURRENT, duration=1.0, reason="Observing")


# ============================================================================
# SIDE-EFFECT FUNCTIONS - INPUT
# ============================================================================

def to_absolute_coords(rel_x: int, rel_y: int, rect: WindowRect) -> Tuple[int, int]:
    """Convert relative to absolute coordinates."""
    return (rect.left + rel_x, rect.top + rel_y)


def execute_click(x: int, y: int, rect: WindowRect, config: Config) -> bool:
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
            debug_path = config.debug_dir / f"click_debug_{timestamp}.png"
            debug_img.save(debug_path)
            log.info(f"Debug screenshot saved: {debug_path}")
    except Exception as e:
        log.warning(f"Could not create debug screenshot: {e}")
    
    # Bring window to foreground before clicking
    if rect.hwnd and win32gui:
        try:
            win32gui.SetForegroundWindow(rect.hwnd)
            time.sleep(0.1)  # Brief delay for window to come to front
            log.debug(f"Brought window to foreground (hwnd={rect.hwnd})")
        except Exception as e:
            log.warning(f"Could not bring window to foreground: {e}")
    
    ax, ay = to_absolute_coords(x, y, rect)
    log.info(f"click({x}, {y}) → abs({ax}, {ay})")
    pyautogui.click(ax, ay, interval=config.click_pause)
    log.info(f"Clicked at ({ax}, {ay})")
    log.info('sleep for %.2f seconds', config.click_pause)
    time.sleep(config.click_pause)

    return True


def execute_swipe(x: int, y: int, distance: int, direction: str,
                 rect: WindowRect, config: Config, enabled: bool) -> bool:
    """Execute swipe action. Side effect."""
    if not enabled:
        return False
    
    log = logging.getLogger(__name__)
    ax, ay = to_absolute_coords(x, y, rect)
    offset = -distance if direction == "up" else distance
    
    log.info(f"swipe_{direction}({x}, {y}, {distance})")
    pyautogui.moveTo(ax, ay)
    pyautogui.drag(0, offset, duration=0.3)
    log.info(f"Swiped {direction} from ({ax}, {ay}) by {offset} pixels")
    log.info('sleep for %.2f seconds', config.click_pause)
    time.sleep(config.click_pause)
    return True


def execute_action(action: Action, rect: Optional[WindowRect],
                  config: Config,
                  scanner_fn: Callable, full_scan_fn: Callable) -> None:
    """Execute an action. Side effect dispatcher."""
    log = logging.getLogger(__name__)
    log.info(f"Action: {action.action_type.name} – {action.reason}")

    if action.action_type == ActionType.CLICK and rect:
        log.info('clicking at (%d, %d) relative to window', action.x, action.y)
        execute_click(action.x, action.y, rect, config)

    elif action.action_type == ActionType.SWIPE_UP and rect:
        execute_swipe(action.x, action.y, action.amount or 300,
                     "up", rect, config)

    elif action.action_type == ActionType.SWIPE_DOWN and rect:
        execute_swipe(action.x, action.y, action.amount or 300,
                     "down", rect, config)

    elif action.action_type == ActionType.WAIT:
        log.info('sleeping for %.2f seconds', action.duration)
        time.sleep(action.duration)

    elif action.action_type == ActionType.SCAN_CURRENT:
        scanner_fn()

    elif action.action_type == ActionType.FULL_SCAN:
        full_scan_fn()


# ============================================================================
# SIDE-EFFECT FUNCTIONS - SCANNING
# ============================================================================

def scan_current_screen(rect: WindowRect, config: Config, ocr_reader=None) -> Tuple[Optional[Image.Image], Optional[OCRFrame]]:
    """Capture and OCR current screen. Returns image and OCR frame."""
    img = capture_window(rect)
    if not img:
        return None, None

    try:
        frame = process_ocr(img, config, ocr_reader)
        # Save debug files
        if frame:
            save_debug_files(img, frame, config)
        return img, frame
    except Exception as e:
        logging.getLogger(__name__).error(f"OCR failed: {e}")
        return img, None


def perform_full_scan(rect: WindowRect, config: Config,
                     ocr_reader=None, enabled: bool = False) -> Dict[str, ScreenState]:
    """Perform full UI scan (simplified - just scans current screen)."""
    log = logging.getLogger(__name__)
    log.info("── Full scan (current screen) ──")

    # Just scan current screen, no tab navigation
    img, frame = scan_current_screen(rect, config, ocr_reader)
    if frame:
        current_state = build_screen_state(frame)
        return {"current": current_state}
    
    return {}


# ============================================================================
# STATEFUL RUNTIME CONTEXT
# ============================================================================

@dataclass
class RuntimeContext:
    """Container for mutable runtime state."""
    config: Config
    ocr_reader: Any = None
    gem_template: Optional[np.ndarray] = None
    game_state: GameState = field(default_factory=GameState)
    window_rect: Optional[WindowRect] = None
    latest_image: Optional[Image.Image] = None
    running: bool = False
    input_enabled: bool = False
    status: str = "stopped"
    last_window_check: float = 0.0

    def update_window(self):
        """Update window rect if needed."""
        if time.time() - self.last_window_check > 2.0:
            self.window_rect = find_window(self.config.window_title)
            self.last_window_check = time.time()

    def scan_current(self):
        """Scan current screen and update state."""
        if not self.window_rect:
            return

        img, frame = scan_current_screen(self.window_rect, self.config, self.ocr_reader)
        if img and frame:
            self.latest_image = img
            screen_state = build_screen_state(frame)
            self.game_state = update_game_state(self.game_state, screen_state, frame)

    def full_scan(self):
        """Perform full scan and update state."""
        if not self.window_rect:
            return

        # Just do a regular scan and update timestamp
        self.scan_current()
        self.game_state = replace(
            self.game_state,
            last_full_scan=time.time()
        )


def automation_loop_tick(ctx: RuntimeContext):
    """Single tick of automation loop."""
    # Update window rect
    log = logging.getLogger(__name__)
    ctx.update_window()
    if not ctx.window_rect:
        ctx.status = "no_window"
        return

    # Scan current screen
    ctx.scan_current()

    # Strategy decides
    action = decide_action(ctx.game_state, ctx.config, ctx.latest_image, ctx.gem_template)
    log.info('next action: %s – %s', action.action_type.name, action.reason)
    # Execute action
    execute_action(
        action, ctx.window_rect, ctx.config,
        scanner_fn=ctx.scan_current,
        full_scan_fn=ctx.full_scan
    )

    # Record action
    ctx.game_state = record_action_in_state(ctx.game_state, action)


def automation_loop_run(ctx: RuntimeContext):
    """Main loop runner."""
    log = logging.getLogger(__name__)
    log.info("Automation loop started")

    while ctx.running:
        t0 = time.time()
        try:
            automation_loop_tick(ctx)
        except Exception as exc:
            log.error(f"Loop tick error: {exc}", exc_info=True)
            ctx.game_state = replace(ctx.game_state,
                                    error_count=ctx.game_state.error_count + 1)
            log.info('sleeping for 2 seconds after error')
            time.sleep(2)

        elapsed = time.time() - t0
        sleep_time = max(0, ctx.config.loop_tick - elapsed)
        log.info('sleeping for %.2f seconds', sleep_time)
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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-24s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )


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

    # Create runtime context
    ctx = RuntimeContext(
        config=config,
        ocr_reader=ocr_reader,
        gem_template=gem_template,
        running=True,
        status="running",
        input_enabled=False,
    )

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
