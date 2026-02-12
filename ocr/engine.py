"""
OCR engine with image pre-processing for game UI text.

Supports pytesseract and easyocr backends.  Results include bounding boxes
so callers know *where* text appears on screen.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

log = logging.getLogger(__name__)


@dataclass
class OCRResult:
    """A single detected text region."""
    text: str
    bbox: Tuple[int, int, int, int]  # (x, y, w, h) in capture coords
    confidence: float                # 0-100

    @property
    def center(self) -> Tuple[int, int]:
        x, y, w, h = self.bbox
        return (x + w // 2, y + h // 2)

    @property
    def area(self) -> int:
        return self.bbox[2] * self.bbox[3]


@dataclass
class OCRFrame:
    """All OCR results from a single capture frame."""
    results: List[OCRResult] = field(default_factory=list)
    image_size: Tuple[int, int] = (0, 0)  # (width, height)
    raw_image: Optional[np.ndarray] = None
    processed_image: Optional[np.ndarray] = None

    def texts(self) -> List[str]:
        return [r.text for r in self.results]

    def find(self, pattern: str, case_insensitive: bool = True) -> List[OCRResult]:
        """Return results whose text matches a regex pattern."""
        flags = re.IGNORECASE if case_insensitive else 0
        return [r for r in self.results if re.search(pattern, r.text, flags)]

    def find_nearest(self, x: int, y: int, pattern: str = "",
                     max_dist: float = 9999) -> Optional[OCRResult]:
        """Find the OCR result closest to (x, y), optionally matching a pattern."""
        candidates = self.find(pattern) if pattern else self.results
        best, best_dist = None, max_dist
        for r in candidates:
            cx, cy = r.center
            d = ((cx - x) ** 2 + (cy - y) ** 2) ** 0.5
            if d < best_dist:
                best, best_dist = r, d
        return best

    def find_all_in_region(self, x: int, y: int, w: int, h: int) -> List[OCRResult]:
        """Return all results whose center falls inside the given rectangle."""
        out = []
        for r in self.results:
            cx, cy = r.center
            if x <= cx <= x + w and y <= cy <= y + h:
                out.append(r)
        return out


class OCREngine:
    """Wraps pytesseract / easyocr with game-tuned preprocessing."""

    def __init__(self, engine: str = "pytesseract", lang: str = "eng",
                 confidence_threshold: float = 40.0):
        self.engine_name = engine
        self.lang = lang
        self.confidence_threshold = confidence_threshold
        self._backend = None
        self._init_backend()

    # ── public ──────────────────────────────────────────────────────────

    def process(self, image: Image.Image | np.ndarray,
                preprocess: bool = True) -> OCRFrame:
        """Run OCR on an image and return structured results."""
        if isinstance(image, Image.Image):
            arr = np.array(image)
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        else:
            arr = image.copy()

        h, w = arr.shape[:2]
        processed = self._preprocess(arr) if preprocess else arr

        if self.engine_name == "easyocr":
            results = self._run_easyocr(processed)
        else:
            results = self._run_pytesseract(processed)

        # Filter by confidence
        results = [r for r in results
                   if r.confidence >= self.confidence_threshold and r.text.strip()]

        frame = OCRFrame(
            results=results,
            image_size=(w, h),
            raw_image=arr,
            processed_image=processed,
        )
        log.debug("OCR found %d results (engine=%s)", len(results), self.engine_name)
        return frame

    # ── preprocessing ───────────────────────────────────────────────────

    @staticmethod
    def _preprocess(img: np.ndarray) -> np.ndarray:
        """Enhance game UI text for OCR accuracy."""
        # Resize up if small (helps tesseract)
        h, w = img.shape[:2]
        scale = 1
        if w < 800:
            scale = 2
            img = cv2.resize(img, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Adaptive threshold works well for varied game backgrounds
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 10,
        )

        # Light denoise
        denoised = cv2.medianBlur(thresh, 3)
        return denoised

    # ── backends ────────────────────────────────────────────────────────

    def _init_backend(self):
        if self.engine_name == "easyocr":
            try:
                import easyocr
                self._backend = easyocr.Reader([self.lang], gpu=False)
                log.info("EasyOCR backend ready")
            except ImportError:
                log.warning("easyocr not installed – falling back to pytesseract")
                self.engine_name = "pytesseract"
                self._init_backend()
        else:
            try:
                import pytesseract  # noqa: F401
                log.info("pytesseract backend ready")
            except ImportError:
                log.error("pytesseract not installed")

    def _run_pytesseract(self, img: np.ndarray) -> List[OCRResult]:
        import pytesseract

        data = pytesseract.image_to_data(
            img, lang=self.lang, output_type=pytesseract.Output.DICT,
            config="--psm 11",  # sparse text – good for game UIs
        )
        results: List[OCRResult] = []
        n = len(data["text"])
        for i in range(n):
            txt = data["text"][i].strip()
            conf = float(data["conf"][i])
            if conf < 0 or not txt:
                continue
            x, y, w, h = (data["left"][i], data["top"][i],
                          data["width"][i], data["height"][i])
            results.append(OCRResult(text=txt, bbox=(x, y, w, h),
                                     confidence=conf))
        return results

    def _run_easyocr(self, img: np.ndarray) -> List[OCRResult]:
        if self._backend is None:
            return []
        raw = self._backend.readtext(img)
        results: List[OCRResult] = []
        for bbox_pts, text, conf in raw:
            # bbox_pts is [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
            xs = [p[0] for p in bbox_pts]
            ys = [p[1] for p in bbox_pts]
            x, y = int(min(xs)), int(min(ys))
            w, h = int(max(xs) - x), int(max(ys) - y)
            results.append(OCRResult(
                text=text.strip(),
                bbox=(x, y, w, h),
                confidence=conf * 100,
            ))
        return results
