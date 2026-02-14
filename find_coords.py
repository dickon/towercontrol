#!/usr/bin/env python3
"""Quick script to find coordinates of specific text."""

import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

from capture.window import WindowCapture
from ocr.engine import OCREngine
import config

# Initialize
capture = WindowCapture(config.WINDOW_TITLE_PATTERN)
ocr = OCREngine(engine=config.OCR_ENGINE, lang=config.OCR_LANG)

# Find window
rect = capture.find_window()
if not rect:
    print("ERROR: Could not find game window")
    sys.exit(1)

print(f"Found window: {rect.width}x{rect.height} at ({rect.left}, {rect.top})")

# Capture frame
img = capture.capture_np()
if img is None:
    print("ERROR: Failed to capture")
    sys.exit(1)

print(f"Captured {img.shape[1]}x{img.shape[0]} image")

# Save the captured images
import cv2
import os
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
raw_path = os.path.abspath(f"screenshot_{timestamp}.png")
overlay_path = os.path.abspath(f"screenshot_{timestamp}_ocr.png")

# Save raw screenshot
cv2.imwrite(raw_path, img)
print(f"Saved raw screenshot to: {raw_path}")

# Run OCR
frame = ocr.process(img)
print(f"Found {len(frame.results)} text regions\n")

# Create overlay image with OCR bounding boxes
overlay_img = img.copy()
for r in frame.results:
    x, y, w, h = r.bbox
    # Draw rectangle
    cv2.rectangle(overlay_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # Draw text label
    label = r.text[:20]  # truncate long text
    cv2.putText(overlay_img, label, (x, y - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# Save overlay screenshot
cv2.imwrite(overlay_path, overlay_img)
print(f"Saved OCR overlay to: {overlay_path}\n")

# Search for the target text
target = "wave"
print(f"Searching for text containing: '{target}'\n")
found_any = False
for r in frame.results:
    if target.lower() in r.text.lower():
        found_any = True
        x, y, w, h = r.bbox
        cx, cy = r.center
        # Convert to fractional coordinates
        fx = cx / rect.width
        fy = cy / rect.height
        print(f"Found: '{r.text}'")
        print(f"  Bbox: x={x}, y={y}, w={w}, h={h}")
        print(f"  Center: ({cx}, {cy})")
        print(f"  Fractional: ({fx:.3f}, {fy:.3f})")
        print(f"  Confidence: {r.confidence:.1f}")
        print()

if not found_any:
    print(f"*** No text containing '{target}' was found ***")
    print("*** Please ensure the upgrade with '97.00%' is visible on screen ***\n")

# Also show all detected text with coordinates
print("\n--- All detected text ---")
for r in sorted(frame.results, key=lambda x: (x.bbox[1], x.bbox[0])):
    x, y, w, h = r.bbox
    cx, cy = r.center
    fx = cx / rect.width
    fy = cy / rect.height
    print(f"'{r.text:20s}' @ ({cx:4d},{cy:4d}) = ({fx:.3f},{fy:.3f})")
