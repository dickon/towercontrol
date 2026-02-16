#!/usr/bin/env python3
"""Debug script to analyze OCR results for the test image."""

import json
from pathlib import Path
from PIL import Image
from towercontrol_functional import (
    process_ocr,
    detect_upgrade_buttons,
    UPGRADE_BUTTON_ROWS,
    UTILITY_UPGRADES,
    Config
)

# Create a minimal config
config = Config()

# Load the test image
test_image = Path("test_images/unexpected_upgrades_1771264045.png")
img = Image.open(test_image)

print(f"Image size: {img.size}")
print(f"Image dimensions: {img.width}x{img.height}")

# Run OCR
frame = process_ocr(img, config)

print(f"\nTotal OCR results: {len(frame.results)}")

# Show all OCR results with positions
print("\nAll OCR results (sorted by Y position):")
sorted_results = sorted(frame.results, key=lambda r: r.fy)
for i, r in enumerate(sorted_results):
    print(f"{i:3d}: y={r.fy:.3f} x={r.fx:.3f} text='{r.text}'")

# Show which upgrade button regions we're checking
print("\n\nUpgrade button regions we're checking:")
for idx, y_frac, (x_min, x_max) in UPGRADE_BUTTON_ROWS:
    print(f"Button {idx}: y={y_frac:.3f}, x=[{x_min:.3f}-{x_max:.3f}]")
    # Find OCR results in this region
    y_tolerance = 0.08
    region_results = [
        r for r in frame.results
        if abs(r.fy - y_frac) < y_tolerance and x_min <= r.fx <= x_max
    ]
    print(f"  Found {len(region_results)} OCR results in this region:")
    for r in region_results:
        print(f"    y={r.fy:.3f} x={r.fx:.3f} text='{r.text}'")

# Look for utility upgrade keywords in OCR results
print("\n\nSearching for utility upgrade keywords:")
utility_keywords = ['recovery', 'amount', 'max', 'package', 'chance', 'enemy', 'attack', 'health', 'level', 'skip']
for keyword in utility_keywords:
    matches = [r for r in frame.results if keyword in r.text.lower()]
    if matches:
        print(f"\n'{keyword}' found in:")
        for r in matches:
            print(f"  y={r.fy:.3f} x={r.fx:.3f} text='{r.text}'")

# Run the actual detection
print("\n\nRunning detect_upgrade_buttons:")
upgrades = detect_upgrade_buttons(frame, img)
print(f"Detected {len(upgrades)} upgrades:")
for label, cost, r in upgrades:
    print(f"  '{label}' - {cost} at y={r.fy:.3f} x={r.fx:.3f}")

# Expected upgrades
print("\n\nExpected upgrades:")
expected = [
    ("Recovery Amount", "340310"),
    ("Max Recovery", "321830"),
    ("Package Chance", "MAX"),
    ("Enemy Attack Level Skip", "1.32M"),
    ("Enemy Health Level Skip", "353280")
]
for label, cost in expected:
    print(f"  '{label}' - ${cost}")
