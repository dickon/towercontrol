#!/usr/bin/env python3
"""
Test helper for upgrade-panel scroll-jump logic.

Exercises the ability to quickly jump between tabs (ATTACK / DEFENSE / UTILITY)
and scroll to random upgrades within each tab.  Validates that the target
upgrade becomes fully visible after the jump.

Usage:
    1. Stop run.py (they share the game window and will interfere).
    2. Make sure the game is running in BlueStacks on the upgrade screen.
    3. Run:
           python test_upgrade_scroll.py [--rounds N] [--delay SECS]

    The script will randomly pick a tab and an upgrade from that tab, switch
    to the correct tab, execute a jump-scroll, re-OCR, and report whether the
    target upgrade is now visible.

    Results are printed to stdout and summarised at the end.
"""

import argparse
import logging
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import pyautogui

import towercontrol_functional as tc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _init_context() -> None:
    """Bootstrap a minimal RuntimeContext (window, OCR reader, templates)."""
    config = tc.Config()
    tc.ctx = tc.RuntimeContext(config=config)
    tc.ctx.input_enabled = True

    # OCR reader
    if tc.pytesseract:
        tc.pytesseract.pytesseract.tesseract_cmd = config.tesseract_cmd
    tc.ctx.ocr_reader = None  # pytesseract doesn't need a reader object

    # Templates (optional for this test, but load to avoid warnings)
    tc.ctx.gem_template = tc.load_gem_template(config)
    tc.ctx.claim_template = tc.load_claim_template(config)
    tc.ctx.battle_template = tc.load_battle_template(config)
    tc.ctx.newperk_template = None
    tc.ctx.resume_battle_template = None

    tc.ctx.update_window()
    if not tc.ctx.window_rect:
        print(f"ERROR: Cannot find window '{config.window_title}'")
        sys.exit(1)
    print(f"Window found: {tc.ctx.window_rect}")


def _capture_and_ocr() -> Tuple[Optional[object], Optional[tc.OCRFrame], int, int]:
    """Capture the window, run OCR, return (img, frame, w, h)."""
    tc.ctx.update_window()
    rect = tc.ctx.window_rect
    if not rect:
        return None, None, 0, 0
    img = tc.capture_window(rect)
    if not img:
        return None, None, 0, 0
    frame = tc.process_ocr(img, tc.ctx.config, tc.ctx.ocr_reader)
    if not frame:
        return img, None, 0, 0
    w, h = frame.image_size
    return img, frame, w, h


def _detect_current_tab(frame: tc.OCRFrame) -> Optional[str]:
    """Return the currently visible upgrade tab ('ATTACK'/'DEFENSE'/'UTILITY') or None."""
    texts = [(r.text, r.fx, r.fy) for r in frame.results]
    defense = [t for t in texts if t[0].lower() == "defense" and abs(t[1] - 0.343) < 0.1 and abs(t[2] - 0.632) < 0.1]
    attack = [t for t in texts if t[0].lower() == "attack" and abs(t[1] - 0.329) < 0.1 and abs(t[2] - 0.632) < 0.1]
    utility = [t for t in texts if t[0].lower() == "utility" and abs(t[1] - 0.333) < 0.1 and abs(t[2] - 0.632) < 0.1]
    if defense:
        return 'DEFENSE'
    if attack:
        return 'ATTACK'
    if utility:
        return 'UTILITY'
    return None


def _pick_random_target() -> Tuple[str, str]:
    """Return a random (category, label) from the upgrade lists."""
    category = random.choice(['ATTACK', 'DEFENSE', 'UTILITY'])
    upgrades = tc._get_category_upgrades_list(category)
    label = random.choice(upgrades)
    return category, label


# ---------------------------------------------------------------------------
# Main test loop
# ---------------------------------------------------------------------------

def run_test(rounds: int = 20, delay: float = 2.5) -> None:
    """Execute *rounds* random scroll-jump tests."""
    _init_context()

    results: List[Dict] = []
    success_count = 0

    print(f"\n{'='*70}")
    print(f"  Upgrade Scroll-Jump Test  —  {rounds} rounds, {delay:.1f}s delay")
    print(f"{'='*70}\n")

    for i in range(1, rounds + 1):
        category, label = _pick_random_target()
        target_row = tc._upgrade_row_index(category, label)
        total_rows = tc._total_category_rows(category)
        print(f"\n--- Round {i}/{rounds}: {category} / '{label}' "
              f"(row {target_row}/{total_rows - 1}) ---")

        # Step 1: Capture and detect current state
        img, frame, w, h = _capture_and_ocr()
        if not frame:
            print("  [SKIP] OCR failed")
            results.append({'round': i, 'category': category, 'label': label, 'result': 'OCR_FAIL'})
            time.sleep(delay)
            continue

        current_tab = _detect_current_tab(frame)
        print(f"  Current tab: {current_tab}")

        # Step 2: Switch tab if needed
        if current_tab != category:
            print(f"  Switching to {category} tab...")
            tc._click_upgrade_tab(category)
            time.sleep(delay)

            # Re-capture after tab switch
            img, frame, w, h = _capture_and_ocr()
            if not frame:
                print("  [SKIP] OCR failed after tab switch")
                results.append({'round': i, 'category': category, 'label': label, 'result': 'OCR_FAIL'})
                time.sleep(delay)
                continue
            current_tab = _detect_current_tab(frame)
            if current_tab != category:
                print(f"  [WARN] Tab switch may have failed (see: {current_tab})")

        # Step 3: Detect visible upgrade buttons
        upgrade_buttons = tc.detect_upgrade_buttons(frame, img, tc.ctx.config)
        visible_labels = list(upgrade_buttons.keys())
        visible_top = tc._estimate_visible_top_row(category, visible_labels)
        print(f"  Visible buttons: {visible_labels}")
        print(f"  Visible top row: {visible_top}")

        # Check if already visible
        if label in upgrade_buttons:
            print(f"  '{label}' already visible — no scroll needed ✓")
            results.append({'round': i, 'category': category, 'label': label, 'result': 'ALREADY_VISIBLE'})
            success_count += 1
            time.sleep(delay)
            continue

        # Step 4: Compute and execute jump-scroll (up to 4 attempts, simulating multiple ticks)
        found = False
        for attempt in range(1, 5):
            scroll_result = tc._compute_upgrade_scroll_vector(
                category, label, upgrade_buttons, h
            )
            if scroll_result is None:
                print(f"  [WARN] Cannot compute scroll vector on attempt {attempt} (no visible buttons?)")
                break

            direction, pixels = scroll_result
            print(f"  Attempt {attempt} scroll: {direction} by {pixels}px")
            tc._do_upgrade_jump_scroll(direction, pixels, w, h,
                                        f"test jump to '{label}' attempt {attempt}")
            time.sleep(delay)

            # Re-capture and check
            img_a, frame_a, w_a, h_a = _capture_and_ocr()
            if not frame_a:
                print(f"  [SKIP] OCR failed after scroll attempt {attempt}")
                break

            upgrade_buttons_a = tc.detect_upgrade_buttons(frame_a, img_a, tc.ctx.config)
            visible_labels_a = list(upgrade_buttons_a.keys())
            visible_top_a = tc._estimate_visible_top_row(category, visible_labels_a)
            print(f"  After attempt {attempt} — visible: {visible_labels_a}, top row: {visible_top_a}")

            if label in upgrade_buttons_a:
                print(f"  ✓ SUCCESS after {attempt} attempt(s): '{label}' is now visible")
                found = True
                # Refresh for next round
                upgrade_buttons = upgrade_buttons_a
                img, frame, w, h = img_a, frame_a, w_a, h_a
                break

            # Update state for next attempt
            upgrade_buttons = upgrade_buttons_a
            img, frame, w, h = img_a, frame_a, w_a, h_a

        if found:
            results.append({'round': i, 'category': category, 'label': label,
                            'result': 'SUCCESS'})
            success_count += 1
        else:
            target_row_now = tc._upgrade_row_index(category, label)
            visible_top_final = tc._estimate_visible_top_row(category, list(upgrade_buttons.keys()))
            if visible_top_final is not None and target_row_now is not None:
                dist = abs(target_row_now - visible_top_final)
                if dist <= 2:
                    print(f"  ~ CLOSE after all attempts: row {target_row_now}, top {visible_top_final} ({dist} rows off)")
                    results.append({'round': i, 'category': category, 'label': label, 'result': 'CLOSE'})
                else:
                    print(f"  ✗ MISS after all attempts: row {target_row_now}, top {visible_top_final} ({dist} rows off)")
                    results.append({'round': i, 'category': category, 'label': label, 'result': 'MISS'})
            else:
                print(f"  ✗ FAIL after all attempts: cannot verify position")
                results.append({'round': i, 'category': category, 'label': label, 'result': 'FAIL'})

    # Summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY: {success_count}/{rounds} successful "
          f"({100*success_count/rounds:.0f}%)")
    print(f"{'='*70}")
    for tag in ['SUCCESS', 'ALREADY_VISIBLE', 'CLOSE', 'MISS', 'FAIL', 'OCR_FAIL']:
        count = sum(1 for r in results if r['result'] == tag)
        if count:
            print(f"  {tag:20s}: {count}")
    print()

    # Print failures for debugging
    failures = [r for r in results if r['result'] in ('MISS', 'FAIL')]
    if failures:
        print("Failed rounds:")
        for r in failures:
            print(f"  Round {r['round']}: {r['category']} / '{r['label']}' → {r['result']}")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test upgrade scroll-jump logic")
    parser.add_argument("--rounds", type=int, default=20,
                        help="Number of random scroll tests (default: 20)")
    parser.add_argument("--delay", type=float, default=2.5,
                        help="Seconds to wait between actions (default: 2.5)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable DEBUG logging")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    # Suppress noisy libraries
    logging.getLogger("PIL").setLevel(logging.WARNING)

    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.02

    run_test(rounds=args.rounds, delay=args.delay)
