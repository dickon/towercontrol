#!/usr/bin/env python3
"""
Torture test for upgrade-panel scroll-jump logic.

Hammers random tab + upgrade combinations for a configurable duration
(default 10 minutes), reporting live stats and a full summary on exit.

Usage:
    1. Stop run.py first (they share the game window and will interfere).
    2. Make sure the game is in BlueStacks on the upgrade screen.
    3. Run:
           python test_upgrade_scroll.py [--minutes 10] [--delay 2.0] [-v]

    Ctrl+C cleanly stops the run and prints the final summary.
"""

import argparse
import collections
import logging
import random
import signal
import sys
import time
from typing import Dict, List, Optional, Tuple

import pyautogui

import towercontrol_functional as tc

# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def _init_context() -> None:
    config = tc.Config()
    tc.ctx = tc.RuntimeContext(config=config)
    tc.ctx.input_enabled = True

    if tc.pytesseract:
        tc.pytesseract.pytesseract.tesseract_cmd = config.tesseract_cmd
    tc.ctx.ocr_reader = None

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


def _capture_and_ocr():
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
    texts = [(r.text, r.fx, r.fy) for r in frame.results]
    # Tight tolerance (0.05) prevents the word "Attack" inside upgrade-button
    # labels (e.g. "Free Attack Upgrade") from being mistaken for the tab header.
    defense = [t for t in texts if t[0].lower() == "defense" and abs(t[1] - 0.343) < 0.05 and abs(t[2] - 0.632) < 0.05]
    attack  = [t for t in texts if t[0].lower() == "attack"  and abs(t[1] - 0.329) < 0.05 and abs(t[2] - 0.632) < 0.05]
    utility = [t for t in texts if t[0].lower() == "utility" and abs(t[1] - 0.333) < 0.05 and abs(t[2] - 0.632) < 0.05]
    if defense: return 'DEFENSE'
    if attack:  return 'ATTACK'
    if utility: return 'UTILITY'
    return None


def _pick_random_target() -> Tuple[str, str]:
    category = random.choice(['ATTACK', 'DEFENSE', 'UTILITY'])
    upgrades = tc._get_category_upgrades_list(category)
    label = random.choice(upgrades)
    return category, label


# ---------------------------------------------------------------------------
# Stats tracker
# ---------------------------------------------------------------------------

RESULT_TAGS = ('SUCCESS', 'ALREADY_VISIBLE', 'OCR_MISS', 'CLOSE', 'MISS', 'FAIL', 'OCR_FAIL')


class Stats:
    def __init__(self):
        self.total   = 0
        self.counts  = collections.Counter()
        self.by_cat  = {c: collections.Counter() for c in ('ATTACK', 'DEFENSE', 'UTILITY')}
        self.failures: List[Dict] = []
        self.collapsed_reopens: int = 0    # times the panel was found collapsed and recovered
        self.deliberate_collapses: int = 0  # times we intentionally collapsed the panel

    def record(self, category: str, label: str, result: str) -> None:
        self.total += 1
        self.counts[result] += 1
        self.by_cat[category][result] += 1
        if result in ('MISS', 'FAIL'):
            self.failures.append({'category': category, 'label': label, 'result': result})

    @property
    def successes(self) -> int:
        return self.counts['SUCCESS'] + self.counts['ALREADY_VISIBLE']

    @property
    def ocr_issues(self) -> int:
        return self.counts['OCR_MISS'] + self.counts['OCR_FAIL']

    def success_rate(self) -> float:
        return 100.0 * self.successes / self.total if self.total else 0.0

    def print_summary(self, duration_secs: float) -> None:
        mins, secs = divmod(int(duration_secs), 60)
        print(f"\n{'='*70}")
        print(f"  TORTURE TEST COMPLETE")
        print(f"  Duration : {mins}m {secs:02d}s  |  Rounds: {self.total}")
        print(f"  Success  : {self.successes}/{self.total} ({self.success_rate():.1f}%)")
        print(f"{'='*70}")
        print(f"  {'Result':<22} {'Count':>6}  {'%':>6}")
        print(f"  {'-'*36}")
        for tag in RESULT_TAGS:
            n = self.counts[tag]
            if n:
                pct = 100.0 * n / self.total
                print(f"  {tag:<22} {n:>6}  {pct:>5.1f}%")

        print(f"\n  Per-category breakdown:")
        for cat in ('ATTACK', 'DEFENSE', 'UTILITY'):
            cat_total = sum(self.by_cat[cat].values())
            if cat_total == 0:
                continue
            cat_ok = self.by_cat[cat]['SUCCESS'] + self.by_cat[cat]['ALREADY_VISIBLE']
            print(f"    {cat:<8}  {cat_ok}/{cat_total} ({100*cat_ok/cat_total:.0f}%)"
                  f"  CLOSE={self.by_cat[cat]['CLOSE']}"
                  f"  MISS={self.by_cat[cat]['MISS']}"
                  f"  FAIL={self.by_cat[cat]['FAIL']}")

        if self.deliberate_collapses or self.collapsed_reopens:
            print(f"  Deliberate collapses       : {self.deliberate_collapses}")
            print(f"  Collapsed-panel recoveries : {self.collapsed_reopens}")

        if self.failures:
            print(f"\n  Failures ({len(self.failures)}):")
            fc: Dict[Tuple, int] = collections.Counter(
                (f['category'], f['label'], f['result']) for f in self.failures
            )
            for (cat, lbl, res), cnt in fc.most_common():
                print(f"    x{cnt:2d}  {cat} / '{lbl}' -> {res}")
        print()


# ---------------------------------------------------------------------------
# Main torture loop
# ---------------------------------------------------------------------------

_stop_requested = False


def _handle_sigint(sig, frame):
    global _stop_requested
    _stop_requested = True
    print("\n  [Ctrl+C -- finishing current round and printing summary...]")


def run_torture(duration_minutes: float = 10.0, delay: float = 2.0,
                max_attempts: int = 5) -> None:
    global _stop_requested

    _init_context()
    stats = Stats()
    start_time     = time.time()
    deadline       = start_time + duration_minutes * 60.0
    round_num      = 0
    last_print_time = start_time

    signal.signal(signal.SIGINT, _handle_sigint)

    print(f"\n{'='*70}")
    print(f"  Upgrade Scroll-Jump TORTURE TEST")
    print(f"  Duration  : {duration_minutes:.0f} min")
    print(f"  Delay     : {delay:.1f}s   |  Max attempts/round: {max_attempts}")
    print(f"{'='*70}\n")

    while time.time() < deadline and not _stop_requested:
        round_num += 1
        elapsed   = time.time() - start_time
        remaining = max(0.0, deadline - time.time())
        category, label = _pick_random_target()
        target_row = tc._upgrade_row_index(category, label)
        total_rows = tc._total_category_rows(category)

        print(f"\n--- Round {round_num}  [{elapsed/60:.1f}m elapsed, {remaining/60:.1f}m left]"
              f"  {category} / '{label}' (row {target_row}/{total_rows - 1}) ---")

        # -- Step 1: capture & detect current tab -------------------------
        img, frame, w, h = _capture_and_ocr()
        if not frame:
            print("  [SKIP] OCR failed")
            stats.record(category, label, 'OCR_FAIL')
            time.sleep(delay)
            continue

        current_tab = _detect_current_tab(frame)
        print(f"  Current tab: {current_tab}")

        # -- Step 2: switch tab if needed ----------------------------------
        if current_tab != category:
            print(f"  Switching to {category} tab...")
            tc._click_upgrade_tab(category)
            time.sleep(delay)

            img, frame, w, h = _capture_and_ocr()
            if not frame:
                print("  [SKIP] OCR failed after tab switch")
                stats.record(category, label, 'OCR_FAIL')
                time.sleep(delay)
                continue
            current_tab = _detect_current_tab(frame)
            if current_tab != category:
                print(f"  [WARN] Tab switch may have failed (detected: {current_tab})")

        # -- Step 2.5: deliberate collapse (33% of rounds) -----------------
        # Intentionally collapse the panel to exercise the recovery path.  We
        # only do this when the panel IS open (current_tab is not None) so we
        # know the toggle will actually close it.  Re-capture afterwards so
        # Step 3 works from a fresh (collapsed) frame.
        if current_tab is not None and random.random() < 0.33:
            print("  [CHAOS] Deliberately collapsing upgrade panel...")
            tc._collapse_upgrade_panel(category)
            stats.deliberate_collapses += 1
            time.sleep(delay)  # let the animation complete
            img, frame, w, h = _capture_and_ocr()
            if not frame:
                print("  [SKIP] OCR failed after deliberate collapse")
                stats.record(category, label, 'OCR_FAIL')
                time.sleep(delay)
                continue
            current_tab = _detect_current_tab(frame)
            print(f"  After collapse: tab={current_tab}")

        # -- Step 3: check already visible ---------------------------------
        upgrade_buttons = tc.detect_upgrade_buttons(frame, img, tc.ctx.config)
        visible_labels  = list(upgrade_buttons.keys())
        visible_top     = tc._estimate_visible_top_row(category, visible_labels)
        print(f"  Visible: {visible_labels}  (top row: {visible_top})")

        # -- Step 3.5: reopen collapsed upgrade panel -----------------------
        # If no upgrade buttons are detected AND no tab header is visible, the
        # upgrade panel has most likely been collapsed (minimised).  Use the
        # dedicated bottom-bar click to expand it, then re-capture.
        if not upgrade_buttons and current_tab is None:
            print("  [INFO] Upgrade panel appears collapsed — reopening...")
            tc._reopen_upgrade_panel(category)
            stats.collapsed_reopens += 1
            time.sleep(delay)
            img, frame, w, h = _capture_and_ocr()
            if not frame:
                print("  [SKIP] OCR failed after panel reopen")
                stats.record(category, label, 'OCR_FAIL')
                time.sleep(delay)
                continue
            current_tab = _detect_current_tab(frame)
            upgrade_buttons = tc.detect_upgrade_buttons(frame, img, tc.ctx.config)
            visible_labels  = list(upgrade_buttons.keys())
            visible_top     = tc._estimate_visible_top_row(category, visible_labels)
            print(f"  After reopen: tab={current_tab}, visible={visible_labels}  (top row: {visible_top})")
            # If the panel is now open but on the wrong sub-tab, switch to it
            if current_tab is not None and current_tab != category:
                print(f"  Switching to {category} tab after reopen...")
                tc._click_upgrade_tab(category)
                time.sleep(delay)
                img, frame, w, h = _capture_and_ocr()
                if not frame:
                    print("  [SKIP] OCR failed after post-reopen tab switch")
                    stats.record(category, label, 'OCR_FAIL')
                    time.sleep(delay)
                    continue
                current_tab = _detect_current_tab(frame)
                upgrade_buttons = tc.detect_upgrade_buttons(frame, img, tc.ctx.config)
                visible_labels  = list(upgrade_buttons.keys())
                visible_top     = tc._estimate_visible_top_row(category, visible_labels)
                print(f"  After tab switch: tab={current_tab}, visible={visible_labels}")

        if label in upgrade_buttons:
            print(f"  '{label}' already visible -- no scroll needed OK")
            stats.record(category, label, 'ALREADY_VISIBLE')
            last_print_time = _maybe_rolling(stats, last_print_time, elapsed)
            time.sleep(delay)
            continue

        # -- Step 4: iterative jump-scroll ---------------------------------
        found = False
        in_viewport_attempts = 0  # consecutive ticks where row is in viewport but label missing
        for attempt in range(1, max_attempts + 1):
            if _stop_requested:
                break

            scroll_result = tc._compute_upgrade_scroll_vector(
                category, label, upgrade_buttons, h
            )
            if scroll_result is tc._UPGRADE_SCROLL_IN_VIEWPORT:
                # Target row is in viewport but OCR missed the label this tick.
                # Try once more without scrolling; if still missing, count as OCR_MISS.
                in_viewport_attempts += 1
                print(f"  Attempt {attempt}: in viewport but OCR missed (#{in_viewport_attempts})")
                time.sleep(delay)

                img_a, frame_a, w_a, h_a = _capture_and_ocr()
                if frame_a:
                    upgrade_buttons_a = tc.detect_upgrade_buttons(frame_a, img_a, tc.ctx.config)
                    visible_a = list(upgrade_buttons_a.keys())
                    top_a = tc._estimate_visible_top_row(category, visible_a)
                    print(f"  -> {visible_a}  (top {top_a})")
                    if label in upgrade_buttons_a:
                        print(f"  OK SUCCESS in {attempt} attempt(s) (OCR catch-up)")
                        found = True
                        upgrade_buttons = upgrade_buttons_a
                        img, frame, w, h = img_a, frame_a, w_a, h_a
                        break
                    upgrade_buttons = upgrade_buttons_a
                    img, frame, w, h = img_a, frame_a, w_a, h_a
                if in_viewport_attempts >= 2:
                    print(f"  OCR missed {in_viewport_attempts}x for a row that should be visible")
                    break
                continue

            in_viewport_attempts = 0
            if scroll_result is None:
                # Visible buttons don't match the target category — likely wrong tab.
                # Force a tab re-click and re-read, then continue scrolling attempts.
                print(f"  [WARN] No scroll vector on attempt {attempt} — forcing tab re-click for {category}")
                tc._click_upgrade_tab(category)
                time.sleep(delay)
                img_r, frame_r, w_r, h_r = _capture_and_ocr()
                if frame_r:
                    upgrade_buttons = tc.detect_upgrade_buttons(frame_r, img_r, tc.ctx.config)
                    visible_r = list(upgrade_buttons.keys())
                    top_r = tc._estimate_visible_top_row(category, visible_r)
                    img, frame, w, h = img_r, frame_r, w_r, h_r
                    print(f"  -> after re-click: {visible_r}  (top {top_r})")
                    if label in upgrade_buttons:
                        print(f"  OK SUCCESS after tab re-click")
                        found = True
                        break
                    # Don't break — fall through to next attempt to scroll toward target
                    continue
                break

            direction, pixels = scroll_result
            print(f"  Attempt {attempt}: {direction} {pixels}px")
            tc._do_upgrade_jump_scroll(direction, pixels, w, h,
                                       f"torture '{label}' attempt {attempt}")
            time.sleep(delay)

            img_a, frame_a, w_a, h_a = _capture_and_ocr()
            if not frame_a:
                print(f"  [SKIP] OCR fail after attempt {attempt}")
                break

            upgrade_buttons_a = tc.detect_upgrade_buttons(frame_a, img_a, tc.ctx.config)
            visible_a = list(upgrade_buttons_a.keys())
            top_a     = tc._estimate_visible_top_row(category, visible_a)
            print(f"  -> {visible_a}  (top {top_a})")

            if label in upgrade_buttons_a:
                print(f"  OK SUCCESS in {attempt} attempt(s)")
                found = True
                upgrade_buttons = upgrade_buttons_a
                img, frame, w, h = img_a, frame_a, w_a, h_a
                break

            upgrade_buttons = upgrade_buttons_a
            img, frame, w, h = img_a, frame_a, w_a, h_a

        if found:
            stats.record(category, label, 'SUCCESS')
        else:
            t_row = tc._upgrade_row_index(category, label)
            v_top = tc._estimate_visible_top_row(category, list(upgrade_buttons.keys()))
            if in_viewport_attempts > 0:
                # Algorithm correctly placed target in viewport but OCR kept missing it
                print(f"  ~ OCR_MISS: row {t_row} was in viewport but label never detected")
                stats.record(category, label, 'OCR_MISS')
            elif v_top is not None and t_row is not None:
                dist = abs(t_row - v_top)
                if dist <= 2:
                    print(f"  ~ CLOSE: row {t_row}, visible_top {v_top} ({dist} off)")
                    stats.record(category, label, 'CLOSE')
                else:
                    print(f"  X MISS: row {t_row}, visible_top {v_top} ({dist} off)")
                    stats.record(category, label, 'MISS')
            else:
                print(f"  X FAIL: cannot determine position")
                stats.record(category, label, 'FAIL')

        last_print_time = _maybe_rolling(stats, last_print_time, elapsed)

    elapsed_total = time.time() - start_time
    stats.print_summary(elapsed_total)


def _maybe_rolling(stats: Stats, last_time: float, elapsed: float) -> float:
    """Print a rolling status line every 60 seconds; return updated last_time."""
    now = time.time()
    if now - last_time >= 60.0:
        print(f"\n  -- [{elapsed/60:.0f}m] Rolling: {stats.successes}/{stats.total} "
              f"({stats.success_rate():.1f}%)  "
              f"CLOSE={stats.counts['CLOSE']}  "
              f"MISS={stats.counts['MISS']}  "
              f"FAIL={stats.counts['FAIL']} --\n")
        return now
    return last_time


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upgrade scroll-jump torture test")
    parser.add_argument("--minutes",  type=float, default=10.0,
                        help="How long to run in minutes (default: 10)")
    parser.add_argument("--delay",    type=float, default=2.0,
                        help="Seconds to wait between actions (default: 2.0)")
    parser.add_argument("--attempts", type=int,   default=5,
                        help="Max scroll attempts per round before giving up (default: 5)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show DEBUG-level log output")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("PIL").setLevel(logging.WARNING)

    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.02

    run_torture(
        duration_minutes=args.minutes,
        delay=args.delay,
        max_attempts=args.attempts,
    )
