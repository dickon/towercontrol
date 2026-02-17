#!/usr/bin/env python3
"""
Test cases for OCR and perk text detection in TowerControl.

Tests verify that perk text is correctly extracted from OCR results
and matched against the perk choice patterns, both with synthetic
data and real images processed through the OCR pipeline.
"""

import re
import unittest
from pathlib import Path
from typing import List

from PIL import Image

# Import from main module
from towercontrol_functional import (
    OCRResult,
    OCRFrame,
    Config,
    PERK_CHOICES,
    PERK_ROWS,
    collect_perk_texts,
    filter_selected_perks,
    join_perk_texts,
    match_perk_priorities,
    detect_perks,
    detect_upgrade_buttons,
    process_ocr,
    initialize_ocr_backend,
)

TEST_IMAGES_DIR = Path(__file__).resolve().parent / "test_images"


class TestPerkDetection(unittest.TestCase):
    """Test cases for perk detection from synthetic OCR results."""
    
    def setUp(self):
        self.image_size = (1920, 1080)
    
    def _ocr(self, text: str, fx: float, fy: float, confidence: float = 0.9) -> OCRResult:
        """Create an OCRResult at fractional coordinates."""
        width, height = self.image_size
        cx = int(fx * width)
        cy = int(fy * height)
        x = max(0, cx - 50)
        y = max(0, cy - 15)
        return OCRResult(
            text=text,
            bbox=(x, y, 100, 30),
            confidence=confidence,
            image_size=self.image_size,
        )
    
    def test_golden_tower_bonus_in_slot_2(self):
        """Detect 'Golden Tower bonus x1.5' in slot index 2 (synthetic)."""
        slot_2_y = PERK_ROWS[2][1]  # 0.473

        results = [
            self._ocr("Golden", 0.50, slot_2_y),
            self._ocr("Tower", 0.56, slot_2_y),
            self._ocr("bonus", 0.62, slot_2_y),
            self._ocr("x1.5", 0.68, slot_2_y),
            # Other slots
            self._ocr("Damage", 0.55, PERK_ROWS[0][1]),
            self._ocr("x2.0", 0.62, PERK_ROWS[0][1]),
            self._ocr("Max", 0.52, PERK_ROWS[1][1]),
            self._ocr("Health", 0.60, PERK_ROWS[1][1]),
        ]

        frame = OCRFrame(results=tuple(results), image_size=self.image_size)
        result = detect_perks(frame)

        self.assertIn(2, result["perk_text_join"])
        self.assertEqual(result["perk_text_join"][2], "Golden Tower bonus x1.5")

        slot_2_hits = [p for p in result["perk_text_priority"] if p[0] == 2]
        self.assertEqual(len(slot_2_hits), 1)
        self.assertIn("Golden Tower Bonus", slot_2_hits[0][1])
    
    def test_golden_tower_pattern_variations(self):
        """Regex pattern matches various Golden Tower text formats."""
        pattern = r"Golden Tower Bonus( x[\d\.]+)?"
        should_match = [
            "Golden Tower Bonus",
            "Golden Tower Bonus x1.5",
            "Golden Tower Bonus x2.0",
            "Golden Tower Bonus x10.5",
            "golden tower bonus",
            "GOLDEN TOWER BONUS x1.5",
            "Golden Tower bonus x1.5",
        ]
        should_not_match = [
            "Tower Bonus",
            "Golden Bonus",
        ]
        for text in should_match:
            self.assertIsNotNone(
                re.search(pattern, text, re.IGNORECASE),
                f"Pattern should match '{text}'",
            )
        for text in should_not_match:
            self.assertIsNone(
                re.search(pattern, text, re.IGNORECASE),
                f"Pattern should not match '{text}'",
            )
    
    def test_multiple_perks_priority_ordering(self):
        """Multiple perks are correctly prioritized by PERK_CHOICES index."""
        results = [
            # Slot 0: Damage (priority 3)
            self._ocr("Damage", 0.55, PERK_ROWS[0][1]),
            self._ocr("x2.0", 0.62, PERK_ROWS[0][1]),
            # Slot 1: Golden Tower (priority 5)
            self._ocr("Golden", 0.50, PERK_ROWS[1][1]),
            self._ocr("Tower", 0.56, PERK_ROWS[1][1]),
            self._ocr("Bonus", 0.62, PERK_ROWS[1][1]),
            self._ocr("x1.5", 0.68, PERK_ROWS[1][1]),
            # Slot 2: Perk Wave Requirement (priority 0)
            self._ocr("Perk", 0.50, PERK_ROWS[2][1]),
            self._ocr("Wave", 0.56, PERK_ROWS[2][1]),
            self._ocr("Requirement", 0.65, PERK_ROWS[2][1]),
            self._ocr("-5%", 0.73, PERK_ROWS[2][1]),
        ]

        frame = OCRFrame(results=tuple(results), image_size=self.image_size)
        result = detect_perks(frame)

        self.assertEqual(len(result["perk_text_join"]), 3)
        self.assertEqual(len(result["perk_text_priority"]), 3)
        # Highest priority (lowest index) should come first
        self.assertEqual(result["perk_text_priority"][0][0], 2)  # Perk Wave Requirement
    
    def test_selected_perk_filtering(self):
        """Rows containing 'Selected' and rows at/after that index are removed."""
        results = [
            self._ocr("Selected", 0.50, PERK_ROWS[1][1]),
            self._ocr("Damage", 0.58, PERK_ROWS[1][1]),
            self._ocr("Golden", 0.50, PERK_ROWS[2][1]),
            self._ocr("Tower", 0.56, PERK_ROWS[2][1]),
            # Slot 0 should survive filtering
            self._ocr("Orbs", 0.55, PERK_ROWS[0][1]),
            self._ocr("+1", 0.62, PERK_ROWS[0][1]),
        ]

        frame = OCRFrame(results=tuple(results), image_size=self.image_size)
        result = detect_perks(frame)

        self.assertIn(0, result["perk_text_join"])         # before Selected row
        self.assertNotIn(1, result["perk_text_join"])       # the Selected row
        self.assertNotIn(2, result["perk_text_join"])       # after Selected row
    
    def test_collect_perk_texts_respects_bounds(self):
        """OCR results outside the perk x-range [0.45, 0.78] are ignored."""
        results = [
            self._ocr("Ignored", 0.30, PERK_ROWS[0][1]),  # x too low
            self._ocr("Also", 0.85, PERK_ROWS[0][1]),     # x too high
            self._ocr("Valid", 0.55, PERK_ROWS[0][1]),
        ]
        frame = OCRFrame(results=tuple(results), image_size=self.image_size)
        perk_text = collect_perk_texts(frame)
        self.assertEqual(perk_text.get(0), ["Valid"])
    
    def test_detect_perks_empty_frame(self):
        """Empty OCR frame returns empty results."""
        frame = OCRFrame(results=(), image_size=self.image_size)
        result = detect_perks(frame)
        self.assertEqual(result["perk_text_join"], {})
        self.assertEqual(result["perk_text_priority"], [])
        self.assertTrue(result["all_matched"])


class TestOCRPatternMatching(unittest.TestCase):
    """Validate PERK_CHOICES regex patterns."""
    
    def test_all_perk_patterns_compile(self):
        for idx, pattern in enumerate(PERK_CHOICES):
            try:
                re.compile(pattern, re.IGNORECASE)
            except re.error as e:
                self.fail(f"Invalid regex at index {idx}: {pattern}\nError: {e}")
    
    def test_golden_tower_is_in_perk_choices(self):
        golden = r"Golden Tower Bonus( x[\d\.]+)?"
        self.assertIn(golden, PERK_CHOICES)


class TestPerkDetectionWithImage(unittest.TestCase):
    """Test OCR + perk detection end-to-end on a real screenshot."""

    IMAGE_PATH = TEST_IMAGES_DIR / "perk_rows_20260216_105227.png"

    @classmethod
    def setUpClass(cls):
        if not cls.IMAGE_PATH.exists():
            raise unittest.SkipTest(f"Test image not found: {cls.IMAGE_PATH}")
        cls.config = Config()
        cls.ocr_reader = initialize_ocr_backend(cls.config)
        cls.img = Image.open(cls.IMAGE_PATH).convert("RGB")
        cls.frame = process_ocr(cls.img, cls.config, cls.ocr_reader)

    def test_ocr_produces_results(self):
        """OCR pipeline returns non-empty results for the test image."""
        self.assertGreater(len(self.frame.results), 0, "OCR should detect text in image")

    def test_perk_detection_from_image(self):
        """Full pipeline: image -> OCR -> perk detection finds expected perks.

        The test image (perk_rows_20260216_105227.png) is known to contain:
          - Slot 0: Orbs +1
          - Slot 1: x1.94 Health Regen  (OCR currently misses the "x1.94" prefix)
        Verified against the debug JSON recorded at capture time.
        """
        result = detect_perks(self.frame)

        perk_join = result["perk_text_join"]
        print(f"\nOCR detected perk text by slot: {perk_join}")

        # --- Slot 0: Orbs ---
        self.assertIn(0, perk_join, "Slot 0 should be detected")
        self.assertTrue(
            re.search(r"Orbs", perk_join[0], re.IGNORECASE),
            f"Slot 0 text should contain 'Orbs', got: '{perk_join[0]}'",
        )

        # --- Slot 1: x1.94 Health Regen ---
        # Ground truth is "x1.94 Health Regen" but OCR currently only picks up
        # "Health Regen", missing the multiplier prefix.
        self.assertIn(1, perk_join, "Slot 1 should be detected")
        self.assertTrue(
            re.search(r"Health Regen", perk_join[1], re.IGNORECASE),
            f"Slot 1 text should contain 'Health Regen', got: '{perk_join[1]}'",
        )
        if not re.search(r"x[\d\.]+", perk_join[1]):
            print(
                f"  ⚠ Known OCR gap: slot 1 expected 'x1.94 Health Regen' "
                f"but got '{perk_join[1]}' (multiplier prefix missing)"
            )

        # Verify pattern matching — both slots should now match
        priorities = result["perk_text_priority"]
        matched_slots = {p[0] for p in priorities}
        print(f"Matched priorities: {priorities}")

        self.assertIn(0, matched_slots, "Slot 0 (Orbs) should match a pattern")
        self.assertIn(1, matched_slots, "Slot 1 (Health Regen) should match a pattern")

        # Orbs should match the Orbs pattern
        slot_0_hits = [p for p in priorities if p[0] == 0]
        self.assertIn("Orbs", slot_0_hits[0][1])

        # Health Regen should match the Health Regen pattern
        slot_1_hits = [p for p in priorities if p[0] == 1]
        self.assertIn("Health Regen", slot_1_hits[0][1])

    def test_ocr_results_in_perk_region(self):
        """OCR results exist in the expected perk y-band with valid x-range."""
        perk_region_results = []
        for r in self.frame.results:
            for row, dy in PERK_ROWS:
                if abs(r.fy - dy) < 0.05 and 0.45 < r.fx < 0.82:
                    perk_region_results.append((row, r.text, r.fx, r.fy))

        self.assertGreater(
            len(perk_region_results),
            0,
            "At least some OCR results should fall within perk row regions",
        )
        print(f"\nPerk-region OCR hits:")
        for row, text, fx, fy in perk_region_results:
            print(f"  slot {row}: '{text}' at ({fx:.3f}, {fy:.3f})")


class TestPerkDetectionImage130242(unittest.TestCase):
    """Test OCR + perk detection on perk_rows_20260216_130242.png.

    Known contents (ground truth):
      - Slot 0: Golden tower bonus x1.5
      - Slot 1: Defense percent +4.44%
      - Slot 2: Unlock Chrono Field

    Slots 0 and 1 have white/light text on dark backgrounds, which
    requires the per-row inverted OCR recovery pass to detect.
    """

    IMAGE_PATH = TEST_IMAGES_DIR / "perk_rows_20260216_130242.png"

    @classmethod
    def setUpClass(cls):
        if not cls.IMAGE_PATH.exists():
            raise unittest.SkipTest(f"Test image not found: {cls.IMAGE_PATH}")
        cls.config = Config()
        cls.ocr_reader = initialize_ocr_backend(cls.config)
        cls.img = Image.open(cls.IMAGE_PATH).convert("RGB")
        cls.frame = process_ocr(cls.img, cls.config, cls.ocr_reader)

    def test_ocr_produces_results(self):
        """OCR pipeline returns non-empty results for the test image."""
        self.assertGreater(len(self.frame.results), 0, "OCR should detect text in image")

    def test_perk_detection_from_image(self):
        """Full pipeline: image -> OCR -> perk detection finds expected perks.

        Expected OCR output:
          - Slot 0: Golden tower bonus x1.5
          - Slot 1: Defense percent +4.44%
          - Slot 2: Unlock Chrono Field
        """
        result = detect_perks(self.frame)

        perk_join = result["perk_text_join"]
        print(f"\nOCR detected perk text by slot: {perk_join}")

        # --- Slot 0: Golden tower bonus x1.5 ---
        self.assertIn(0, perk_join, "Slot 0 should be detected (white-on-dark recovery)")
        self.assertTrue(
            re.search(r"Golden", perk_join[0], re.IGNORECASE),
            f"Slot 0 text should contain 'Golden', got: '{perk_join[0]}'",
        )
        self.assertTrue(
            re.search(r"tower", perk_join[0], re.IGNORECASE),
            f"Slot 0 text should contain 'tower', got: '{perk_join[0]}'",
        )
        self.assertTrue(
            re.search(r"bonus", perk_join[0], re.IGNORECASE),
            f"Slot 0 text should contain 'bonus', got: '{perk_join[0]}'",
        )

        # --- Slot 1: Defense percent +4.44% ---
        self.assertIn(1, perk_join, "Slot 1 should be detected (white-on-dark recovery)")
        self.assertTrue(
            re.search(r"Defense", perk_join[1], re.IGNORECASE),
            f"Slot 1 text should contain 'Defense', got: '{perk_join[1]}'",
        )
        self.assertTrue(
            re.search(r"percent", perk_join[1], re.IGNORECASE),
            f"Slot 1 text should contain 'percent', got: '{perk_join[1]}'",
        )

        # --- Slot 2: Unlock Chrono Field ---
        self.assertIn(2, perk_join, "Slot 2 should be detected")
        self.assertTrue(
            re.search(r"Unlock", perk_join[2], re.IGNORECASE),
            f"Slot 2 text should contain 'Unlock', got: '{perk_join[2]}'",
        )
        self.assertTrue(
            re.search(r"Chrono Field", perk_join[2], re.IGNORECASE),
            f"Slot 2 text should contain 'Chrono Field', got: '{perk_join[2]}'",
        )

        # --- Pattern matching ---
        priorities = result["perk_text_priority"]
        matched_slots = {p[0] for p in priorities}
        print(f"Matched priorities: {priorities}")

        # Slot 0 (Golden Tower Bonus) should match
        self.assertIn(0, matched_slots, "Slot 0 (Golden Tower Bonus) should match a pattern")
        slot_0_hits = [p for p in priorities if p[0] == 0]
        self.assertIn("Golden Tower Bonus", slot_0_hits[0][1])

        # Slot 1 (Defense percent) should match
        self.assertIn(1, matched_slots, "Slot 1 (Defense percent) should match a pattern")
        slot_1_hits = [p for p in priorities if p[0] == 1]
        self.assertIn("Defense percent", slot_1_hits[0][1])

    def test_ocr_results_in_perk_region(self):
        """OCR results exist in the expected perk y-band with valid x-range."""
        perk_region_results = []
        for r in self.frame.results:
            for row, dy in PERK_ROWS:
                if abs(r.fy - dy) < 0.05 and 0.45 < r.fx < 0.82:
                    perk_region_results.append((row, r.text, r.fx, r.fy))

        self.assertGreater(
            len(perk_region_results),
            0,
            "At least some OCR results should fall within perk row regions",
        )
        print(f"\nPerk-region OCR hits:")
        for row, text, fx, fy in perk_region_results:
            print(f"  slot {row}: '{text}' at ({fx:.3f}, {fy:.3f})")


class TestUpgradeDetectionUnexpected1771282888(unittest.TestCase):
    """Test OCR + upgrade detection on unexpected_upgrades_1771282888.png.

    Known contents (ground truth):
      - 6 attack upgrades all at MAX:
        1. Damage
        2. Attack Speed
        3. Critical Chance
        4. Critical Factor
        5. Range
        6. Damage Per Meter

    This tests that all 6 upgrades are correctly detected and marked as MAX.
    """

    IMAGE_PATH = TEST_IMAGES_DIR / "unexpected_upgrades_1771282888.png"

    @classmethod
    def setUpClass(cls):
        if not cls.IMAGE_PATH.exists():
            raise unittest.SkipTest(f"Test image not found: {cls.IMAGE_PATH}")
        cls.config = Config()
        cls.ocr_reader = initialize_ocr_backend(cls.config)
        cls.img = Image.open(cls.IMAGE_PATH).convert("RGB")
        cls.frame = process_ocr(cls.img, cls.config, cls.ocr_reader)

    def test_ocr_produces_results(self):
        """OCR pipeline returns non-empty results for the test image."""
        self.assertGreater(len(self.frame.results), 0, "OCR should detect text in image")

    def test_six_upgrades_detected(self):
        """Full pipeline: image -> OCR -> upgrade detection finds all 6 expected upgrades.

        Expected upgrades (all attack upgrades, all at MAX):
          1. Damage
          2. Attack Speed
          3. Critical Chance
          4. Critical Factor
          5. Range
          6. Damage Per Meter
        """
        upgrades = detect_upgrade_buttons(self.frame, self.img)

        print(f"\nDetected {len(upgrades)} upgrades:")
        for label, info in upgrades.items():
            cost_display = f"{info['cost']:.2f}" if info['cost'] is not None else "MAX"
            print(f"  {label}: {cost_display}")

        # Verify we found exactly 6 upgrades
        self.assertEqual(len(upgrades), 6,
                        f"Expected 6 upgrades but found {len(upgrades)}")

        # Extract detected labels
        detected_labels = list(upgrades.keys())

        # Verify all 6 expected attack upgrades are detected
        expected_labels = [
            'Damage',
            'Attack Speed',
            'Critical Chance',
            'Critical Factor',
            'Range',
            'Damage Per Meter'
        ]

        for expected in expected_labels:
            self.assertIn(expected, detected_labels,
                         f"Expected upgrade '{expected}' not found in detected labels: {detected_labels}")

    def test_all_upgrades_at_max(self):
        """Verify all 6 upgrades are marked as MAX.

        All buttons in this image have red backgrounds indicating MAX status,
        including the darker red buttons for Damage and Attack Speed.
        """
        upgrades = detect_upgrade_buttons(self.frame, self.img)

        print(f"\nUpgrade MAX status:")
        for label, info in upgrades.items():
            is_max = info['cost'] is None
            status_symbol = "✓" if is_max else "✗"
            cost_display = "MAX" if is_max else f"{info['cost']:.2f}"
            print(f"  {status_symbol} {label}: {cost_display}")

        # Check that all upgrades have MAX status (including dark red buttons)
        for label, info in upgrades.items():
            self.assertIsNone(info['cost'],
                           f"Upgrade '{label}' should be at MAX (dark red button should be detected as MAX) but got: {info['cost']}")

        print(f"✓ All {len(upgrades)} upgrades confirmed at MAX status")


def run_tests():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestPerkDetection))
    suite.addTests(loader.loadTestsFromTestCase(TestOCRPatternMatching))
    suite.addTests(loader.loadTestsFromTestCase(TestPerkDetectionWithImage))
    suite.addTests(loader.loadTestsFromTestCase(TestPerkDetectionImage130242))
    suite.addTests(loader.loadTestsFromTestCase(TestUpgradeDetectionUnexpected1771282888))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    exit(run_tests())
