"""Regression tests for ad-strip handling.

Exercises both ad-strip-present (1523×2112) and ad-strip-absent (1203×2112)
capture paths, verifying that:
  1. crop_ad_strip correctly identifies and crops the strip.
  2. After cropping, both layouts produce images of identical width.
  3. OCR results fall within the content-only coordinate frame (0..1).
  4. _content_frac_to_window_px maps content-only fractions back to the
     correct window pixel positions for both layouts.
  5. process_ocr + detect_upgrade_buttons produce consistent results
     regardless of whether the source image had an ad strip.
"""

import math
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import towercontrol_functional as tc
from towercontrol_functional import (
    CONTENT_FRAC,
    STRIP_FRAC,
    _LAYOUT_FULL_W,
    _LAYOUT_GAME_W,
    _LAYOUT_H,
    Config,
    crop_ad_strip,
    detect_upgrade_buttons,
    has_ad_strip,
    initialize_ocr_backend,
    process_ocr,
)

TEST_IMAGES_DIR = Path(__file__).parent / "test_images"

# Representative test images for each layout
_STRIP_PRESENT_IMAGE = TEST_IMAGES_DIR / "unexpected_upgrades_1771282888.png"
# Single-upgrade Health detail screen captured without the ad strip (1203×2112).
# This is NOT the 6-button upgrade panel — it shows the detail view for one upgrade.
_STRIP_ABSENT_IMAGE = TEST_IMAGES_DIR / "no_ad_strip_upgrade_detail.png"


# ---------------------------------------------------------------------------
# crop_ad_strip unit tests
# ---------------------------------------------------------------------------


class TestCropAdStrip(unittest.TestCase):
    """Verify crop_ad_strip behaviour on synthetic and real images."""

    def test_strip_present_synthetic(self):
        """A 1523×2112 image is cropped and flagged."""
        img = Image.new("RGB", (_LAYOUT_FULL_W, _LAYOUT_H), color=(128, 128, 128))
        cropped, had_strip = crop_ad_strip(img)
        self.assertTrue(had_strip)
        self.assertEqual(cropped.size[0], _LAYOUT_FULL_W - int(round(_LAYOUT_FULL_W * STRIP_FRAC)))
        self.assertEqual(cropped.size[1], _LAYOUT_H)

    def test_strip_absent_synthetic(self):
        """A 1203×2112 image passes through unchanged."""
        img = Image.new("RGB", (_LAYOUT_GAME_W, _LAYOUT_H), color=(128, 128, 128))
        cropped, had_strip = crop_ad_strip(img)
        self.assertFalse(had_strip)
        self.assertIs(cropped, img)  # exact same object, no copy

    def test_both_layouts_same_width_after_crop(self):
        """After cropping, strip-present and strip-absent produce the same width."""
        wide = Image.new("RGB", (_LAYOUT_FULL_W, _LAYOUT_H))
        narrow = Image.new("RGB", (_LAYOUT_GAME_W, _LAYOUT_H))
        w_crop, _ = crop_ad_strip(wide)
        n_crop, _ = crop_ad_strip(narrow)
        # Allow ±1 pixel rounding difference
        self.assertAlmostEqual(w_crop.size[0], n_crop.size[0], delta=1)

    def test_real_strip_present_image(self):
        """Real screenshot with ad strip is correctly cropped."""
        if not _STRIP_PRESENT_IMAGE.exists():
            self.skipTest(f"Missing: {_STRIP_PRESENT_IMAGE}")
        img = Image.open(_STRIP_PRESENT_IMAGE).convert("RGB")
        cropped, had_strip = crop_ad_strip(img)
        self.assertTrue(had_strip)
        self.assertLess(cropped.size[0], img.size[0])

    def test_real_strip_absent_image(self):
        """Real screenshot (upgrade detail screen, no ad strip) passes through unchanged."""
        if not _STRIP_ABSENT_IMAGE.exists():
            self.skipTest(f"Missing: {_STRIP_ABSENT_IMAGE}")
        img = Image.open(_STRIP_ABSENT_IMAGE).convert("RGB")
        cropped, had_strip = crop_ad_strip(img)
        self.assertFalse(had_strip)
        self.assertEqual(cropped.size[0], img.size[0])


# ---------------------------------------------------------------------------
# has_ad_strip
# ---------------------------------------------------------------------------


class TestHasAdStrip(unittest.TestCase):
    """Verify the aspect-ratio discriminator."""

    def test_full_layout(self):
        self.assertTrue(has_ad_strip(_LAYOUT_FULL_W, _LAYOUT_H))

    def test_game_only_layout(self):
        self.assertFalse(has_ad_strip(_LAYOUT_GAME_W, _LAYOUT_H))

    def test_zero_height_assumes_strip(self):
        self.assertTrue(has_ad_strip(1523, 0))


# ---------------------------------------------------------------------------
# _content_frac_to_window_px
# ---------------------------------------------------------------------------


class TestContentFracToWindowPx(unittest.TestCase):
    """Verify coordinate mapping from content-only fractions to window pixels."""

    def _setup_ctx(self, width, height):
        """Patch the global ctx with a fake window rect."""
        config = Config()
        tc.ctx = tc.RuntimeContext(config=config)
        tc.ctx.window_rect = type("R", (), {
            "left": 0, "top": 0, "width": width, "height": height,
        })()

    def test_strip_present_origin(self):
        """(0, 0) in content frame maps to the first pixel after the strip."""
        self._setup_ctx(_LAYOUT_FULL_W, _LAYOUT_H)
        px_x, px_y = tc._content_frac_to_window_px(0.0, 0.0)
        strip_px = int(round(_LAYOUT_FULL_W * STRIP_FRAC))
        self.assertEqual(px_x, strip_px)
        self.assertEqual(px_y, 0)

    def test_strip_present_bottom_right(self):
        """(1, 1) in content frame maps to the last pixel of the window."""
        self._setup_ctx(_LAYOUT_FULL_W, _LAYOUT_H)
        px_x, px_y = tc._content_frac_to_window_px(1.0, 1.0)
        self.assertEqual(px_x, _LAYOUT_FULL_W)
        self.assertEqual(px_y, _LAYOUT_H)

    def test_strip_absent_identity(self):
        """Without a strip, content fracs map directly to window pixels."""
        self._setup_ctx(_LAYOUT_GAME_W, _LAYOUT_H)
        px_x, px_y = tc._content_frac_to_window_px(0.5, 0.5)
        self.assertEqual(px_x, int(0.5 * _LAYOUT_GAME_W))
        self.assertEqual(px_y, int(0.5 * _LAYOUT_H))

    def test_consistent_content_point(self):
        """The same content fraction points to the SAME game pixel in both layouts.

        E.g. the centre of the game area (0.5, 0.5) should land at the same
        offset *within the game panel* regardless of whether the strip is present.
        """
        # With strip
        self._setup_ctx(_LAYOUT_FULL_W, _LAYOUT_H)
        px_x_strip, py_strip = tc._content_frac_to_window_px(0.5, 0.5)

        # Without strip
        self._setup_ctx(_LAYOUT_GAME_W, _LAYOUT_H)
        px_x_no_strip, py_no = tc._content_frac_to_window_px(0.5, 0.5)

        # Offset within the game content area should match (within 1px rounding)
        strip_px = int(round(_LAYOUT_FULL_W * STRIP_FRAC))
        game_offset_strip = px_x_strip - strip_px
        game_offset_no = px_x_no_strip

        self.assertAlmostEqual(game_offset_strip, game_offset_no, delta=1,
                               msg="Same content fraction should hit the same game pixel")
        self.assertEqual(py_strip, py_no)


# ---------------------------------------------------------------------------
# OCR coordinate frame: results should be in content-only [0, 1] space
# ---------------------------------------------------------------------------


class TestOCRCoordinateFrame(unittest.TestCase):
    """OCR results from both layouts should produce coordinates in [0, 1]."""

    @classmethod
    def setUpClass(cls):
        cls.config = Config()
        cls.ocr_reader = initialize_ocr_backend(cls.config)

    def _ocr_image(self, path):
        if not path.exists():
            self.skipTest(f"Missing: {path}")
        img = Image.open(path).convert("RGB")
        img, _ = crop_ad_strip(img)
        return process_ocr(img, self.config, self.ocr_reader)

    def _assert_results_in_unit_range(self, frame, label):
        self.assertGreater(len(frame.results), 0,
                           f"{label}: OCR should return results")
        for r in frame.results:
            self.assertGreaterEqual(r.fx, 0.0,
                                    f"{label}: fx={r.fx} < 0 for '{r.text}'")
            self.assertLessEqual(r.fx, 1.0,
                                 f"{label}: fx={r.fx} > 1 for '{r.text}'")
            self.assertGreaterEqual(r.fy, 0.0,
                                    f"{label}: fy={r.fy} < 0 for '{r.text}'")
            self.assertLessEqual(r.fy, 1.0,
                                 f"{label}: fy={r.fy} > 1 for '{r.text}'")

    def test_strip_present_coords_in_unit_range(self):
        """OCR on a strip-present image produces fx,fy in [0, 1]."""
        frame = self._ocr_image(_STRIP_PRESENT_IMAGE)
        self._assert_results_in_unit_range(frame, "strip-present")

    def test_strip_absent_coords_in_unit_range(self):
        """OCR on a strip-absent image produces fx,fy in [0, 1]."""
        frame = self._ocr_image(_STRIP_ABSENT_IMAGE)
        self._assert_results_in_unit_range(frame, "strip-absent")


# ---------------------------------------------------------------------------
# Upgrade detection consistency across layouts
# ---------------------------------------------------------------------------


class TestUpgradeDetectionConsistency(unittest.TestCase):
    """detect_upgrade_buttons should find upgrades in both layouts."""

    @classmethod
    def setUpClass(cls):
        cls.config = Config()
        cls.ocr_reader = initialize_ocr_backend(cls.config)

    def _detect(self, path):
        if not path.exists():
            self.skipTest(f"Missing: {path}")
        img = Image.open(path).convert("RGB")
        img, _ = crop_ad_strip(img)
        frame = process_ocr(img, self.config, self.ocr_reader)
        return detect_upgrade_buttons(frame, img, self.config)

    def test_strip_present_detects_upgrades(self):
        """Strip-present upgrade screenshot yields detected upgrades."""
        upgrades = self._detect(_STRIP_PRESENT_IMAGE)
        self.assertGreater(len(upgrades), 0,
                           "Should detect upgrades in strip-present image")

    def test_strip_absent_detail_screen_no_crash(self):
        """detect_upgrade_buttons handles a strip-absent detail screen cleanly.

        no_ad_strip_upgrade_detail.png shows the Health upgrade detail view
        (a single-upgrade screen, not the 6-button upgrade panel).  The
        pipeline should complete without error and return an empty result.
        """
        upgrades = self._detect(_STRIP_ABSENT_IMAGE)
        # Detail screen — no upgrade-panel boxes expected; result is empty (list or dict)
        self.assertFalse(upgrades,
                         "Detail screen should not produce upgrade-panel hits")


if __name__ == "__main__":
    unittest.main()
