#!/usr/bin/env python3
"""Test cases for upgrade detection using test images."""

import pytest
from pathlib import Path
from PIL import Image
from towercontrol_functional import (
    process_ocr,
    detect_upgrade_buttons,
    parse_number_with_suffix,
    UPGRADE_BUTTON_ROWS,
    UTILITY_UPGRADES,
    Config
)


class TestUtilityUpgrades1771264045:
    """Test upgrade detection on test image 1771264045 (UTILITY upgrades)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        self.config = Config()
        test_image = Path("test_images/unexpected_upgrades_1771264045.png")
        self.img = Image.open(test_image)
        self.frame = process_ocr(self.img, self.config)
        self.upgrades = detect_upgrade_buttons(self.frame, self.img, self.config)

    def test_image_loaded(self):
        """Verify test image loaded successfully."""
        assert self.img is not None
        assert self.img.size[0] > 0
        assert self.img.size[1] > 0

    def test_ocr_results_present(self):
        """Verify OCR produced results."""
        assert len(self.frame.results) > 0, "OCR should detect text in image"

    def test_upgrade_button_regions_have_ocr_results(self):
        """Verify OCR results are found in upgrade button regions."""
        y_tolerance = 0.08
        regions_with_results = 0

        for idx, y_frac, (x_min, x_max) in UPGRADE_BUTTON_ROWS:
            region_results = [
                r for r in self.frame.results
                if abs(r.fy - y_frac) < y_tolerance and x_min <= r.fx <= x_max
            ]
            if region_results:
                regions_with_results += 1

        assert regions_with_results > 0, "At least one upgrade region should have OCR results"

    def test_utility_keywords_detected(self):
        """Verify utility upgrade keywords are detected in OCR."""
        utility_keywords = ['recovery', 'amount', 'max', 'package', 'chance', 'enemy', 'attack', 'health', 'level', 'skip']
        detected_keywords = []

        for keyword in utility_keywords:
            matches = [r for r in self.frame.results if keyword in r.text.lower()]
            if matches:
                detected_keywords.append(keyword)

        assert len(detected_keywords) > 0, f"Should detect at least some utility keywords, found: {detected_keywords}"

    def test_upgrades_detected(self):
        """Verify upgrade buttons are detected."""
        assert len(self.upgrades) > 0, "Should detect at least one upgrade button"

    def test_upgrade_structure(self):
        """Verify upgrade info has correct structure."""
        for label, info in self.upgrades.items():
            assert isinstance(label, str), f"Label should be string, got {type(label)}"
            assert 'current_value' in info, f"Missing 'current_value' for {label}"
            assert 'cost' in info, f"Missing 'cost' for {label}"
            assert 'upgrades_to_purchase' in info, f"Missing 'upgrades_to_purchase' for {label}"
            assert 'cell_color' in info, f"Missing 'cell_color' for {label}"
            assert 'cell_color_name' in info, f"Missing 'cell_color_name' for {label}"
            assert isinstance(info['cell_color_name'], str), f"cell_color_name for {label} should be string"
            assert 'label_position' in info, f"Missing 'label_position' for {label}"
            assert 'button_position' in info, f"Missing 'button_position' for {label}"

    def test_upgrade_costs_parsed(self):
        """Verify upgrade costs are properly parsed as floats or None."""
        for label, info in self.upgrades.items():
            cost = info['cost']
            assert cost is None or isinstance(cost, float), \
                f"Cost for {label} should be float or None, got {type(cost)}: {cost}"

    def test_cell_colors_are_rgb_tuples(self):
        """Verify cell colors are RGB tuples."""
        for label, info in self.upgrades.items():
            color = info['cell_color']
            assert isinstance(color, tuple), f"Color for {label} should be tuple, got {type(color)}"
            assert len(color) == 3, f"Color for {label} should have 3 values (RGB), got {len(color)}"
            assert all(isinstance(c, int) and 0 <= c <= 255 for c in color), \
                f"Color for {label} should be 3 ints in range 0-255, got {color}"

    def test_label_positions_are_fractional(self):
        """Verify label positions are fractional coordinates."""
        for label, info in self.upgrades.items():
            pos = info['label_position']
            assert isinstance(pos, tuple), f"Label position for {label} should be tuple, got {type(pos)}"
            assert len(pos) == 2, f"Label position for {label} should have 2 values (fx, fy), got {len(pos)}"
            fx, fy = pos
            assert 0 <= fx <= 1, f"Label fx for {label} should be 0-1, got {fx}"
            assert 0 <= fy <= 1, f"Label fy for {label} should be 0-1, got {fy}"

    def test_expected_upgrades_detected(self):
        """Verify some expected upgrades are detected."""
        # We expect these upgrades to be visible in the image
        expected_labels = ['Recovery Amount', 'Max Recovery', 'Package Chance',
                          'Enemy Attack Level Skip', 'Enemy Health Level Skip']

        detected_labels = list(self.upgrades.keys())

        # Should detect at least 2 of the expected upgrades
        matches = [label for label in expected_labels if label in detected_labels]
        assert len(matches) >= 2, \
            f"Should detect at least 2 expected upgrades. Expected: {expected_labels}, Detected: {detected_labels}, Matches: {matches}"

    def test_recovery_amount_if_detected(self):
        """Verify Recovery Amount upgrade details if detected."""
        if 'Recovery Amount' in self.upgrades:
            info = self.upgrades['Recovery Amount']
            # Cost should be around 340310
            if info['cost'] is not None:
                assert 300000 < info['cost'] < 400000, \
                    f"Recovery Amount cost should be ~340310, got {info['cost']}"


class TestUtilityUpgrades1771351783:
    """Test upgrade detection on test image 1771351783 (UTILITY upgrades with specific values).

    These tests document the IDEAL expected behavior. Many will fail until OCR improvements are made.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        self.config = Config()
        test_image = Path("test_images/unexpected_upgrades_1771351783.png")
        self.img = Image.open(test_image)
        self.frame = process_ocr(self.img, self.config)
        self.upgrades = detect_upgrade_buttons(self.frame, self.img, self.config)

    def test_package_chance_detected(self):
        """Verify Package Chance upgrade is detected."""
        assert 'Package Chance' in self.upgrades, \
            f"Package Chance should be detected. Found: {list(self.upgrades.keys())}"

    def test_package_chance_is_dark(self):
        """Verify Package Chance has very dark background color (near-black with warm tint)."""
        if 'Package Chance' not in self.upgrades:
            pytest.skip("Package Chance not detected")

        color = self.upgrades['Package Chance']['cell_color']
        brightness = sum(color) / 3
        # The MAX cell background is very dark (near-black), brightness should be well under 100
        assert brightness < 100, f"Should be very dark (near-black), got RGB={color} with brightness={brightness:.1f}"

    def test_package_chance_value_is_63_4_percent(self):
        """Verify Package Chance current value is 63.4%.

        FAILING: OCR not detecting the percentage value from the cell.
        Need to improve value extraction from upgrade cells.
        """
        assert 'Package Chance' in self.upgrades, "Package Chance should be detected"

        current_value = self.upgrades['Package Chance']['current_value']
        assert current_value is not None, "Package Chance current value should be detected"

        # Value could be stored as 63.4 or 0.634 depending on parsing
        assert 60 < current_value < 70 or 0.6 < current_value < 0.7, \
            f"Package Chance value should be ~63.4% (or 0.634), got {current_value}"

    def test_package_chance_is_max(self):
        """Verify Package Chance cost is MAX (None)."""
        assert 'Package Chance' in self.upgrades, "Package Chance should be detected"

        cost = self.upgrades['Package Chance']['cost']
        assert cost is None, \
            f"Package Chance should be at MAX (cost=None), got {cost}"

    def test_enemy_attack_level_skip_detected(self):
        """Verify Enemy Attack Level Skip upgrade is detected."""
        assert 'Enemy Attack Level Skip' in self.upgrades, \
            f"Enemy Attack Level Skip should be detected. Found: {list(self.upgrades.keys())}"

    def test_enemy_attack_level_skip_is_dark_blue(self):
        """Verify Enemy Attack Level Skip has dark blue background color."""
        assert 'Enemy Attack Level Skip' in self.upgrades, "Enemy Attack Level Skip should be detected"

        color = self.upgrades['Enemy Attack Level Skip']['cell_color']
        r, g, b = color

        # Dark blue should have: B > R and B > G, and relatively low overall brightness
        assert b > r, f"Dark blue should have B > R, got RGB={color}"
        assert b > g, f"Dark blue should have B > G, got RGB={color}"
        brightness = (r + g + b) / 3
        assert brightness < 150, \
            f"Should be dark blue (low brightness), got RGB={color} with brightness={brightness}"

    def test_enemy_attack_level_skip_value_is_6_25_percent(self):
        """Verify Enemy Attack Level Skip current value is 6.25%.

        FAILING: OCR not detecting the percentage value from the cell.
        Need to improve value extraction from upgrade cells.
        """
        assert 'Enemy Attack Level Skip' in self.upgrades, "Enemy Attack Level Skip should be detected"

        current_value = self.upgrades['Enemy Attack Level Skip']['current_value']
        assert current_value is not None, "Enemy Attack Level Skip current value should be detected"

        # Value could be stored as 6.25 or 0.0625 depending on parsing
        assert 5 < current_value < 8 or 0.05 < current_value < 0.08, \
            f"Enemy Attack Level Skip value should be ~6.25% (or 0.0625), got {current_value}"

    def test_enemy_attack_level_skip_cost_is_315_43k(self):
        """Verify Enemy Attack Level Skip cost is $315.43K (315430).

        FAILING: OCR not detecting the cost - shows as MAX instead.
        Need to improve cost text detection in upgrade cells.
        """
        assert 'Enemy Attack Level Skip' in self.upgrades, "Enemy Attack Level Skip should be detected"

        cost = self.upgrades['Enemy Attack Level Skip']['cost']
        assert cost is not None, "Enemy Attack Level Skip should have a numeric cost (not MAX)"

        # Cost should be around 315,430 (315.43K)
        expected = 315430
        tolerance = 5000  # Allow ±5K tolerance for OCR variations
        assert expected - tolerance < cost < expected + tolerance, \
            f"Enemy Attack Level Skip cost should be ~315.43K (315430), got {cost}"

    def test_enemy_health_level_skip_detected(self):
        """Verify Enemy Health Level Skip upgrade is detected with full label.

        FAILING: Currently only detected as "Health" instead of full label.
        Label matching needs improvement for multi-word labels.
        """
        assert 'Enemy Health Level Skip' in self.upgrades, \
            f"Enemy Health Level Skip should be detected with full label. Found: {list(self.upgrades.keys())}"

    def test_enemy_health_level_skip_cost_is_353_28k(self):
        """Verify Enemy Health Level Skip cost is $353.28K (353280).

        FAILING: Label not detected correctly AND cost not detected.
        Multiple issues to fix: label matching and cost extraction.
        """
        assert 'Enemy Health Level Skip' in self.upgrades, "Enemy Health Level Skip should be detected"

        cost = self.upgrades['Enemy Health Level Skip']['cost']
        assert cost is not None, "Enemy Health Level Skip should have a numeric cost (not MAX)"

        # Cost should be around 353,280 (353.28K) - note image shows 353.21K
        expected = 353280
        tolerance = 5000  # Allow ±5K tolerance for OCR variations
        assert expected - tolerance < cost < expected + tolerance, \
            f"Enemy Health Level Skip cost should be ~353.28K (353280), got {cost}"


class TestDefenseUpgrades1771362963:
    """Test upgrade detection on test image 1771362963 (DEFENSE upgrades).

    Image shows: Health (1.81T, Max), Health Regen (1.06B/sec, $495.14M, x157),
    Defense % (91.46%, Max), Defense Absolute (974.70K, $496.86M, x794),
    Thorn Damage (108%, ???), Lifesteal (12.21%, ???).
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        self.config = Config()
        test_image = Path("test_images/unexpected_upgrades_1771362963.png")
        self.img = Image.open(test_image)
        self.frame = process_ocr(self.img, self.config)
        self.upgrades = detect_upgrade_buttons(self.frame, self.img, self.config)

    def test_health_regen_detected(self):
        """Verify Health Regen upgrade is detected."""
        assert 'Health Regen' in self.upgrades, \
            f"Health Regen should be detected. Found: {list(self.upgrades.keys())}"

    def test_health_regen_cost_is_495_14m(self):
        """Verify Health Regen cost is $495.14M (495140000).

        Image shows "$495.14M" as the upgrade cost.
        """
        assert 'Health Regen' in self.upgrades, "Health Regen should be detected"

        cost = self.upgrades['Health Regen']['cost']
        assert cost is not None, "Health Regen should have a numeric cost"

        expected = 495_140_000
        tolerance = 5_000_000  # Allow ±5M tolerance for OCR variations
        assert expected - tolerance < cost < expected + tolerance, \
            f"Health Regen cost should be ~$495.14M ({expected}), got {cost}"

    def test_health_regen_upgrades_to_purchase_is_157(self):
        """Verify Health Regen upgrades_to_purchase is 157.

        Image shows "x157" in the top-right corner of the Health Regen cell.
        """
        assert 'Health Regen' in self.upgrades, "Health Regen should be detected"

        count = self.upgrades['Health Regen']['upgrades_to_purchase']
        assert count == 157, \
            f"Health Regen upgrades_to_purchase should be 157, got {count}"


class TestParseNumberWithSuffix:
    """Test cases for parse_number_with_suffix helper function."""

    def test_parse_simple_integer(self):
        """Test parsing simple integers."""
        assert parse_number_with_suffix("123") == 123.0
        assert parse_number_with_suffix("1000") == 1000.0

    def test_parse_decimal(self):
        """Test parsing decimal numbers."""
        assert parse_number_with_suffix("3.62") == 3.62
        assert parse_number_with_suffix("28.10") == 28.10

    def test_parse_with_k_suffix(self):
        """Test parsing numbers with K suffix."""
        assert parse_number_with_suffix("1.5K") == 1500.0
        assert parse_number_with_suffix("100K") == 100000.0

    def test_parse_with_m_suffix(self):
        """Test parsing numbers with M suffix."""
        assert parse_number_with_suffix("1.32M") == 1320000.0
        assert parse_number_with_suffix("5M") == 5000000.0

    def test_parse_with_b_suffix(self):
        """Test parsing numbers with B suffix."""
        assert parse_number_with_suffix("3.62B") == 3620000000.0
        assert parse_number_with_suffix("1B") == 1000000000.0

    def test_parse_with_t_suffix(self):
        """Test parsing numbers with T suffix."""
        assert parse_number_with_suffix("1.5T") == 1500000000000.0
        assert parse_number_with_suffix("2T") == 2000000000000.0

    def test_parse_with_commas(self):
        """Test parsing numbers with thousand separators."""
        assert parse_number_with_suffix("1,000") == 1000.0
        assert parse_number_with_suffix("340,310") == 340310.0

    def test_parse_max_returns_none(self):
        """Test that MAX returns None."""
        assert parse_number_with_suffix("MAX") is None
        assert parse_number_with_suffix("max") is None

    def test_parse_question_marks_returns_none(self):
        """Test that ??? returns None."""
        assert parse_number_with_suffix("???") is None

    def test_parse_invalid_returns_none(self):
        """Test that invalid strings return None."""
        assert parse_number_with_suffix("abc") is None
        assert parse_number_with_suffix("") is None
        assert parse_number_with_suffix("123abc") is None


if __name__ == "__main__":
    # Allow running as script for quick debugging
    pytest.main([__file__, "-v"])
