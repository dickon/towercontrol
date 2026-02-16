#!/usr/bin/env python3
"""
Test cases for OCR and perk text detection in TowerControl.

Tests verify that perk text is correctly extracted from OCR results
and matched against the perk choice patterns.
"""

import re
import unittest
from typing import List, Tuple

# Import from main module
from towercontrol_functional import OCRResult, OCRFrame, PERK_CHOICES, PERK_ROWS


class TestPerkDetection(unittest.TestCase):
    """Test cases for perk detection from OCR results."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Standard image size for testing
        self.image_size = (1920, 1080)
    
    def create_ocr_result(self, text: str, fx: float, fy: float, confidence: float = 0.9) -> OCRResult:
        """Helper to create an OCRResult at fractional coordinates.
        
        Args:
            text: The OCR text
            fx: Fractional X position (0.0-1.0)
            fy: Fractional Y position (0.0-1.0)
            confidence: OCR confidence score
        
        Returns:
            OCRResult with calculated pixel bbox
        """
        width, height = self.image_size
        # Calculate center pixel position
        cx = int(fx * width)
        cy = int(fy * height)
        
        # Create a small bbox around the center
        # Assume text is about 100px wide and 30px tall
        x = max(0, cx - 50)
        y = max(0, cy - 15)
        w = 100
        h = 30
        
        return OCRResult(
            text=text,
            bbox=(x, y, w, h),
            confidence=confidence,
            image_size=self.image_size
        )
    
    def simulate_perk_detection(self, ocr_results: List[OCRResult]) -> dict:
        """Simulate the perk detection logic from towercontrol_functional.py.
        
        This replicates the core perk detection algorithm to test it independently.
        
        Args:
            ocr_results: List of OCRResult objects
        
        Returns:
            dict with 'perk_text_join' and 'perk_text_priority'
        """
        # Collect perk text by row (slot index)
        perk_text = {}
        
        for r in ocr_results:
            # Check each perk row position
            for (row, dy) in PERK_ROWS:
                # This matches the logic in towercontrol_functional.py lines 1470-1473
                if abs(r.fy - dy) < 0.05 and r.fx > 0.45 and r.fx < 0.78:
                    perk_text.setdefault(row, [])
                    perk_text[row].append(r.text)
        
        # Filter out rows that contain "Selected" (already chosen perks)
        for i in range(5):
            if 'Selected' in ' '.join(perk_text.get(i, [])):
                perk_text = {k: v for k, v in perk_text.items() if k < i}
        
        # Join text fragments for each row
        perk_text_join = {row: " ".join(texts) for row, texts in perk_text.items()}
        
        # Match each perk text against known patterns and prioritize
        perk_text_priority = []
        for row, text in perk_text_join.items():
            for idx, choice_pattern in enumerate(PERK_CHOICES):
                if re.search(choice_pattern, text, re.IGNORECASE):
                    perk_text_priority.append((row, choice_pattern, idx))
                    break
        
        # Sort by priority (choice index)
        perk_text_priority.sort(key=lambda x: x[2])
        
        return {
            'perk_text_join': perk_text_join,
            'perk_text_priority': perk_text_priority
        }
    
    def test_golden_tower_bonus_in_slot_2(self):
        """Test case: Detect 'Golden Tower bonus x1.5' in slot index 2.
        
        This test verifies that:
        1. OCR text fragments are correctly grouped by row/slot
        2. The Golden Tower pattern matches the text
        3. The text is detected in the correct slot (index 2)
        """
        # Create OCR results simulating "Golden Tower bonus x1.5" detected in slot 2
        # Slot 2 has y position 0.473 (from PERK_ROWS)
        slot_2_y = 0.473
        
        # Simulate OCR detecting text fragments in slot 2
        # OCR often splits text into multiple results
        ocr_results = [
            self.create_ocr_result("Golden", 0.50, slot_2_y, 0.95),
            self.create_ocr_result("Tower", 0.56, slot_2_y, 0.93),
            self.create_ocr_result("bonus", 0.62, slot_2_y, 0.91),
            self.create_ocr_result("x1.5", 0.68, slot_2_y, 0.89),
        ]
        
        # Add some other perks in different slots to make it realistic
        slot_0_y = 0.273
        slot_1_y = 0.373
        ocr_results.extend([
            self.create_ocr_result("Damage", 0.55, slot_0_y, 0.92),
            self.create_ocr_result("x2.0", 0.62, slot_0_y, 0.90),
            self.create_ocr_result("Max", 0.52, slot_1_y, 0.93),
            self.create_ocr_result("Health", 0.60, slot_1_y, 0.91),
            self.create_ocr_result("x1.8", 0.68, slot_1_y, 0.89),
        ])
        
        # Run perk detection
        result = self.simulate_perk_detection(ocr_results)
        
        # Verify slot 2 text was joined correctly
        self.assertIn(2, result['perk_text_join'], "Slot 2 should be detected")
        slot_2_text = result['perk_text_join'][2]
        self.assertEqual(slot_2_text, "Golden Tower bonus x1.5")
        
        # Verify the pattern matched
        golden_tower_pattern = r'Golden Tower Bonus( x[\d\.]+)?'
        self.assertTrue(
            re.search(golden_tower_pattern, slot_2_text, re.IGNORECASE),
            f"Text '{slot_2_text}' should match Golden Tower pattern"
        )
        
        # Verify slot 2 appears in priority list
        slot_2_priorities = [p for p in result['perk_text_priority'] if p[0] == 2]
        self.assertEqual(len(slot_2_priorities), 1, "Slot 2 should have exactly one priority match")
        
        slot, pattern, priority_idx = slot_2_priorities[0]
        self.assertEqual(slot, 2, "Match should be for slot 2")
        self.assertEqual(pattern, golden_tower_pattern, "Should match Golden Tower pattern")
        
        print(f"✓ Test passed: Golden Tower bonus x1.5 detected in slot 2")
        print(f"  Detected text: '{slot_2_text}'")
        print(f"  Pattern: {pattern}")
        print(f"  Priority index: {priority_idx}")
    
    def test_golden_tower_pattern_variations(self):
        """Test that the Golden Tower pattern matches various text formats."""
        pattern = r'Golden Tower Bonus( x[\d\.]+)?'
        
        # Test cases: (text, should_match)
        test_cases = [
            ("Golden Tower Bonus", True),
            ("Golden Tower Bonus x1.5", True),
            ("Golden Tower Bonus x2.0", True),
            ("Golden Tower Bonus x10.5", True),
            ("golden tower bonus", True),  # Case insensitive
            ("GOLDEN TOWER BONUS x1.5", True),  # All caps
            ("Golden Tower bonus x1.5", True),  # Mixed case
            ("Tower Bonus", False),  # Incomplete
            ("Golden Bonus", False),  # Missing "Tower"
        ]
        
        for text, should_match in test_cases:
            match = re.search(pattern, text, re.IGNORECASE)
            if should_match:
                self.assertIsNotNone(match, f"Pattern should match '{text}'")
            else:
                self.assertIsNone(match, f"Pattern should not match '{text}'")
    
    def test_multiple_perks_priority_ordering(self):
        """Test that multiple perks are correctly prioritized."""
        # Create perks in different slots
        ocr_results = [
            # Slot 0: Damage (priority 3)
            self.create_ocr_result("Damage", 0.55, 0.273, 0.92),
            self.create_ocr_result("x2.0", 0.62, 0.273, 0.90),
            
            # Slot 1: Golden Tower (priority 5)
            self.create_ocr_result("Golden", 0.50, 0.373, 0.95),
            self.create_ocr_result("Tower", 0.56, 0.373, 0.93),
            self.create_ocr_result("Bonus", 0.62, 0.373, 0.91),
            self.create_ocr_result("x1.5", 0.68, 0.373, 0.89),
            
            # Slot 2: Perk Wave Requirement (priority 0)
            self.create_ocr_result("Perk", 0.50, 0.473, 0.94),
            self.create_ocr_result("Wave", 0.56, 0.473, 0.93),
            self.create_ocr_result("Requirement", 0.65, 0.473, 0.92),
            self.create_ocr_result("-5%", 0.73, 0.473, 0.90),
        ]
        
        result = self.simulate_perk_detection(ocr_results)
        
        # Verify all three slots detected
        self.assertEqual(len(result['perk_text_join']), 3, "Should detect 3 perk slots")
        
        # Verify priority ordering
        self.assertEqual(len(result['perk_text_priority']), 3, "Should have 3 prioritized perks")
        
        # First priority should be Perk Wave Requirement (index 0 in PERK_CHOICES)
        self.assertEqual(result['perk_text_priority'][0][0], 2, "Highest priority should be slot 2")
        
        print(f"✓ Test passed: Multiple perks correctly prioritized")
        for slot, pattern, priority in result['perk_text_priority']:
            text = result['perk_text_join'][slot]
            print(f"  Slot {slot} (priority {priority}): '{text}'")
    
    def test_ocr_result_fractional_coordinates(self):
        """Test that OCRResult fractional coordinates are calculated correctly."""
        # Create a result at known fractional position
        result = self.create_ocr_result("Test", 0.5, 0.473, 0.9)
        
        # Verify fractional coordinates
        self.assertAlmostEqual(result.fx, 0.5, places=2, msg="Fractional X should be 0.5")
        self.assertAlmostEqual(result.fy, 0.473, places=2, msg="Fractional Y should be 0.473")
        
        # Verify is_near works for slot detection
        # Slot 2 is at y=0.473 with tolerance 0.05
        self.assertTrue(
            abs(result.fy - 0.473) < 0.05,
            "Result should be within tolerance of slot 2 position"
        )
    
    def test_selected_perk_filtering(self):
        """Test that rows containing 'Selected' are filtered out correctly."""
        # Create perks with one already selected
        ocr_results = [
            # Slot 0: Selected perk (should be filtered)
            self.create_ocr_result("Selected", 0.50, 0.273, 0.95),
            self.create_ocr_result("Damage", 0.58, 0.273, 0.92),
            
            # Slot 1: Should also be filtered (< selected row)
            # This simulates behavior where rows above selected are kept
            self.create_ocr_result("Golden", 0.50, 0.373, 0.95),
            self.create_ocr_result("Tower", 0.56, 0.373, 0.93),
        ]
        
        result = self.simulate_perk_detection(ocr_results)
        
        # When "Selected" appears in slot 0, all slots >= 0 should be filtered
        # Actually checking the logic: it filters slots >= the selected one
        # Let me verify the actual behavior matches the code
        self.assertNotIn(0, result['perk_text_join'], "Slot 0 (with Selected) should be filtered")


class TestOCRPatternMatching(unittest.TestCase):
    """Test cases for OCR pattern matching against PERK_CHOICES."""
    
    def test_all_perk_patterns_compile(self):
        """Verify all perk choice regex patterns are valid."""
        for idx, pattern in enumerate(PERK_CHOICES):
            try:
                re.compile(pattern, re.IGNORECASE)
            except re.error as e:
                self.fail(f"Invalid regex pattern at index {idx}: {pattern}\nError: {e}")
    
    def test_golden_tower_is_in_perk_choices(self):
        """Verify Golden Tower pattern exists in PERK_CHOICES."""
        golden_tower_pattern = r'Golden Tower Bonus( x[\d\.]+)?'
        self.assertIn(
            golden_tower_pattern,
            PERK_CHOICES,
            "Golden Tower Bonus pattern should be in PERK_CHOICES"
        )
        
        # Find its priority index
        idx = PERK_CHOICES.index(golden_tower_pattern)
        print(f"✓ Golden Tower Bonus is at priority index {idx} in PERK_CHOICES")


def run_tests():
    """Run all tests and display results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestPerkDetection))
    suite.addTests(loader.loadTestsFromTestCase(TestOCRPatternMatching))
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    exit(run_tests())
