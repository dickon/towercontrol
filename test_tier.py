"""Tests for tier detection via update_tier_from_frame."""
import pytest
from pathlib import Path
from PIL import Image

import towercontrol_functional as tc

IMAGES = Path(__file__).parent / "test_images"


def _run(image_path: Path) -> tc.Optional[int]:
    """OCR an image, run tier detection, return the detected tier."""
    config = tc.Config()
    img = Image.open(image_path)
    frame = tc.process_ocr(img, config, ocr_reader=None)

    # Wire up a fresh global ctx so update_tier_from_frame can write to it
    tc.ctx = tc.RuntimeContext(config=config)
    tc.update_tier_from_frame(frame)
    return tc.ctx.game_state.tier


# ---------------------------------------------------------------------------
# Images with a definite tier number in the HUD
# ---------------------------------------------------------------------------

def test_tier_def_abs():
    assert _run(IMAGES / "def_abs.png") == 4


def test_tier_unexpected_upgrades_1():
    assert _run(IMAGES / "unexpected_upgrades_1771264045.png") == 5


def test_tier_unexpected_upgrades_2():
    assert _run(IMAGES / "unexpected_upgrades_1771282888.png") == 5


# ---------------------------------------------------------------------------
# Perk screens — tier widget not visible, state must not be updated
# ---------------------------------------------------------------------------

def test_tier_perk_rows_1():
    assert _run(IMAGES / "perk_rows_20260216_105227.png") is None


def test_tier_perk_rows_2():
    assert _run(IMAGES / "perk_rows_20260216_130242.png") is None


# ---------------------------------------------------------------------------
# Special cases — OCR may or may not see the tier on these screens
# ---------------------------------------------------------------------------

def test_tier_battle():
    """Home / battle lobby screen.  Tier 4 may or may not be visible to OCR."""
    tier = _run(IMAGES / "battle.png")
    assert tier in (None, 4)


def test_tier_killed_by_screen():
    """Post-game GAME STATS screen.  Tier 4 may or may not be visible to OCR."""
    tier = _run(IMAGES / "killed_by_screen.png")
    assert tier in (None, 4)
