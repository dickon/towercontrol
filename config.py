"""Global configuration for TowerControl."""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "web" / "templates"
STATIC_DIR = BASE_DIR / "web" / "static"
SCREENSHOTS_DIR = BASE_DIR / "screenshots"
SCREENSHOTS_DIR.mkdir(exist_ok=True)

# ── Window capture ─────────────────────────────────────────────────────
WINDOW_TITLE_PATTERN = "BlueStacks"  # substring match
CAPTURE_INTERVAL = 0.5               # seconds between captures

# ── OCR ────────────────────────────────────────────────────────────────
OCR_ENGINE = "pytesseract"            # "pytesseract" or "easyocr"
OCR_LANG = "eng"
OCR_CONFIDENCE_THRESHOLD = 40.0       # discard results below this %
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ── Input ──────────────────────────────────────────────────────────────
CLICK_PAUSE = 0.08                    # seconds after each click
SCROLL_AMOUNT = 3                     # scroll wheel ticks
INPUT_DELAY = 0.15                    # delay between compound actions

# ── Automation ─────────────────────────────────────────────────────────
LOOP_TICK = 1.0                       # main loop interval (seconds)
FULL_SCAN_INTERVAL = 30.0             # full UI rescan every N seconds
ACTION_COOLDOWN = 0.5                 # min seconds between actions

# ── Web server ─────────────────────────────────────────────────────────
WEB_HOST = "127.0.0.1"
WEB_PORT = 7700

# ── Game ───────────────────────────────────────────────────────────────
ACTIVE_GAME = "tower_idle"

# ── Tower Idle Game Layout ─────────────────────────────────────────────
# Tab positions (assuming ~540x960 portrait layout in BlueStacks)
TAB_Y = 920
TAB_POSITIONS = {
    "battle":  (70,  TAB_Y),
    "upgrades": (170, TAB_Y),
    "shop":    (370, TAB_Y),
    "more":    (470, TAB_Y),
}
SCROLL_REGION = (20, 200, 500, 650)
