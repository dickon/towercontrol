# TowerControl

Visual automation for mobile games running in BlueStacks.  
Uses **screen capture + OCR only** — no ADB, no memory reading, no packet inspection.

## Architecture

```
run.py                  ← entry point
capture/window.py       ← find BlueStacks window, grab screenshots (mss + win32)
ocr/engine.py           ← pytesseract / easyocr with game-tuned preprocessing
input/controller.py     ← mouse clicks, scrolls, drags via pyautogui
state/machine.py        ← state machine, tracks screens/resources/elements
scanner/ui_scanner.py   ← tabs through the UI, scrolls, builds full state
games/base.py           ← abstract game definition (tab layout, screen IDs, element classifier)
games/tower_idle.py     ← Tower Idle concrete definition
strategy/base.py        ← abstract strategy (decide next action from state)
strategy/tower_idle.py  ← Tower Idle strategy (upgrade heroes, dismiss dialogs, etc.)
automation/loop.py      ← main tick loop tying everything together
web/server.py           ← FastAPI dashboard + REST API + WebSocket
web/templates/index.html
web/static/app.js
web/static/style.css
config.py               ← all tuneable knobs in one place
```

## Prerequisites

1. **Python 3.10+**
2. **Tesseract OCR** (if using the default `pytesseract` backend):
   - Download: <https://github.com/UB-Mannheim/tesseract/wiki>
   - Add `tesseract.exe` to your `PATH`, or set `TESSERACT_CMD` env var.
   - Or pass `--ocr easyocr` to skip Tesseract (downloads a PyTorch model on first run).
3. **BlueStacks** running a supported game in a visible window.

## Quick start

```bash
cd towercontrol
pip install -r requirements.txt
python run.py
```

This will:
- Find the BlueStacks window automatically
- Start capturing, OCR-ing, and playing
- Open a dashboard at **http://127.0.0.1:7700**

### Command-line options

| Flag | Default | Description |
|------|---------|-------------|
| `--game` | `tower_idle` | Game definition to load |
| `--window-title` | `BlueStacks` | Window title substring |
| `--ocr` | `pytesseract` | OCR backend (`pytesseract` or `easyocr`) |
| `--lang` | `eng` | OCR language |
| `--confidence` | `40` | Discard OCR results below this % confidence |
| `--tick` | `1.0` | Seconds between automation ticks |
| `--host` | `127.0.0.1` | Web server bind address |
| `--port` | `7700` | Web server port |
| `--no-auto` | off | Start the dashboard without auto-starting the bot |

## Web dashboard

- **Live view** of the game with OCR bounding boxes overlaid
- **Click to inject**: click anywhere on the game image to send a manual click
- **Start / Pause / Resume / Stop** the automation loop
- **Full Scan** button to force a re-scan of all tabs
- **Strategy parameters** are live-editable (toggle booleans, change numbers)
- **Resource readout**, current screen, element count, and action log
- Communicates via WebSocket for real-time updates (~1 fps)

## Adding a new game

1. Create `games/my_game.py` subclassing `GameDefinition`:
   - Define tabs (names, click positions, scroll regions)
   - Define screen-identification keywords
   - Define resource-extraction regex patterns
   - Implement `classify_element()` to sort OCR hits into buttons/labels/values

2. Create `strategy/my_game.py` subclassing `Strategy`:
   - Implement `decide(state) → Action`
   - Define tuneable parameters + `get_parameter_schema()`

3. Register in `run.py`'s `build_components()`.

## How it works

1. **Capture**: screenshot the BlueStacks window region every tick
2. **Preprocess**: resize, grayscale, adaptive threshold, denoise
3. **OCR**: extract text + bounding boxes from the processed image
4. **Classify**: game definition turns OCR results into typed UI elements (button, tab, resource, label)
5. **State update**: state machine determines current screen, updates resources
6. **Strategy**: inspects the state and returns an `Action` (click, scroll, scan, wait)
7. **Execute**: input controller performs the action via pyautogui
8. **Repeat**

The **scanner** periodically does a full pass: clicks each tab, scrolls through its content, merges all discovered elements into a complete state snapshot.

## Notes

- The BlueStacks window must be **visible** (not minimised) for screen capture to work.
- `pyautogui` moves the real mouse cursor. Don't move the mouse while the bot is clicking.
- All coordinates are relative to the captured window region — no hard-coded screen positions.
- The `FAILSAFE` is on: move your mouse to the **top-left corner** of the screen to abort.
