# TowerControl - Functional Style Edition

This is a complete rewrite of the TowerControl game automation system in functional programming style, consolidated into a single file.

## Key Changes from Original

### Architecture

**Original**: Object-oriented with 9+ modules, classes with methods and internal state
**Functional**: Single file (~1000 lines), pure functions with immutable data structures

### Functional Programming Principles Applied

1. **Pure Functions**: Most logic is in pure functions that don't modify state
   - `identify_screen()`, `classify_ocr_result()`, `build_screen_state()`
   - `decide_action()`, `annotate_image()`, `state_to_dict()`

2. **Immutable Data**: Uses `@dataclass(frozen=True)` and tuple collections
   - `Config`, `OCRResult`, `OCRFrame`, `UIElement`, `ScreenState`, `GameState`, `Action`
   - State updates return new instances rather than mutating

3. **Separation of Pure and Impure**: Clear distinction between pure logic and side effects
   - Pure: OCR processing, state analysis, decision making
   - Impure: Window capture, input execution, web API

4. **Function Composition**: Complex operations built from simple functions
   - `scan_current_screen()` = `capture_window()` → `process_ocr()`
   - `automation_loop_tick()` orchestrates the full pipeline

5. **Minimal Mutable State**: Only `RuntimeContext` holds mutable state
   - Replaces scattered class instance variables
   - Centralized, explicit, easy to reason about

6. **Higher-Order Functions**: Functions as parameters
   - `execute_action()` takes `scanner_fn` and `full_scan_fn` callbacks

### Code Organization

```
towercontrol_functional.py (single file organized into sections):
├── Configuration (immutable Config dataclass)
├── Data Structures (immutable dataclasses & enums)
├── Pure Functions
│   ├── Window Capture (find_window, capture_window)
│   ├── OCR (preprocess_image, run_*_ocr, process_ocr)
│   ├── State Machine (identify_screen, classify_ocr_result, etc.)
│   ├── Strategy (decide_action)
│   └── Rendering (annotate_image, state_to_dict)
├── Side-Effect Functions
│   ├── Input (execute_click, execute_swipe, execute_action)
│   └── Scanning (scan_current_screen, perform_full_scan)
├── Runtime Context (minimal mutable state container)
├── Web API (FastAPI app factory)
└── Main (initialization & composition)
```

## Usage

### Basic

```bash
python towercontrol_functional.py
```

Dashboard at http://127.0.0.1:7700

### Command-Line Options

```bash
python towercontrol_functional.py --help

Options:
  --window-title TEXT  Window title pattern (default: BlueStacks)
  --ocr TEXT          OCR backend: pytesseract | easyocr
  --lang TEXT         OCR language (default: eng)
  --port INT          Web server port (default: 7700)
  --host TEXT         Web server host (default: 127.0.0.1)
  --no-auto           Don't auto-start the bot
```

### Safety

**Input is DISABLED by default** for safety. The bot will:
- ✅ Capture screenshots
- ✅ Run OCR
- ✅ Analyze game state
- ❌ NOT send clicks/input to the game

To enable input, make a POST request:
```bash
curl -X POST http://127.0.0.1:7700/api/enable_input \
  -H "Content-Type: application/json" \
  -d '{"enabled": true}'
```

Or add a button in the web UI.

## Dependencies

Same as original:
```bash
pip install -r requirements.txt
```

## Comparison

| Aspect | Original (OOP) | Functional |
|--------|---------------|------------|
| Files | 30+ files | 1 file |
| Lines | ~2500 | ~1000 |
| Classes | 15+ | 0 |
| Mutable State | Scattered | 1 container |
| Testability | Complex mocking | Pure functions |
| Readability | Navigate modules | Linear sections |
| Maintainability | Split concerns | Single source |

## Functional Programming Benefits

✅ **Easier Testing**: Pure functions are trivial to unit test  
✅ **Easier Reasoning**: No hidden state mutations  
✅ **Better Composability**: Functions combine cleanly  
✅ **Fewer Bugs**: Immutability prevents whole classes of errors  
✅ **Simpler Deployment**: One file, no package structure  
✅ **Learning**: Great example of FP in Python  

## Limitations

- Still uses `RuntimeContext` for mutable state (threading, window cache)
- Web server inherently stateful (FastAPI endpoints)
- Side effects (screen capture, OCR, input) can't be pure
- Python isn't a pure FP language (no tail call optimization, etc.)

## Next Steps

To make it even more functional:
1. Use `toolz` or `fn.py` for functional utilities
2. Implement State monad for explicit state threading
3. Use `returns` library for Railway-Oriented Programming
4. Switch to purely functional OCR mock for testing

But this version strikes a good balance between FP principles and practicality.

---

**Original modular version**: Use `run.py`  
**Functional single-file version**: Use `towercontrol_functional.py`
