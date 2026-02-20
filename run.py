"""
Hot-reloading runner for towercontrol_functional.

Run this instead of towercontrol_functional.py directly:
    python run.py [same args as usual]

When towercontrol_functional.py changes on disk, functions are reloaded
automatically without losing the running RuntimeContext (game state, wave
history, upgrade state, etc.).

How it works:
    importlib.reload() updates the module's __dict__ in-place.  The running
    automation_loop_run() loop looks up automation_loop_tick() by name on
    every iteration, so after a reload the next tick executes the new code.
    We save/restore ctx across the reload so all runtime state is preserved.

Limitations:
    - Adding new fields to RuntimeContext won't take effect on the live
      instance; restart for structural dataclass changes.
    - Module-level initialisation (template loading, OCR setup) does not
      re-run on reload, only function bodies are refreshed.
"""

import importlib
import sys
import threading

import pyautogui

import towercontrol_functional as mod


def _watch_and_reload(path: str) -> None:
    """Background thread: reload module whenever the source file changes."""
    from watchfiles import watch

    print(f"[hot-reload] Watching {path}")
    for _ in watch(path):
        print("[hot-reload] Change detected — reloading...")
        try:
            saved_ctx = getattr(mod, "ctx", None)
            importlib.reload(mod)
            if saved_ctx is not None:
                mod.ctx = saved_ctx
            print("[hot-reload] Reloaded OK. ctx preserved.")
        except Exception as exc:
            print(f"[hot-reload] Reload error: {exc}")


if __name__ == "__main__":
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.02

    src = mod.__file__

    watcher = threading.Thread(target=_watch_and_reload, args=(src,), daemon=True)
    watcher.start()

    # Delegate argv so all existing CLI args still work
    mod.main()
