from __future__ import annotations

from pathlib import Path
import sys

from core import config


_DEBUG_LOG_PATH = Path(__file__).resolve().parent.parent / "debug.log"


def write_debug_line(message: str) -> None:
    """Mirror a debug/status line to stdout and ``debug.log`` when debug logging is enabled."""
    if not getattr(config, "DEBUG_WORKER_LOGGING", False):
        return
    text = str(message)
    print(text, file=sys.stdout, flush=True)
    with _DEBUG_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(text)
        handle.write("\n")

