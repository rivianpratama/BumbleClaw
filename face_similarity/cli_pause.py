from __future__ import annotations

import os
import sys
from typing import Callable


def pause_if_requested(
    *,
    key_reader: Callable[[], str | None] | None = None,
    resume_reader: Callable[[str], str] = input,
) -> bool:
    key = (key_reader or read_key)()
    if key is None or key.lower() != "p":
        return False
    resume_reader("Paused. Press Enter to resume.")
    return True


def read_key() -> str | None:
    if os.name == "nt":
        try:
            import msvcrt

            return msvcrt.getwch() if msvcrt.kbhit() else None
        except OSError:
            return None
    if not sys.stdin.isatty():
        return None
    try:
        import select

        readable, _, _ = select.select([sys.stdin], [], [], 0)
        return sys.stdin.read(1) if readable else None
    except Exception:
        return None
