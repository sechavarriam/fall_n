#!/usr/bin/env python3
"""Small helpers to normalize Python launcher strings across Windows/Linux."""

from __future__ import annotations

import os
import shlex
import shutil
import sys
from pathlib import Path


def python_launcher_command(raw: str | None) -> list[str]:
    """Return a process prefix for a configured Python launcher string.

    The validation scripts often persist a launcher as a single CLI string.
    On Windows that string may be a quoted absolute path with spaces, while on
    other setups it may be something like ``py -3.12``. This helper accepts
    both forms and returns a process-ready argument list.
    """

    if raw is None or not raw.strip():
        return [sys.executable]

    text = raw.strip()
    unquoted = text[1:-1] if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'" else text
    direct_path = Path(unquoted)
    if direct_path.exists():
        return [str(direct_path)]

    split_variants: list[list[str]] = []
    for posix in ((os.name != "nt"), False):
        try:
            tokens = shlex.split(text, posix=posix)
        except ValueError:
            continue
        if tokens:
            split_variants.append(tokens)

    for tokens in split_variants:
        first = tokens[0].strip("\"'")
        if Path(first).exists():
            return [str(Path(first)), *tokens[1:]]
        if shutil.which(first):
            return [first, *tokens[1:]]

    return [text]
