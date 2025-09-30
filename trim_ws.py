# .pre-commit-scripts/trim_ws.py
# Minimal, quiet trailing-whitespace fixer for text-like files.

import sys
from pathlib import Path
import re

EXIT_CODE = 0
TRAILING_WS = re.compile(r"[ \t]+(\r?\n)")


def is_small_text_file(p: Path) -> bool:
    # Adjust size threshold if needed
    if not p.is_file():
        return False
    if p.suffix.lower() in {
        ".ipynb",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".svg",
        ".pdf",
        ".zip",
        ".bin",
        ".pt",
        ".onnx",
        ".csv",
        ".tsv",
        ".jsonl",
        ".ndjson",
    }:
        return False
    # skip anything larger than ~2 MB
    try:
        if p.stat().st_size > 2_000_000:
            return False
    except OSError:
        return False
    return True


def fix_file(p: Path) -> bool:
    """Return True if modified."""
    try:
        text = p.read_text(encoding="utf-8")
    except Exception:
        return False  # non-UTF8 or unreadable, skip quietly

    fixed = TRAILING_WS.sub(r"\1", text)
    if fixed != text:
        p.write_text(fixed, encoding="utf-8", newline="")
        return True
    return False


def main(paths):
    global EXIT_CODE
    changed = []
    for fp in paths:
        p = Path(fp)
        if not is_small_text_file(p):
            continue
        if fix_file(p):
            changed.append(str(p))
    # Quiet output, only list changed filenames
    if changed:
        print("trim_ws modified:", *changed, sep="\n  ")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
