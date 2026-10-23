"""Shared drawing helpers."""
import os
from pathlib import Path

CODE_ALPHABET = "ABCDEFGHJKLMNPRSTUVWXYZ2345789"  # no I, O, Q, 0, 1, 6: avoids visually confusable characters
_SYSTEM_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def font_path() -> str:
    """DejaVu Sans: $JEVISION_FONT, else the system copy used for the reported results, else matplotlib's copy.

    Different font files can change glyph rendering slightly; the reported stimuli used the system file.
    """
    if os.environ.get("JEVISION_FONT"):
        return os.environ["JEVISION_FONT"]
    if Path(_SYSTEM_FONT).exists():
        return _SYSTEM_FONT
    import matplotlib

    return str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf")


def place(rng, boxes, w, h, size, tries=200):
    """A random square box of side `size` that keeps a 3 px margin from every box in `boxes`, or None."""
    for _ in range(tries):
        x, y = rng.randint(4, w - size - 4), rng.randint(4, h - size - 4)
        b = (x, y, x + size, y + size)
        if all(b[2] < o[0] - 3 or b[0] > o[2] + 3 or b[3] < o[1] - 3 or b[1] > o[3] + 3 for o in boxes):
            return b
    return None
