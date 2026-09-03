"""Plot styling shared by every figure (CLAUDE.md Sec. 8, "Plotting").

Line styles are keyed by randomness channel and match the existing reference
figure ``figures/Randoms_vs__Control.png``. Keep them consistent -- the whole
point is that figures from different runs can be laid side by side.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import matplotlib.pyplot as plt

#: Publication DPI floor (CLAUDE.md Sec. 8).
DPI: int = 300

FIGURES_DIR: Path = Path(__file__).resolve().parents[2] / "figures"


class ChannelStyle(TypedDict):
    label: str
    color: str
    linestyle: str


#: Channel -> line style. ``label`` is the shorthand used in the existing
#: figures (CLAUDE.md Sec. 4), not the formal name.
CHANNEL_STYLE: dict[str, ChannelStyle] = {
    "pure": {"label": "Pure QW", "color": "blue", "linestyle": "solid"},
    "discrete_coin": {"label": "Jittered", "color": "red", "linestyle": "dashed"},
    "continuous_coin": {
        "label": "Uniform Jittered",
        "color": "green",
        "linestyle": "dotted",
    },
    "random_translation": {
        "label": "Random Transform",
        "color": "purple",
        "linestyle": "dashdot",
    },
}


def save_figure(fig: plt.Figure, name: str, *, overwrite: bool = False) -> Path:
    """Save ``fig`` to ``figures/<name>.png`` at publication DPI.

    Refuses to overwrite an existing figure unless ``overwrite=True``, so a
    reference plot is never silently replaced (CLAUDE.md Sec. 8).
    """
    FIGURES_DIR.mkdir(exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists; pass overwrite=True to replace it"
        )
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    return path
