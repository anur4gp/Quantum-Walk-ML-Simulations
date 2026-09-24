"""Shared figure conventions (CLAUDE.md Sec. 8, "Plotting").

Line styles are keyed by channel and match figures/Randoms_vs__Control.png, so
figures from different runs can be laid side by side.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import matplotlib.pyplot as plt

DPI: int = 300  # publication floor

FIGURES_DIR: Path = Path(__file__).resolve().parents[2] / "figures"


class ChannelStyle(TypedDict):
    label: str
    color: str
    linestyle: str


#: ``label`` is the project shorthand of CLAUDE.md Sec. 4, not the formal name.
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
    """Save to ``figures/<name>.png`` at publication DPI, refusing to overwrite."""
    FIGURES_DIR.mkdir(exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists; pass overwrite=True to replace it"
        )
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    return path
