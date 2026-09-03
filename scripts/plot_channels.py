#!/usr/bin/env python3
"""Plot P(x) for each randomness channel from a run config.

Thin entry point: parses args, loads the config, calls into ``src``. No
physics here (CLAUDE.md Sec. 8).

    python scripts/plot_channels.py configs/pure_vs_random.json [--save NAME]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from plotting.style import CHANNEL_STYLE, save_figure  # noqa: E402
from qw.observables import probability  # noqa: E402
from qw.walk import run_walk  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--save", metavar="NAME", help="save to figures/NAME.png")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text())

    fig, ax = plt.subplots()
    for channel, control_value in cfg["channels"].items():
        result = run_walk(
            cfg["n"],
            channel,
            theta_0=cfg["theta_0"],
            control_value=control_value,
            seed=cfg["seed"],
            parity=cfg.get("parity", "odd"),
            n_steps=cfg.get("n_steps"),
        )
        style = CHANNEL_STYLE[channel]
        ax.plot(
            result.positions,
            probability(result.psi_plus, result.psi_minus),
            label=style["label"],
            color=style["color"],
            linestyle=style["linestyle"],
        )

    n_steps = cfg.get("n_steps") or cfg["n"]
    ax.set_xlabel("x")
    ax.set_ylabel("P(x)")
    ax.set_title(f"Pure vs. Random Walks, {n_steps} Steps")
    ax.legend()

    if args.save:
        print(f"saved {save_figure(fig, args.save, overwrite=args.overwrite)}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
