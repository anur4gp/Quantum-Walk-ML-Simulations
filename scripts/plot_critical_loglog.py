#!/usr/bin/env python3
"""Log-log of Delta_theta_c vs t_f, power-law fit, and a comparison table with Paper A.

Reads notes/critical_vs_time_results.json (from scripts/plot_critical_vs_time.py).
Fit: log Delta_theta_c = log A - alpha log t_f, ordinary least squares; the
uncertainty is the standard error of the slope, which like Paper A's Table I
reflects only the scatter about the line.

    python3 scripts/plot_critical_loglog.py [--overwrite]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RESULTS = ROOT / "notes" / "critical_vs_time_results.json"

# Paper A, Table I, discrete random rotation column: Delta_theta_c ~ N^-alpha.
PAPER_A_ALPHA: list[tuple[str, float, float]] = [
    ("SVM", 0.36, 0.04),
    ("MLP NN", 0.32, 0.02),
    ("CNN", 0.39, 0.07),
    ("MoI", 0.62, 0.02),
    ("Human", 0.32, 0.01),
    ("IPR", 0.46, 0.02),
]


def fit_power_law(t: np.ndarray, dc: np.ndarray) -> tuple[float, float, float]:
    """Return (alpha, alpha_err, A) for dc = A * t^-alpha."""
    coef, cov = np.polyfit(np.log(t), np.log(dc), 1, cov=True)
    return float(-coef[0]), float(np.sqrt(cov[0, 0])), float(np.exp(coef[1]))


def plot_loglog(t, dc, alpha, err, amp, cfg, overwrite) -> Path:
    import matplotlib.pyplot as plt

    from plotting.style import CHANNEL_STYLE, save_figure

    style = CHANNEL_STYLE[cfg["channel"]]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(t, dc, linestyle="none", marker="o", color=style["color"],
              label=f"MLP, {style['label']} ($\\Delta\\theta$)")
    tt = np.geomspace(t.min(), t.max(), 100)
    ax.loglog(tt, amp * tt**-alpha, color="black", linestyle="dashed",
              label=f"fit: $\\Delta\\theta_c \\propto t_f^{{-\\alpha}}$, "
                    f"$\\alpha = {alpha:.2f} \\pm {err:.2f}$")
    ax.set_xlabel(r"final time $t_f$ (steps)")
    ax.set_ylabel(r"$\Delta\theta_c$")
    ax.set_title(r"Critical $\Delta\theta_c$ vs final time $t_f$, log-log"
                 f"\n$\\theta_0=\\pi/6$, lattice $x=-N..N$ (no 0), $t_f=N-1$, seed {cfg['seed']}",
                 fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = save_figure(fig, "mlp_critical_vs_time_loglog", overwrite=overwrite)
    plt.close(fig)
    return path


def plot_table(alpha, err, t, dc, overwrite) -> Path:
    import matplotlib.pyplot as plt

    from plotting.style import save_figure

    rows = [["This work: MLP (Paper B lattice, seed 0)", f"{alpha:.2f} ± {err:.2f}"]]
    rows += [[f"Paper A: {name}", f"{a:.2f} ± {e:.2f}"] for name, a, e in PAPER_A_ALPHA]

    i490 = int(np.argmin(np.abs(t - 490)))
    point_rows = [
        [f"This work: MLP, t_f = {int(t[i490])}", f"{dc[i490]:.3f}"],
        ["Paper A, Fig. 1(b): critical example, N = 490", "0.035"],
        ["Paper A, Fig. 1(c): already localized, N = 490", "0.09"],
    ]

    fig, axes = plt.subplots(2, 1, figsize=(7.5, 5.2),
                             gridspec_kw={"height_ratios": [len(rows) + 1, len(point_rows) + 1]})
    for ax, data, header, title in (
        (axes[0], rows, ["Method", "Exponent α"],
         r"Scaling exponent α in $\Delta\theta_c \propto N^{-\alpha}$ — discrete random rotation, $\theta_0=\pi/6$"),
        (axes[1], point_rows, ["Source", "Δθ"],
         "Single critical values near N ≈ 500 (Paper A gives no table of Δθ_c, only these examples)"),
    ):
        ax.axis("off")
        tbl = ax.table(cellText=data, colLabels=header, loc="center",
                       cellLoc="left", colWidths=[0.72, 0.28])
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 1.35)
        for (r, _), cell in tbl.get_celld().items():
            if r == 0:
                cell.set_text_props(weight="bold")
                cell.set_facecolor("0.88")
            elif r == 1:
                cell.set_facecolor("#fde7e7")
        ax.set_title(title, fontsize=9)
    fig.tight_layout()
    path = save_figure(fig, "mlp_critical_comparison_table", overwrite=overwrite)
    plt.close(fig)
    return path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    data = json.loads(RESULTS.read_text())
    cfg, recs = data["config"], data["records"]
    recs = [r for r in recs if r["delta_theta_c"] is not None]
    t = np.array([r["t_f"] for r in recs], dtype=float)
    dc = np.array([r["delta_theta_c"] for r in recs], dtype=float)

    alpha, err, amp = fit_power_law(t, dc)
    print(f"alpha = {alpha:.3f} +- {err:.3f}   A = {amp:.3f}   ({t.size} points, seed {cfg['seed']})")
    print(f"saved {plot_loglog(t, dc, alpha, err, amp, cfg, args.overwrite)}")
    print(f"saved {plot_table(alpha, err, t, dc, args.overwrite)}")


if __name__ == "__main__":
    main()
