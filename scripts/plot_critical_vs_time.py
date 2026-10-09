#!/usr/bin/env python3
"""Critical Delta_theta_c vs final time t_f, one freshly trained MLP per lattice.

For each lattice size N (Paper B lattice: 2N sites, x = -N..N without 0, walk
run to t_f = N - 1 steps; Paper B Sec. II):

    1. one walk per sample, final-time P(x, t_f), discrete random coin
       theta_0 +- Delta_theta (Paper A Sec. II B 1 / Paper B Sec. III A)
    2. label 0 from the two-peak window, label 1 from the one-peak window,
       train Paper A's MLP (Sec. III B 2) on an 80/20 split
    3. feed walks at every Delta_theta on a grid over [0, theta_0], average
       P(deloc) over realizations, and take Delta_theta_c as the first grid
       point with P(deloc) < 0.5 (Paper A's MLP rule, Sec. III B 2)

Then plot t_f (x axis) vs Delta_theta_c (y axis) on linear axes.

    python3 scripts/plot_critical_vs_time.py [--sizes 30 50 80] [--overwrite]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analysis.critical import first_below_half, sustained_below_half  # noqa: E402
from data.generate import DELOCALIZED, make_dataset  # noqa: E402
from models.mlp import MLPClassifier  # noqa: E402
from qw.observables import probability  # noqa: E402
from qw.operators import THETA_0_DEFAULT as THETA_0  # noqa: E402
from qw.walk import max_steps, run_walk  # noqa: E402

CONFIG: dict = {
    "channel": "discrete_coin",
    "parity": "even",
    "theta_0": THETA_0,
    "sizes": [30, 40, 50, 60, 80, 100, 120, 150, 200, 250, 300, 400, 500, 600, 800, 1000],
    # Chosen from the shapes: [0, 0.01] stays two-peaked up to N = 1000,
    # [0.42, 0.50] is one central peak from N ~ 100 up and stops short of theta_0.
    "window_delocalized": [0.0, 0.01],
    "window_localized": [0.42, 0.50],
    "n_samples": 1800,
    "test_size": 0.2,
    "probe_step": 0.0025,
    "probe_realizations": 10,
    "seed": 0,
    "probe_seed": 1,
    "random_state": 0,
    "max_iter": 400,
}

RESULTS = ROOT / "notes" / "critical_vs_time_results.json"


def probe_p_deloc(clf: MLPClassifier, n: int, grid: np.ndarray, cfg: dict) -> np.ndarray:
    """Mean P(deloc) over fresh walks at each grid value; seeds never reuse training walks."""
    ss = np.random.SeedSequence(cfg["probe_seed"] * 1_000_003 + n)
    seeds = [int(s.generate_state(1)[0]) for s in ss.spawn(grid.size * cfg["probe_realizations"])]
    rows = []
    for k, value in enumerate(np.repeat(grid, cfg["probe_realizations"])):
        r = run_walk(n, cfg["channel"], cfg["theta_0"], float(value),
                     seed=seeds[k], parity=cfg["parity"])
        rows.append(probability(r.psi_plus, r.psi_minus))
    p = clf.predict_proba(np.asarray(rows))[:, DELOCALIZED]
    return p.reshape(grid.size, cfg["probe_realizations"]).mean(axis=1)


def run_size(n: int, cfg: dict) -> dict:
    t0 = time.time()
    ds = make_dataset(n, cfg["channel"], tuple(cfg["window_delocalized"]),
                      tuple(cfg["window_localized"]), n_samples=cfg["n_samples"],
                      theta_0=cfg["theta_0"], seed=cfg["seed"], parity=cfg["parity"])
    X_tr, X_te, y_tr, y_te = train_test_split(
        ds.X, ds.y, test_size=cfg["test_size"], random_state=cfg["random_state"], stratify=ds.y)

    clf = MLPClassifier(max_iter=cfg["max_iter"], random_state=cfg["random_state"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        clf.fit(X_tr, y_tr)

    grid = np.round(np.arange(0.0, cfg["theta_0"] + 1e-12, cfg["probe_step"]), 6)
    p_deloc = probe_p_deloc(clf, n, grid, cfg)
    rec = {
        "n": n,
        "t_f": max_steps(n, cfg["parity"]),
        "delta_theta_c": first_below_half(grid, p_deloc),
        "delta_theta_c_sustained": sustained_below_half(grid, p_deloc),
        "test_accuracy": clf.score(X_te, y_te),
        "n_iter": clf.n_iter,
        "grid": grid.tolist(),
        "p_deloc": p_deloc.tolist(),
        "seconds": round(time.time() - t0, 1),
    }
    print(f"N={n:5d}  t_f={rec['t_f']:5d}  dtheta_c={rec['delta_theta_c']}  "
          f"(sustained {rec['delta_theta_c_sustained']})  test acc={rec['test_accuracy']:.4f}  "
          f"iters={rec['n_iter']}  {rec['seconds']}s", flush=True)
    return rec


def plot(records: list[dict], cfg: dict, overwrite: bool) -> Path:
    import matplotlib.pyplot as plt

    from plotting.style import CHANNEL_STYLE, save_figure

    style = CHANNEL_STYLE[cfg["channel"]]
    t = np.array([r["t_f"] for r in records], dtype=float)
    dc = np.array([np.nan if r["delta_theta_c"] is None else r["delta_theta_c"] for r in records])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(t, dc, linestyle="none", marker="o", color=style["color"],
            label=f"MLP, {style['label']} ($\\Delta\\theta$)")
    ax.set_xlabel(r"final time $t_f$ (steps)")
    ax.set_ylabel(r"$\Delta\theta_c$")
    ax.set_ylim(bottom=0)
    ax.set_title(r"Critical $\Delta\theta_c$ vs final time $t_f$"
                 f"\n$\\theta_0=\\pi/6$, lattice $x=-N..N$ (no 0), $t_f=N-1$, seed {cfg['seed']}",
                 fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = save_figure(fig, "mlp_critical_vs_time", overwrite=overwrite)
    plt.close(fig)
    return path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sizes", type=int, nargs="+", help="lattice half-widths N")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    cfg = dict(CONFIG)
    if args.sizes:
        cfg["sizes"] = args.sizes

    records = [run_size(n, cfg) for n in cfg["sizes"]]
    RESULTS.write_text(json.dumps({"config": cfg, "records": records}, indent=1))
    print(f"saved {RESULTS}")
    print(f"saved {plot(records, cfg, args.overwrite)}")


if __name__ == "__main__":
    main()
