#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.generate import LOCALIZED  # noqa: E402
from qw.operators import THETA_0_DEFAULT as THETA_0  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "notes" / "mlp_evolution_results.json"
ORDER = ("discrete_coin", "continuous_coin", "random_translation")

# Localized training windows, needed for the analytic ceiling. These mirror
# CHANNELS in scripts/train_mlp_evolution.py; a mismatch is a bug there or here.
WINDOW_LOCALIZED = {
    "discrete_coin": (0.45, THETA_0),
    "continuous_coin": (0.45, THETA_0),
    "random_translation": (0.45, 0.5),
}


def degenerate_fraction(
    channel: str, t: np.ndarray, eps: float = 0.05, d_max: int = 0
) -> np.ndarray:
    """Fraction of localized-window walks that still look like a pure walk at ``t``.

    For ``random_translation``, ``d = min(k, t - k)`` is a schedule's distance
    from a pure one: ``k`` inverted steps out of ``t``, with ``k = t`` counted
    as pure because the all-inverted walk is the mirror of the pure walk and
    the symmetric initial spinor makes ``P`` mirror-symmetric. ``d_max = 0`` is
    exact degeneracy; ``d_max = 1`` adds the schedules one inverted step away,
    which become indistinguishable in practice once ``t`` is large enough that
    one step barely moves ``P(x, t)``.

    For the coin channels no exact degeneracy exists, so this reports the
    nearest comparable quantity -- every jitter in the first ``t`` steps
    smaller than ``eps`` -- which is an upper bound on how pure-looking they
    can get, and is already negligible by ``t = 2``.
    """
    from math import comb

    lo, hi = WINDOW_LOCALIZED[channel]
    c = np.linspace(lo, hi, 20001)
    t = np.asarray(t, dtype=float)
    if channel == "random_translation":
        out = np.empty((t.size, c.size))
        for i, ti in enumerate(t):
            n = int(round(ti))
            ks = {k for k in range(d_max + 1) if k <= n}
            ks |= {n - k for k in range(d_max + 1) if n - k >= 0}
            out[i] = sum(comb(n, k) * c**k * (1 - c) ** (n - k) for k in sorted(ks))
        f = out
    elif channel == "continuous_coin":
        f = np.minimum(eps / c, 1.0) ** t[:, None]
    else:
        f = np.zeros((t.size, c.size))  # |Delta_theta| is fixed and nonzero
    return np.trapezoid(f, c, axis=1) / (hi - lo)


def error_floor(n_test: int) -> float:
    """Where to draw a zero-error point on a log axis: half the resolution."""
    return 0.5 / n_test


def _plot_error(ax, x, err, floor, style, *, label=None, marker="o"):
    """One channel's error curve, with exact-zero points drawn hollow at ``floor``."""
    y = np.where(err > 0, err, floor)
    zero = err <= 0
    ax.plot(x, y, color=style["color"], linestyle=style["linestyle"],
            linewidth=1.6, label=label, zorder=3)
    ax.plot(x[~zero], y[~zero], linestyle="none", marker=marker, markersize=5,
            color=style["color"], zorder=4)
    ax.plot(x[zero], y[zero], linestyle="none", marker=marker, markersize=5,
            markerfacecolor="white", markeredgecolor=style["color"],
            markeredgewidth=1.2, zorder=4)


def plot_scaling(recs: dict, n_test: int, overwrite: bool) -> Path:
    import matplotlib.pyplot as plt

    from plotting.style import CHANNEL_STYLE, save_figure

    floor = error_floor(n_test)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    fig.suptitle(
        "Test error of the evolution MLPs, re-scaled -- why Random Transform "
        "rises so slowly",
        fontsize=13,
    )

    rows = [
        ("snapshot model: one MLP per $P(x,t)$", "t (steps)",
         lambda r: (np.asarray(r["snapshot_times"]),
                    1.0 - np.asarray(r["snapshot_accuracy"]))),
        (r"full-evolution model on $P(x, t \leq T)$",
         "T (steps of history given to the model)",
         lambda r: (np.asarray(r["prefix_times"]),
                    1.0 - np.asarray(r["prefix_test_accuracy"]))),
    ]
    scales = [("y vs $\\log x$", "log", "linear"),
              ("$\\log y$ vs $x$", "linear", "log"),
              ("log--log", "log", "log")]

    for i, (row_title, xlabel, get) in enumerate(rows):
        for j, (scale_title, xs, ys) in enumerate(scales):
            ax = axes[i, j]
            for name in ORDER:
                r, style = recs[name], CHANNEL_STYLE[name]
                x, err = get(r)
                keep = x > 0  # t = 0 has no place on a log axis
                _plot_error(ax, x[keep], err[keep], floor, style,
                            label=style["label"])

            # The analytic label-noise band, on the snapshot row only. Errors
            # only ever land on the localized half of the test set, hence 1/2.
            if i == 0:
                tt = np.arange(1, 41, dtype=float)
                lo_b = 0.5 * degenerate_fraction("random_translation", tt, d_max=0)
                hi_b = 0.5 * degenerate_fraction("random_translation", tt, d_max=1)
                ax.fill_between(tt, lo_b, hi_b, color="black", alpha=0.12,
                                linewidth=0, zorder=1,
                                label="label noise: walks within $d$ inverted\n"
                                      "steps of pure, $d = 0$ to $1$")
                for b in (lo_b, hi_b):
                    ax.plot(tt, b, color="black", linewidth=0.9, alpha=0.5,
                            zorder=2)

            ax.set_xscale(xs)
            ax.set_yscale(ys)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("test error $1 -$ accuracy")
            ax.set_title(f"{row_title}\n{scale_title}", fontsize=9)
            ax.grid(True, which="both", alpha=0.2)
            if ys == "log":
                ax.set_ylim(floor / 1.8, 0.75)
                ax.axhspan(floor / 1.8, 1.0 / n_test, color="0.85", alpha=0.5,
                           zorder=0)
            if ys == "linear":
                ax.margins(y=0.22)
                ax.set_ylim(bottom=0.0)  # an error is never negative
            # Every curve decays left to right, so the top right is always clear.
            ax.legend(fontsize=7, loc="upper right")

    fig.text(0.5, 0.005,
             f"Hollow markers = zero test errors (exactly 0 cannot be drawn on a "
             f"log axis); they sit in the shaded band below the 1/{n_test} "
             f"resolution of a {n_test}-sample test set.",
             ha="center", fontsize=8, color="0.3")
    fig.tight_layout(rect=(0, 0.025, 1, 0.95))
    path = save_figure(fig, "mlp_evolution_accuracy_scaling", overwrite=overwrite)
    plt.close(fig)
    return path


def stratify(cfg: dict) -> dict:
    """Snapshot errors for random_translation, split by how many inversions fired.

    Re-runs the snapshot fits at early ``t`` and reconstructs each sample's
    schedule from its recorded seed, so the error can be attributed per walk.
    """
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.model_selection import train_test_split

    from data.evolution import make_evolution_dataset
    from models.mlp import MLPClassifier
    from qw.randomness import make_schedule

    ds = make_evolution_dataset(
        cfg["n"], "random_translation", (0.0, 0.02), (0.45, 0.5),
        n_samples=cfg["n_samples"], theta_0=cfg["theta_0"],
        seed=cfg["seed"], parity=cfg["parity"],
    )
    inv = np.empty((len(ds), ds.n_steps), dtype=bool)
    for k in range(len(ds)):
        rng = np.random.default_rng(int(ds.seeds[k]))
        inv[k] = make_schedule("random_translation", ds.n_steps, cfg["theta_0"],
                               float(ds.control_values[k]), rng).inverse_translation
    cum = np.cumsum(inv, axis=1)

    itr, ite = train_test_split(np.arange(len(ds)), test_size=0.2,
                                random_state=cfg["random_state"], stratify=ds.y)
    t_max = cfg["strat_t_max"]
    keys = ("t", "acc", "acc_d2", "err_d0", "err_d1", "err_d2", "n_d0", "n_d1")
    out = {k: [] for k in keys}
    for t in range(1, t_max + 1):
        X = ds.at_time(t - 1)
        clf = MLPClassifier(normalize=True, max_iter=cfg["max_iter"],
                            random_state=cfg["random_state"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            clf.fit(X[itr], ds.y[itr])
        wrong = clf.predict(X[ite]) != ds.y[ite]
        loc = ds.y[ite] == LOCALIZED
        k = cum[ite, t - 1]
        # Distance from a pure schedule. k = t counts as pure: the all-inverted
        # walk is the mirror of the pure one, and P is mirror-symmetric here.
        dist = np.minimum(k, t - k)
        g = [loc & (dist == 0), loc & (dist == 1), loc & (dist >= 2)]
        mean = lambda m: float(wrong[m].mean()) if m.any() else np.nan
        out["t"].append(t)
        out["acc"].append(float(1 - wrong.mean()))
        # Undefined while no localized walk has reached d >= 2 yet (t = 1),
        # where dropping d <= 1 would leave the delocalized half alone.
        out["acc_d2"].append(float(1 - wrong[~(g[0] | g[1])].mean())
                             if g[2].any() else np.nan)
        for i in range(3):
            out[f"err_d{i}"].append(mean(g[i]))
        out["n_d0"].append(int(g[0].sum()))
        out["n_d1"].append(int(g[1].sum()))
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


def plot_diagnosis(recs: dict, strat: dict, overwrite: bool) -> Path:
    import matplotlib.pyplot as plt

    from plotting.style import CHANNEL_STYLE, save_figure

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    fig.suptitle(
        "Random Transform is slow because its randomness can fail to fire, "
        "not because it is hard to learn",
        fontsize=12,
    )

    # (a) how much of each localized window is still a pure walk at time t
    ax = axes[0]
    tt = np.arange(1, 21, dtype=float)
    absent = []
    for name in ORDER:
        style = CHANNEL_STYLE[name]
        f = degenerate_fraction(name, tt)
        if not (f > 0).any():
            absent.append(style["label"])
            continue
        ax.plot(tt[f > 0], f[f > 0], color=style["color"],
                linestyle=style["linestyle"], marker="o", markersize=4,
                label=style["label"])
    if absent:
        ax.text(0.5, 0.06, " and ".join(absent) + r": identically $0$"
                "\n" r"($|\Delta\theta|$ is fixed and never small)",
                transform=ax.transAxes, ha="center", fontsize=8,
                color=CHANNEL_STYLE["discrete_coin"]["color"])
    ax.set_yscale("log")
    ax.set_xlabel("t (steps)")
    ax.set_ylabel("fraction of localized window\nindistinguishable from pure")
    ax.set_title("(a) exposure to randomness\n"
                 r"RT: $(1-P_r)^t + P_r^t$;  coins: all $|\Delta\theta| < 0.05$",
                 fontsize=9)
    ax.set_ylim(1e-8, 1.5)

    # (b) error split by distance from a pure schedule
    ax = axes[1]
    t = strat["t"]
    spec = [("err_d0", "solid", "o", 1.0, r"$d = 0$ (exactly a pure walk)"),
            ("err_d1", "dashed", "s", 0.75, r"$d = 1$ (one inverted step)"),
            ("err_d2", "dotted", "^", 0.55, r"$d \geq 2$")]
    for key, ls, mk, al, lab in spec:
        ax.plot(t, strat[key], color="purple", linestyle=ls, marker=mk,
                markersize=5, alpha=al, label=lab)
    ax.set_xlabel("t (steps)")
    ax.set_ylabel("per-group test error")
    ax.set_ylim(-0.05, 1.08)
    ax.set_title("(b) every error is a walk within one inverted step of pure\n"
                 r"localized test walks, grouped by $d = \min(k,\, t-k)$",
                 fontsize=9)

    # (c) the corrected curve against the coin channels
    ax = axes[2]
    for name in ("discrete_coin", "continuous_coin"):
        style = CHANNEL_STYLE[name]
        r = recs[name]
        n = len(strat["t"])
        ax.plot(r["snapshot_times"][:n], r["snapshot_accuracy"][:n],
                color=style["color"], linestyle=style["linestyle"],
                marker="o", markersize=4, label=style["label"])
    ax.plot(t, strat["acc"], color="purple", linestyle="dashdot", marker="o",
            markersize=4, label="Random Transform (as plotted)")
    ax.plot(t, strat["acc_d2"], color="purple", linestyle="solid", marker="D",
            markersize=5, alpha=0.6,
            label=r"Random Transform, $d \geq 2$ walks only")
    ax.set_xlabel("t (steps)")
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0.45, 1.03)
    ax.set_title("(c) drop the degenerate walks and the gap closes", fontsize=9)

    for ax in axes:
        ax.grid(True, which="both", alpha=0.2)
        ax.legend(fontsize=7, loc="best")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    path = save_figure(fig, "mlp_evolution_rt_diagnosis", overwrite=overwrite)
    plt.close(fig)
    return path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-strat", action="store_true",
                   help="skip the per-walk diagnosis figure (it refits the "
                        "early snapshot models)")
    args = p.parse_args()

    recs = {r["channel"]: r for r in json.loads(RESULTS.read_text())}
    missing = [c for c in ORDER if c not in recs]
    if missing:
        raise SystemExit(f"{RESULTS} is missing channels: {missing}")

    cfg = dict(n=recs["random_translation"]["n"],
               n_samples=recs["random_translation"]["n_samples"],
               theta_0=recs["random_translation"]["theta_0"],
               seed=recs["random_translation"]["seed"],
               parity="odd",
               random_state=recs["random_translation"]["random_state"],
               max_iter=recs["random_translation"]["max_iter"],
               strat_t_max=12)
    n_test = int(round(0.2 * cfg["n_samples"]))

    print(f"saved {plot_scaling(recs, n_test, args.overwrite)}")
    if not args.no_strat:
        print("refitting early snapshot models for the per-walk diagnosis...")
        strat = stratify(cfg)
        print(f"saved {plot_diagnosis(recs, strat, args.overwrite)}")


if __name__ == "__main__":
    main()
