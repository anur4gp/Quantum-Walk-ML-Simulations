#!/usr/bin/env python3
"""Train the MLP on the full evolution P(x, t) and time-resolve the transition.

``scripts/train_mlp.py`` hands the MLP one final-time distribution per walk;
this hands it the whole history, and asks when in walk time the classifier
stops calling a walk delocalized. Three separate quantities come out:

    onset time      first step from which the two training windows are
                    separable. A property of the classifier, not the transition.
    critical value  the Paper A quantity (first probe point with P(deloc) < 0.5,
                    CLAUDE.md Sec. 7.4), computed at every time step so it can
                    be watched drift.
    crossover time  the same rule read along the time axis at fixed randomness:
                    localization needs time to set in, so a walk in the
                    transition region reads delocalized early, localized late.

This touches the intermediate regime, which Phase 1 does not. Training windows
are still the placeholders from ``scripts/train_mlp.py``, so every critical
value is conditional on them (CLAUDE.md Sec. 6, Sec. 9). Nothing here compares
to Paper A Table I: Paper A classifies final-time distributions.

    python3 scripts/train_mlp_evolution.py [--channel NAME] [--quick]
    python3 scripts/train_mlp_evolution.py --norm global [--save-results PATH]
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analysis.critical import (  # noqa: E402
    first_below_half,
    onset,
    sustained_below_half,
)
from data.evolution import (  # noqa: E402
    EvolutionDataset,
    ProbeSet,
    make_evolution_dataset,
    make_probe_evolutions,
)
from data.generate import DELOCALIZED, LOCALIZED  # noqa: E402
from data.preprocess import EVOLUTION_NORMALIZATIONS, normalize_pmax_evolution  # noqa: E402
from models.mlp import MLPClassifier  # noqa: E402
from qw.observables import moment_of_inertia_series  # noqa: E402
from qw.walk import lattice_positions  # noqa: E402

PI = np.pi
THETA_0 = PI / 6  # Paper A default (CLAUDE.md Sec. 2.1)

# Windows are the placeholders from scripts/train_mlp.py, unchanged so the two
# scripts stay comparable. ``probe_range`` is the full range the control
# parameter is defined on (CLAUDE.md Sec. 2.3).

CHANNELS: dict[str, dict] = {
    "discrete_coin": {
        "control": "delta_theta",
        "control_label": r"$\Delta\theta$",
        "window_delocalized": (0.0, 0.05),
        "window_localized": (0.45, THETA_0),
        "probe_range": (0.0, THETA_0),
        "note": "Paper A 'Jittered'. Two coins at theta_0 +- delta_theta, 50/50.",
    },
    "continuous_coin": {
        "control": "delta_theta_max",
        "control_label": r"$\Delta\theta_M$",
        "window_delocalized": (0.0, 0.05),
        "window_localized": (0.45, THETA_0),
        "probe_range": (0.0, THETA_0),
        "note": "Paper A 'Uniform Jittered'. One-sided uniform draw each step.",
    },
    "random_translation": {
        "control": "p_r",
        "control_label": r"$P_r$",
        "window_delocalized": (0.0, 0.02),
        "window_localized": (0.45, 0.5),
        "probe_range": (0.0, 0.5),
        "note": (
            "KNOWN-HARD CASE (CLAUDE.md Sec. 3). MLP and CNN systematically deviate "
            "here; a clean-looking critical value is a reason for suspicion."
        ),
    },
}

DEFAULTS: dict = {
    "n": 80,  # 2N+1 = 161 sites, N = 80 steps
    "n_samples": 1800,  # Paper A's value (CLAUDE.md Sec. 6)
    "test_size": 0.2,  # 80/20 split (CLAUDE.md Sec. 6)
    "theta_0": THETA_0,
    "parity": "odd",
    "seed": 0,  # simulation master seed for the labelled set
    "probe_seed": 1,  # kept distinct so probes never reuse training walks
    "random_state": 0,  # sklearn: weight init, shuffling, splits
    "max_iter": 400,
    "n_probe_controls": 25,
    "n_probe_realizations": 40,
    "slice_stride": 1,  # train a snapshot model every k-th recorded time step
    "onset_threshold": 0.95,
    # MoI comparator: a walk has left the ballistic regime once its MoI falls
    # to this fraction of the pure walk's at the same step. Our threshold, not
    # a paper value -- Sec. 2.5 defines the kink but fixes no number.
    "moi_ratio": 0.5,
    # Truncations T (as fractions of the step count) at which a model is
    # trained on P(x, t <= T). The last one is the full evolution.
    "prefix_fractions": (0.0625, 0.125, 0.25, 0.375, 0.5, 0.75, 1.0),
}

QUICK_OVERRIDES: dict = {
    "n": 40,
    "n_samples": 400,
    "n_probe_controls": 13,
    "n_probe_realizations": 12,
    "slice_stride": 2,
    "prefix_fractions": (0.125, 0.25, 0.5, 1.0),
}


def _fit(X: np.ndarray, y: np.ndarray, cfg: dict, *, normalize: bool) -> MLPClassifier:
    """One Paper A MLP (400-200-100-50, alpha=1e-3), fitted.

    Hitting ``max_iter`` is a result to report, not a reason to abort;
    ``n_iter`` is what flags it.
    """
    clf = MLPClassifier(
        normalize=normalize, max_iter=cfg["max_iter"], random_state=cfg["random_state"]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        clf.fit(X, y)
    return clf


def _p_deloc(clf: MLPClassifier, probe: np.ndarray) -> np.ndarray:
    """Mean P(delocalized) over realisations, from ``(n_controls, n_real, n_features)``.

    Averaging probabilities rather than hard labels is what makes the 0.5
    crossing a smooth function of the control parameter.
    """
    n_controls, n_real = probe.shape[0], probe.shape[1]
    flat = probe.reshape(n_controls * n_real, -1)
    return clf.predict_proba(flat)[:, DELOCALIZED].reshape(n_controls, n_real).mean(axis=1)


def snapshot_sweep(
    ds: EvolutionDataset, probe: ProbeSet, itr, ite, cfg: dict
) -> dict:
    """One snapshot MLP per recorded time step: accuracy and P(deloc) vs control.

    Each model sees a single P(x, t), the ordinary Paper A input at an earlier
    time, so ``normalize=True`` here is Paper A's P_max = 1.
    """
    k_indices = np.arange(0, ds.n_times, cfg["slice_stride"])
    if k_indices[-1] != ds.n_times - 1:
        k_indices = np.append(k_indices, ds.n_times - 1)

    times, accuracy, curves = [], [], []
    for k in k_indices:
        X = ds.at_time(int(k))
        clf = _fit(X[itr], ds.y[itr], cfg, normalize=True)
        accuracy.append(clf.score(X[ite], ds.y[ite]))
        curves.append(_p_deloc(clf, probe.P[:, :, int(k), :]))
        times.append(int(ds.times[k]))
    return {
        "times": np.asarray(times, dtype=float),
        "accuracy": np.asarray(accuracy, dtype=float),
        "p_deloc": np.asarray(curves, dtype=float),  # (n_times, n_controls)
    }


def prefix_sweep(
    ds: EvolutionDataset, probe: ProbeSet, itr, ite, cfg: dict, norm_mode: str
) -> dict:
    """Full-evolution MLPs trained on the history P(x, t <= T), for several T.

    The last entry is the full-evolution classifier; the shorter ones say how
    much history the answer needs. Normalisation is applied here and the
    estimator's own flag switched off, so nothing is normalised twice.
    """
    p_norm = normalize_pmax_evolution(ds.P, norm_mode)
    truncations = sorted(
        {max(1, int(round(f * ds.n_times))) for f in cfg["prefix_fractions"]}
    )

    times, train_acc, test_acc, n_iter, curves = [], [], [], [], []
    loss_curve = np.asarray([])
    for t_cut in truncations:
        X = p_norm[:, :t_cut, :].reshape(len(ds), -1)
        clf = _fit(X[itr], ds.y[itr], cfg, normalize=False)
        q = normalize_pmax_evolution(probe.P[:, :, :t_cut, :], norm_mode)
        curves.append(_p_deloc(clf, q))
        times.append(int(ds.times[t_cut - 1]))
        train_acc.append(clf.score(X[itr], ds.y[itr]))
        test_acc.append(clf.score(X[ite], ds.y[ite]))
        n_iter.append(clf.n_iter)
        loss_curve = clf.loss_curve  # kept from the last (= full-evolution) model
    return {
        "times": np.asarray(times, dtype=float),
        "n_features": [t * ds.n_sites for t in truncations],
        "train_accuracy": np.asarray(train_acc, dtype=float),
        "test_accuracy": np.asarray(test_acc, dtype=float),
        "n_iter": n_iter,
        "p_deloc": np.asarray(curves, dtype=float),
        "loss_curve": loss_curve,
    }


def moi_crossover(probe: ProbeSet, cfg: dict) -> dict:
    """Manual comparator: when does MoI leave the ballistic curve? (Sec. 2.5)

    MoI(t) is averaged over realisations at each control value and divided by
    the control = 0 walk's MoI(t), the ballistic reference; the crossover is
    the first step where that ratio drops below ``moi_ratio`` and stays there.
    A different estimator on the same walks, not a validation of the ML one.
    """
    positions = lattice_positions(probe.n, probe.parity)
    moi = moment_of_inertia_series(probe.P, positions).mean(axis=1)  # (controls, times)
    ballistic = moi[0]
    ratio = np.divide(moi, ballistic, out=np.ones_like(moi), where=ballistic > 0)
    times = probe.times.astype(float)

    crossover = []
    for row in ratio:
        above = np.flatnonzero(row >= cfg["moi_ratio"])
        if above.size == 0:
            crossover.append(float(times[0]))
        elif int(above[-1]) == row.size - 1:
            crossover.append(np.nan)
        else:
            crossover.append(float(times[int(above[-1]) + 1]))
    return {
        "moi": moi,
        "ratio": ratio,
        "times": times,
        "crossover_time": np.asarray(crossover, dtype=float),
    }


def usable_from(sweep: dict, cfg: dict) -> float | None:
    """First time step from which a model's probe output may be read at all.

    Both conditions must hold from there on: test accuracy at or above
    ``onset_threshold``, and P(deloc) >= 0.5 at the weakest probe randomness,
    which is ground truth rather than a fit. Without the second condition the
    Sec. 7.4 rule reports a crossing at the first step of every sweep, because
    an early model has no signal and its probe curve is arbitrary.
    """
    ok = (sweep["accuracy"] >= cfg["onset_threshold"]) & (sweep["p_deloc"][:, 0] >= 0.5)
    return onset(sweep["times"], ok.astype(float), 1.0)


def extract_critical(
    sweep: dict, controls: np.ndarray, t_min: float | None = None
) -> dict:
    """Apply the Sec. 7.4 MLP rule along both axes of ``p_deloc`` (n_times, n_controls).

    Along the control axis at fixed time gives the critical randomness; along
    the time axis at fixed control gives the critical time. Rows before
    ``t_min`` come back NaN rather than dropped, so the time axis stays aligned.
    """
    p = sweep["p_deloc"]
    times = sweep["times"]
    keep = np.ones(times.shape, dtype=bool) if t_min is None else times >= t_min
    if not keep.any():
        nan_c = np.full(times.shape, np.nan)
        nan_k = np.full(controls.shape, np.nan)
        return {
            "t_min": t_min,
            "critical_control": nan_c,
            "crossover_time": nan_k,
            "crossover_time_sustained": nan_k,
        }
    p_kept, times_kept = p[keep], times[keep]
    critical_control = np.full(times.shape, np.nan)
    critical_control[keep] = [
        np.nan if (v := first_below_half(controls, row)) is None else v
        for row in p_kept
    ]
    crossover_time = np.array(
        [
            np.nan if (v := first_below_half(times_kept, col)) is None else v
            for col in p_kept.T
        ]
    )
    crossover_sustained = np.array(
        [
            np.nan if (v := sustained_below_half(times_kept, col)) is None else v
            for col in p_kept.T
        ]
    )
    return {
        "t_min": t_min,
        "critical_control": critical_control,
        "crossover_time": crossover_time,
        "crossover_time_sustained": crossover_sustained,
    }


def _fmt(value: float) -> str:
    return "  --  " if not np.isfinite(value) else f"{value:.4f}"


def print_header(name: str, spec: dict, cfg: dict, norm_mode: str) -> None:
    import textwrap

    print("=" * 78)
    print(f"channel             {name}  ({spec['control']})")
    for line in textwrap.wrap(spec["note"], 56):
        print(f"                    {line}")
    print(f"lattice             n={cfg['n']}  ({2 * cfg['n'] + 1} sites, {cfg['parity']})")
    print(f"steps recorded      1..{cfg['n']}  (every {cfg['slice_stride']} for snapshots)")
    print(f"theta_0             {cfg['theta_0']:.6f}")
    print(f"window delocalized  {tuple(spec['window_delocalized'])}   -> label 0")
    print(f"window localized    {tuple(spec['window_localized'])}   -> label 1")
    print(f"samples             {cfg['n_samples']}  (sim seed {cfg['seed']}, "
          f"test_size {cfg['test_size']})")
    print(f"probe sweep         {cfg['n_probe_controls']} x "
          f"{cfg['n_probe_realizations']} over {tuple(spec['probe_range'])}  "
          f"(seed {cfg['probe_seed']})")
    print(f"space-time P_max    {norm_mode}")
    print(f"architecture        (400, 200, 100, 50), alpha=0.001  [Paper A]")
    print(f"max_iter            {cfg['max_iter']}   (sklearn seed {cfg['random_state']})")
    print("=" * 78)


def report(name: str, spec: dict, cfg: dict, res: dict) -> None:
    snap, pre = res["snapshot"], res["prefix"]
    controls = res["controls"]

    print(f"\nfull evolution      {pre['n_features'][-1]} features "
          f"({len(pre['times'])} truncations trained)")
    print(f"  train accuracy    {pre['train_accuracy'][-1]:.4f}")
    print(f"  test accuracy     {pre['test_accuracy'][-1]:.4f}"
          + ("   <-- hit max_iter" if pre["n_iter"][-1] >= cfg["max_iter"] else ""))
    print(f"final snapshot      {snap['accuracy'][-1]:.4f} test accuracy "
          f"({len(controls)} sites -> baseline)")

    t_onset = res["onset_time"]
    print(f"\nonset time          "
          + ("never" if t_onset is None else f"t = {t_onset:.0f}")
          + f"   (snapshot test accuracy >= {cfg['onset_threshold']} and staying)")
    print("  Separability of the two extreme windows only. The windows differ in\n"
          "  coin angle as well as localization, so an early onset need not be\n"
          "  about localization.")

    t_usable = res["usable_from"]
    print("readable from       "
          + ("never" if t_usable is None else f"t = {t_usable:.0f}")
          + "   (accuracy gate AND the weakest probe still reads delocalized)")
    print("  Everything below is read from t >= this; earlier models have no\n"
          "  signal and their probe curves are arbitrary.")
    if t_usable is None:
        print("\n  NO USABLE TIME RANGE. Check the training windows and the probe\n"
              "  range before anything else.")
        return

    print(f"\ncritical {spec['control']} vs observation time  "
          f"(first probe point with P(deloc) < 0.5)")
    print(f"  {'t':>5}  {'snapshot':>9}  {'full evol.':>10}")
    pre_crit = res["prefix_critical"]["critical_control"]
    snap_crit = res["snapshot_critical"]["critical_control"]
    for t in pre["times"]:
        i = int(np.argmin(np.abs(snap["times"] - t)))
        j = int(np.argmin(np.abs(pre["times"] - t)))
        print(f"  {t:>5.0f}  {_fmt(snap_crit[i]):>9}  {_fmt(pre_crit[j]):>10}")

    print(f"\ncrossover time vs {spec['control']}  "
          f"(first t with P(deloc) < 0.5, snapshot models)")
    print(f"  {spec['control']:>12}  {'t_c':>8}  {'t_c sustained':>14}  {'MoI t_c':>8}")
    cross = res["snapshot_critical"]["crossover_time"]
    cross_s = res["snapshot_critical"]["crossover_time_sustained"]
    cross_m = res["moi"]["crossover_time"]
    for c, t_c, t_s, t_m in zip(controls, cross, cross_s, cross_m):
        print(f"  {c:>12.4f}  {_fmt(t_c):>8}  {_fmt(t_s):>14}  {_fmt(t_m):>8}")
    print("\n  A finite t_c means that randomness strength sits inside the\n"
          "  transition region at this N; '--' means it never crossed within N\n"
          "  steps. 'MoI t_c' is the manual comparator (Sec. 2.5).")


def _pick(values: np.ndarray, count: int) -> np.ndarray:
    """Evenly spaced indices into ``values``, always including the last."""
    return np.unique(np.linspace(0, values.size - 1, count).round().astype(int))


def plot_channel(name: str, spec: dict, cfg: dict, res: dict, args) -> Path:
    import matplotlib.pyplot as plt

    from plotting.style import CHANNEL_STYLE, save_figure

    snap, pre, controls = res["snapshot"], res["prefix"], res["controls"]
    style = CHANNEL_STYLE[name]
    label = spec["control_label"]

    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    fig.suptitle(
        f"MLP on the full evolution $P(x,t)$ -- {style['label']} ({name}), "
        f"N={cfg['n']}, {cfg['n_samples']} samples",
        fontsize=12,
    )
    positions = np.arange(-cfg["n"], cfg["n"] + 1)
    extent = [positions[0], positions[-1], res["times_all"][0], res["times_all"][-1]]

    # Row 0 -- what the classifier is fed.
    for ax, cls, title in (
        (axes[0, 0], DELOCALIZED, "delocalized (label 0)"),
        (axes[0, 1], LOCALIZED, "localized (label 1)"),
    ):
        mean_p = res["mean_evolution"][cls]
        ax.imshow(mean_p, origin="lower", aspect="auto", extent=extent, cmap="viridis")
        ax.set_xlabel("x")
        ax.set_ylabel("t (steps)")
        ax.set_title(f"mean $P(x,t)$, {title}", fontsize=10)

    # Row 1 left -- separability vs time.
    ax = axes[1, 0]
    ax.plot(snap["times"], snap["accuracy"], color="black", linewidth=1.0,
            label="snapshot $P(x,t)$")
    ax.plot(pre["times"], pre["test_accuracy"], color=style["color"],
            linestyle=style["linestyle"], marker="o", markersize=4,
            label=r"full evolution $P(x,t'\leq t)$")
    ax.axhline(cfg["onset_threshold"], color="grey", linewidth=0.7, linestyle=":")
    if res["onset_time"] is not None:
        ax.axvline(res["onset_time"], color="grey", linewidth=0.8)
        ax.annotate(f"onset t={res['onset_time']:.0f}",
                    (res["onset_time"], 0.55), fontsize=8, rotation=90,
                    ha="right", color="grey")
    ax.set_xlabel("t (steps)")
    ax.set_ylabel("test accuracy")
    ax.set_title("separability of the two training windows", fontsize=10)
    ax.set_ylim(0.45, 1.02)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    # Row 1 right -- P(deloc) vs control, at several observation times.
    ax = axes[1, 1]
    cmap = plt.get_cmap("plasma")
    # Only readable models: an unusable early one draws a flat line near 0.5
    # that looks like a measurement and is not one.
    readable = np.flatnonzero(
        snap["times"] >= (res["usable_from"] or snap["times"][0])
    )
    picks = readable[_pick(readable, 5)]
    for rank, i in enumerate(picks):
        ax.plot(controls, snap["p_deloc"][i], marker=".", markersize=4,
                color=cmap(rank / max(1, len(picks) - 1)),
                label=f"t={snap['times'][i]:.0f}")
    ax.axhline(0.5, color="black", linewidth=0.8, linestyle="--")
    crit = res["snapshot_critical"]["critical_control"][-1]
    if np.isfinite(crit):
        ax.axvline(crit, color="black", linewidth=0.8, linestyle=":")
    ax.set_xlabel(label)
    ax.set_ylabel("P(delocalized)")
    ax.set_title("probe sweep -- critical value is the first crossing of 0.5",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    # Row 2 left -- P(deloc) vs time, at several randomness strengths.
    ax = axes[2, 0]
    cross = res["snapshot_critical"]["crossover_time"]
    picks = _pick(controls, 7)
    for rank, j in enumerate(picks):
        colour = cmap(rank / max(1, len(picks) - 1))
        ax.plot(snap["times"], snap["p_deloc"][:, j], color=colour, linewidth=1.0,
                label=f"{controls[j]:.3f}")
        if np.isfinite(cross[j]):
            ax.plot([cross[j]], [0.5], marker="v", color=colour, markersize=6)
    ax.axhline(0.5, color="black", linewidth=0.8, linestyle="--")
    if res["usable_from"] is not None:
        ax.axvspan(snap["times"][0], res["usable_from"], color="grey", alpha=0.15)
        ax.annotate("models not\nreadable here", (snap["times"][0], 0.05),
                    fontsize=7, color="grey", va="bottom")
    ax.set_xlabel("t (steps)")
    ax.set_ylabel("P(delocalized)")
    ax.set_title(f"crossover in time -- markers are $t_c$ ({label} in legend)",
                 fontsize=10)
    ax.legend(fontsize=7, title=label, title_fontsize=7, ncol=2)
    ax.grid(True, alpha=0.25)

    # Row 2 right -- the two critical curves.
    ax = axes[2, 1]
    ax.plot(controls, cross, marker="o", markersize=4, color=style["color"],
            linestyle=style["linestyle"], label="$t_c$ (Sec. 7.4 rule)")
    ax.plot(controls, res["snapshot_critical"]["crossover_time_sustained"],
            marker="x", markersize=5, linestyle="none", color="black",
            label="$t_c$ sustained (robustness)")
    ax.plot(controls, res["moi"]["crossover_time"], marker="s", markersize=3,
            linestyle="none", color="grey",
            label=f"MoI manual, ratio {cfg['moi_ratio']}")
    ax.set_xlabel(label)
    ax.set_ylabel("crossover time $t_c$ (steps)")
    ax.set_title("time point of the critical point", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = save_figure(fig, f"mlp_evolution_{name}", overwrite=args.overwrite)
    plt.close(fig)
    return path


def plot_summary(records: dict, cfg: dict, args) -> Path:
    import matplotlib.pyplot as plt

    from plotting.style import CHANNEL_STYLE, save_figure

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        f"MLP on the full evolution -- all three randomness channels, N={cfg['n']}",
        fontsize=12,
    )

    for name, res in records.items():
        style = CHANNEL_STYLE[name]
        spec = CHANNELS[name]
        snap = res["snapshot"]
        # Control parameters have different ranges, so compare them rescaled.
        scale = spec["probe_range"][1]
        kw = dict(color=style["color"], linestyle=style["linestyle"],
                  label=f"{style['label']} ({spec['control']})")

        axes[0, 0].plot(snap["times"], snap["accuracy"], **kw)
        axes[0, 1].plot(
            snap["times"],
            res["snapshot_critical"]["critical_control"] / scale,
            marker=".", markersize=4, **kw,
        )
        axes[1, 0].plot(
            res["controls"] / scale,
            res["snapshot_critical"]["crossover_time_sustained"],
            marker="o", markersize=4, **kw,
        )
        axes[1, 0].plot(
            res["controls"] / scale, res["moi"]["crossover_time"],
            marker="s", markersize=3, linestyle="none", color=style["color"],
            alpha=0.45,
        )
        axes[1, 1].plot(res["prefix"]["times"], res["prefix"]["test_accuracy"],
                        marker="o", markersize=4, **kw)

    axes[0, 0].set_xlabel("t (steps)")
    axes[0, 0].set_ylabel("test accuracy")
    axes[0, 0].set_title("snapshot model -- separability of the training windows",
                         fontsize=10)
    axes[0, 0].set_ylim(0.45, 1.02)

    axes[0, 1].set_xlabel("t (steps)")
    axes[0, 1].set_ylabel("critical control / range")
    axes[0, 1].set_title("critical randomness vs observation time", fontsize=10)

    axes[1, 0].set_xlabel("control / range")
    axes[1, 0].set_ylabel("crossover time $t_c$ (steps)")
    axes[1, 0].set_title(
        "time point of the critical point (lines: MLP sustained rule; "
        "faint squares: manual MoI)", fontsize=9)

    axes[1, 1].set_xlabel("T (steps of history given to the model)")
    axes[1, 1].set_ylabel("test accuracy")
    axes[1, 1].set_title(r"full-evolution model on $P(x, t\leq T)$", fontsize=10)
    axes[1, 1].set_ylim(0.45, 1.02)

    for ax in axes.ravel():
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = save_figure(fig, "mlp_evolution_summary", overwrite=args.overwrite)
    plt.close(fig)
    return path


def run_channel(name: str, cfg: dict, norm_mode: str, args) -> dict:
    spec = CHANNELS[name]
    print_header(name, spec, cfg, norm_mode)

    t0 = time.perf_counter()
    ds = make_evolution_dataset(
        cfg["n"],
        name,
        tuple(spec["window_delocalized"]),
        tuple(spec["window_localized"]),
        n_samples=cfg["n_samples"],
        theta_0=cfg["theta_0"],
        seed=cfg["seed"],
        parity=cfg["parity"],
    )
    controls = np.linspace(*spec["probe_range"], cfg["n_probe_controls"])
    probe = make_probe_evolutions(
        cfg["n"],
        name,
        controls,
        n_realizations=cfg["n_probe_realizations"],
        theta_0=cfg["theta_0"],
        seed=cfg["probe_seed"],
        parity=cfg["parity"],
    )
    print(f"\nsimulated {len(ds)} labelled + {probe.P.shape[0] * probe.P.shape[1]} probe "
          f"evolutions in {time.perf_counter() - t0:.1f}s "
          f"({ds.P.nbytes / 1e6:.0f} + {probe.P.nbytes / 1e6:.0f} MB)")

    idx = np.arange(len(ds))
    itr, ite = train_test_split(
        idx, test_size=cfg["test_size"], random_state=cfg["random_state"], stratify=ds.y
    )

    t0 = time.perf_counter()
    snap = snapshot_sweep(ds, probe, itr, ite, cfg)
    print(f"fitted {len(snap['times'])} snapshot models in {time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()
    pre = prefix_sweep(ds, probe, itr, ite, cfg, norm_mode)
    print(f"fitted {len(pre['times'])} full-evolution models in "
          f"{time.perf_counter() - t0:.1f}s")

    t_usable = usable_from(snap, cfg)

    res = {
        "moi": moi_crossover(probe, cfg),
        "channel": name,
        "controls": controls,
        "times_all": ds.times.astype(float),
        "snapshot": snap,
        "prefix": pre,
        "snapshot_critical": extract_critical(snap, controls, t_usable),
        "prefix_critical": extract_critical(pre, controls, t_usable),
        "onset_time": onset(snap["times"], snap["accuracy"], cfg["onset_threshold"]),
        "usable_from": t_usable,
        "mean_evolution": {
            DELOCALIZED: ds.P[ds.y == DELOCALIZED].mean(axis=0),
            LOCALIZED: ds.P[ds.y == LOCALIZED].mean(axis=0),
        },
    }
    report(name, spec, cfg, res)
    return res


def to_record(name: str, cfg: dict, norm_mode: str, res: dict) -> dict:
    """JSON-serialisable provenance for everything printed (CLAUDE.md Sec. 9)."""
    spec = CHANNELS[name]

    def arr(a):
        return [None if not np.isfinite(v) else float(v) for v in np.asarray(a, float)]

    return {
        "channel": name,
        "control_parameter": spec["control"],
        "n": cfg["n"],
        "n_steps": cfg["n"],
        "n_samples": cfg["n_samples"],
        "theta_0": cfg["theta_0"],
        "window_delocalized": list(spec["window_delocalized"]),
        "window_localized": list(spec["window_localized"]),
        "probe_range": list(spec["probe_range"]),
        "n_probe_controls": cfg["n_probe_controls"],
        "n_probe_realizations": cfg["n_probe_realizations"],
        "seed": cfg["seed"],
        "probe_seed": cfg["probe_seed"],
        "random_state": cfg["random_state"],
        "max_iter": cfg["max_iter"],
        "spacetime_normalization": norm_mode,
        "hidden_layer_sizes": [400, 200, 100, 50],
        "alpha": 0.001,
        "onset_time": res["onset_time"],
        "usable_from": res["usable_from"],
        "onset_threshold": cfg["onset_threshold"],
        "full_evolution_train_accuracy": float(res["prefix"]["train_accuracy"][-1]),
        "full_evolution_test_accuracy": float(res["prefix"]["test_accuracy"][-1]),
        "final_snapshot_test_accuracy": float(res["snapshot"]["accuracy"][-1]),
        "controls": arr(res["controls"]),
        "snapshot_times": arr(res["snapshot"]["times"]),
        "snapshot_accuracy": arr(res["snapshot"]["accuracy"]),
        "critical_control_vs_time": arr(res["snapshot_critical"]["critical_control"]),
        "crossover_time_vs_control": arr(res["snapshot_critical"]["crossover_time"]),
        "crossover_time_sustained": arr(
            res["snapshot_critical"]["crossover_time_sustained"]
        ),
        "moi_ratio": cfg["moi_ratio"],
        "moi_crossover_time": arr(res["moi"]["crossover_time"]),
        "prefix_times": arr(res["prefix"]["times"]),
        "prefix_test_accuracy": arr(res["prefix"]["test_accuracy"]),
        "prefix_critical_control": arr(res["prefix_critical"]["critical_control"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--channel", choices=sorted(CHANNELS), action="append",
                        help="repeatable; default is all three")
    parser.add_argument("--norm", choices=EVOLUTION_NORMALIZATIONS, default="slice",
                        help="how P_max=1 is applied to a space-time sample")
    parser.add_argument("--quick", action="store_true",
                        help="small/fast settings for checking the plumbing")
    parser.add_argument("--no-save", action="store_true", help="skip the figures")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace existing figures of the same name")
    parser.add_argument("--save-results", metavar="PATH", type=Path)
    parser.add_argument("--n", type=int)
    parser.add_argument("--n-samples", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--random-state", type=int)
    parser.add_argument("--slice-stride", type=int)
    args = parser.parse_args()

    cfg = dict(DEFAULTS)
    if args.quick:
        cfg.update(QUICK_OVERRIDES)
    for key, value in (("n", args.n), ("n_samples", args.n_samples),
                       ("seed", args.seed), ("random_state", args.random_state),
                       ("slice_stride", args.slice_stride)):
        if value is not None:
            cfg[key] = value

    names = args.channel or list(CHANNELS)
    records, results = [], {}
    for name in names:
        res = run_channel(name, cfg, args.norm, args)
        results[name] = res
        records.append(to_record(name, cfg, args.norm, res))
        if not args.no_save:
            print(f"\nsaved {plot_channel(name, CHANNELS[name], cfg, res, args)}")
        print()

    print("=" * 78)
    print(f"  {'channel':<20} {'full evol.':>10} {'snapshot':>9} "
          f"{'onset t':>8} {'critical':>9}")
    for name, res in results.items():
        crit = res["snapshot_critical"]["critical_control"][-1]
        t_on = res["onset_time"]
        print(f"  {name:<20} {res['prefix']['test_accuracy'][-1]:>10.4f} "
              f"{res['snapshot']['accuracy'][-1]:>9.4f} "
              f"{('--' if t_on is None else f'{t_on:.0f}'):>8} {_fmt(crit):>9}")
    print("=" * 78)
    print("  'critical' is the control value at the final step, Sec. 7.4 MLP rule,\n"
          "  conditional on the placeholder windows. Not a Paper A Table I number.")

    if len(results) > 1 and not args.no_save:
        print(f"\nsaved {plot_summary(results, cfg, args)}")

    if args.save_results:
        args.save_results.parent.mkdir(parents=True, exist_ok=True)
        existing = (json.loads(args.save_results.read_text())
                    if args.save_results.exists() else [])
        existing.extend(records)
        args.save_results.write_text(json.dumps(existing, indent=2))
        print(f"appended {len(records)} record(s) to {args.save_results}")


if __name__ == "__main__":
    main()
