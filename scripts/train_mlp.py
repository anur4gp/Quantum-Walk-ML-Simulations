#!/usr/bin/env python3
"""Train / tune the MLP on one randomness channel.

Phase 1: the two extreme regimes only, never the transition regime.

    python3 scripts/train_mlp.py [--regime NAME] [--all] [--plot]
    python3 scripts/train_mlp.py --list                 # regimes and grids
    python3 scripts/train_mlp.py --grid coarse [--cv 5] # GridSearchCV
    python3 scripts/train_mlp.py --n 300 --alpha 0.01 --no-normalize

Regimes live in ``REGIMES`` below, grids in ``src/models/mlp.py``. Everything
printed is a function of (regime, grid, seed, random_state), all echoed in the
header; anything you intend to quote belongs in a named regime rather than a
command-line override (CLAUDE.md Sec. 9).
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.generate import DELOCALIZED, LOCALIZED, make_dataset  # noqa: E402
from models.mlp import PARAM_GRIDS, MLPClassifier  # noqa: E402

PI = np.pi

# A regime is one labelled two-class problem: channel, lattice size, and the
# two control-parameter windows the classes are drawn from.
#
# THE WINDOWS ARE PLACEHOLDERS from configs/svm_*.json. Choosing them is a
# research decision and a recorded result (CLAUDE.md Sec. 6, Sec. 9); every
# accuracy below is conditional on them.

THETA_0 = PI / 6  # Paper A default (CLAUDE.md Sec. 2.1)

REGIMES: dict[str, dict] = {
    "discrete_coin": {
        "channel": "discrete_coin",
        "n": 80,
        "window_delocalized": (0.0, 0.05),
        "window_localized": (0.45, THETA_0),
        "note": "Paper A 'Jittered'. delta_theta in [0, theta_0].",
    },
    "discrete_coin_narrow": {
        "channel": "discrete_coin",
        "n": 80,
        "window_delocalized": (0.0, 0.02),
        "window_localized": (0.50, THETA_0),
        "note": "Narrower windows: less information per class.",
    },
    "discrete_coin_wide": {
        "channel": "discrete_coin",
        "n": 80,
        "window_delocalized": (0.0, 0.15),
        "window_localized": (0.30, THETA_0),
        "note": "Wider windows: risks eating into the transition regime.",
    },
    "continuous_coin": {
        "channel": "continuous_coin",
        "n": 80,
        "window_delocalized": (0.0, 0.05),
        "window_localized": (0.45, THETA_0),
        "note": "Paper A 'Uniform Jittered'. One-sided uniform draw each step.",
    },
    "random_translation": {
        "channel": "random_translation",
        "n": 80,
        "window_delocalized": (0.0, 0.02),
        "window_localized": (0.45, 0.5),
        "note": (
            "KNOWN-HARD CASE (CLAUDE.md Sec. 3). A clean result here means label "
            "leakage or a window that swallowed the transition."
        ),
    },
    "stress": {
        "channel": "discrete_coin",
        "n": 80,
        "n_samples": 600,
        "window_delocalized": (0.0, 0.24),
        "window_localized": (0.26, THETA_0),
        "note": (
            "DIAGNOSTIC ONLY. Windows squeezed until they almost touch, the only "
            "Phase-1 setting where a grid scores anything but 1.0000. They straddle "
            "the transition, so never quote a number from them."
        ),
    },
    "quick": {
        "channel": "discrete_coin",
        "n": 40,
        "n_samples": 400,
        "window_delocalized": (0.0, 0.05),
        "window_localized": (0.45, THETA_0),
        "note": "Small and fast; for checking the plumbing.",
    },
    "large_lattice": {
        "channel": "discrete_coin",
        "n": 300,
        "n_samples": 600,
        "window_delocalized": (0.0, 0.05),
        "window_localized": (0.45, THETA_0),
        "note": "Paper A kept the layer sizes fixed from N=80 to N=1000; the check.",
    },
}

DEFAULT_REGIME = "discrete_coin"

#: Filled in for any key a regime does not override.
REGIME_DEFAULTS: dict = {
    "theta_0": THETA_0,
    "parity": "odd",
    "n_samples": 1800,  # Paper A's value (CLAUDE.md Sec. 6)
    "test_size": 0.2,  # 80/20 split (CLAUDE.md Sec. 6)
    "seed": 0,  # master seed for the simulations
    "random_state": 0,  # sklearn's seed: weight init, shuffling, CV splits
    "normalize_pmax": True,  # P_max = 1 (Paper A, Sec. III B 4)
    "max_iter": 400,
}


def resolve_regime(name: str, cfg_path: Path | None, args: argparse.Namespace) -> dict:
    """Merge defaults, the named regime (or a JSON config), then CLI overrides."""
    cfg = dict(REGIME_DEFAULTS)
    if cfg_path is not None:
        cfg.update(json.loads(cfg_path.read_text()))
        cfg["regime"] = str(cfg_path)
    else:
        if name not in REGIMES:
            raise SystemExit(
                f"unknown regime {name!r}; known: {', '.join(REGIMES)}\n"
                f"(run with --list to see them with their windows)"
            )
        cfg.update(REGIMES[name])
        cfg["regime"] = name

    for key, value in (
        ("n", args.n),
        ("n_samples", args.n_samples),
        ("seed", args.seed),
        ("random_state", args.random_state),
        ("max_iter", args.max_iter),
        ("alpha", args.alpha),
    ):
        if value is not None:
            cfg[key] = value
    if args.hidden is not None:
        cfg["hidden_layer_sizes"] = tuple(int(v) for v in args.hidden.split(","))
    if args.no_normalize:
        cfg["normalize_pmax"] = False
    return cfg


def build_estimator(cfg: dict) -> MLPClassifier:
    """Unfitted MLP with the regime's settings; Paper A values where unset."""
    kwargs: dict = {
        "normalize": cfg["normalize_pmax"],
        "max_iter": cfg["max_iter"],
        "random_state": cfg["random_state"],
    }
    for key in ("hidden_layer_sizes", "alpha"):
        if key in cfg:
            kwargs[key] = cfg[key]
    return MLPClassifier(**kwargs)


def load_data(cfg: dict):
    """Simulate the regime's dataset and split it, stratified."""
    t0 = time.perf_counter()
    ds = make_dataset(
        cfg["n"],
        cfg["channel"],
        tuple(cfg["window_delocalized"]),
        tuple(cfg["window_localized"]),
        n_samples=cfg["n_samples"],
        theta_0=cfg["theta_0"],
        seed=cfg["seed"],
        parity=cfg["parity"],
    )
    elapsed = time.perf_counter() - t0
    split = train_test_split(
        ds.X,
        ds.y,
        test_size=cfg["test_size"],
        random_state=cfg["random_state"],
        stratify=ds.y,
    )
    return ds, split, elapsed


def print_header(cfg: dict) -> None:
    print("=" * 74)
    print(f"regime              {cfg['regime']}")
    if cfg.get("note"):
        for line in textwrap.wrap(cfg["note"], 54):
            print(f"                    {line}")
    print(f"channel             {cfg['channel']}")
    print(f"lattice             n={cfg['n']}  ({2 * cfg['n'] + 1} sites, {cfg['parity']})")
    print(f"theta_0             {cfg['theta_0']:.6f}")
    print(f"window delocalized  {tuple(cfg['window_delocalized'])}   -> label 0")
    print(f"window localized    {tuple(cfg['window_localized'])}   -> label 1")
    print(f"samples             {cfg['n_samples']}  (sim seed {cfg['seed']}, "
          f"test_size {cfg['test_size']})")
    print(f"P_max normalization {cfg['normalize_pmax']}")
    print(f"hidden layers       {cfg.get('hidden_layer_sizes', '(400, 200, 100, 50)  [Paper A]')}")
    print(f"alpha               {cfg.get('alpha', '0.001  [Paper A]')}")
    print(f"max_iter            {cfg['max_iter']}   (sklearn seed {cfg['random_state']})")
    print("=" * 74)


def report_fit(clf: MLPClassifier, split, cfg: dict) -> dict:
    """Accuracies, confusion matrix, and how confident the network is."""
    X_train, X_test, y_train, y_test = split
    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)

    print(f"\ntrain accuracy      {train_acc:.4f}  ({len(y_train)} samples)")
    print(f"test accuracy       {test_acc:.4f}  ({len(y_test)} samples)")
    print(f"iterations          {clf.n_iter} / {cfg['max_iter']}"
          + ("   <-- hit the cap, did NOT converge" if clf.n_iter >= cfg["max_iter"] else ""))
    print(f"final training loss {clf.loss_curve[-1]:.6f}")

    predicted = clf.predict(X_test)
    print("\nconfusion (rows = true, cols = predicted)")
    print("                 pred deloc   pred loc")
    for label, name in ((DELOCALIZED, "true deloc"), (LOCALIZED, "true loc  ")):
        row = [int(np.sum((y_test == label) & (predicted == p))) for p in (0, 1)]
        print(f"    {name}      {row[0]:>8}   {row[1]:>8}")

    proba = clf.predict_proba(X_test)[:, DELOCALIZED]
    graded = int(np.sum((proba > 0.01) & (proba < 0.99)))
    print(f"\nP(deloc) in (0.01, 0.99)   {graded} / {len(proba)} test samples")
    print(f"mean |P(deloc) - 0.5|      {np.abs(proba - 0.5).mean():.4f}  (0.5 = fully confident)")
    print("\n  Near-perfect accuracy on the extremes shows only that the pipeline")
    print("  works: the classes are one-peak vs two-peak (Paper A, Sec. III B 1).")
    return {"train_accuracy": train_acc, "test_accuracy": test_acc, "n_iter": clf.n_iter}


def run_grid(cfg: dict, split, grid_name: str, cv: int, n_jobs: int) -> tuple:
    """Exhaustive GridSearchCV over ``PARAM_GRIDS[grid_name]``."""
    if grid_name not in PARAM_GRIDS:
        raise SystemExit(
            f"unknown grid {grid_name!r}; known: {', '.join(PARAM_GRIDS)}\n"
            f"(grids live in src/models/mlp.py, PARAM_GRIDS)"
        )
    grid = PARAM_GRIDS[grid_name]
    X_train, X_test, y_train, y_test = split

    n_combos = int(np.prod([len(v) for v in grid.values()]))
    print(f"\ngrid                {grid_name}")
    for key, values in grid.items():
        print(f"  {key:<22}{values}")
    print(f"  {'combinations':<22}{n_combos}  x {cv}-fold CV = {n_combos * cv} fits")

    search = GridSearchCV(
        build_estimator(cfg),
        grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=n_jobs,
        refit=True,
        return_train_score=True,
    )
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        # A non-converged fit is a legitimate grid point; n_iter flags it.
        warnings.simplefilter("ignore", ConvergenceWarning)
        search.fit(X_train, y_train)
    print(f"  {'elapsed':<22}{time.perf_counter() - t0:.1f}s")

    res = search.cv_results_
    order = np.argsort(-res["mean_test_score"])
    print("\nCV results, best first")
    print(f"  {'rank':>4}  {'CV acc':>8} {'+-std':>7}  {'train acc':>9}  params")
    for rank, i in enumerate(order, start=1):
        print(
            f"  {rank:>4}  {res['mean_test_score'][i]:8.4f} {res['std_test_score'][i]:7.4f}  "
            f"{res['mean_train_score'][i]:9.4f}  {res['params'][i]}"
        )

    print(f"\nbest params         {search.best_params_}")
    print(f"best CV accuracy    {search.best_score_:.4f}")
    best = search.best_estimator_
    print(f"held-out test acc   {best.score(X_test, y_test):.4f}  "
          f"(refit on the full training split)")

    spread = float(res["mean_test_score"].max() - res["mean_test_score"].min())
    if spread < 0.005:
        print(f"\n  CV accuracy spans only {spread:.4f} across the grid: every")
        print("  architecture saturates on the extremes, so this is not measuring")
        print("  anything. Re-run once Phase 2 gives a score to select on.")
    return search, {
        "grid": grid_name,
        "cv": cv,
        "best_params": {k: list(v) if isinstance(v, tuple) else v
                        for k, v in search.best_params_.items()},
        "best_cv_accuracy": float(search.best_score_),
        "test_accuracy": float(best.score(X_test, y_test)),
        "mean_test_score": [float(v) for v in res["mean_test_score"]],
        "params": [str(p) for p in res["params"]],
    }


def plot(clf: MLPClassifier, ds, cfg: dict, search, args: argparse.Namespace) -> None:
    """Training classes, loss curve, first-layer sensitivity, grid scores."""
    import matplotlib.pyplot as plt

    from plotting.style import save_figure

    n_panels = 4 if search is not None else 3
    fig, axes = plt.subplots(n_panels, 1, figsize=(8, 3.0 * n_panels))
    positions = np.arange(-ds.n, ds.n + 1)

    ax = axes[0]
    ax.plot(positions, ds.X[ds.y == DELOCALIZED][0], color="blue", label="delocalized (0)")
    ax.plot(positions, ds.X[ds.y == LOCALIZED][0], color="red", linestyle="dashed",
            label="localized (1)")
    ax.set_xlabel("x")
    ax.set_ylabel("P(x)")
    ax.set_title(f"one training sample per class -- {cfg['channel']}, N={ds.n}")
    ax.legend()

    ax = axes[1]
    ax.plot(clf.loss_curve, color="black", linewidth=1.0)
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("training loss")
    ax.set_title(f"convergence -- {clf.n_iter} iterations")
    ax.grid(True, alpha=0.25)

    ax = axes[2]
    ax.plot(positions, clf.input_weight_magnitude, color="black", linewidth=0.8)
    ax.set_xlabel("x")
    ax.set_ylabel(r"$\|W_1\|_2$ per site")
    ax.set_title("first-layer sensitivity (not a decision boundary)", fontsize=9)

    if search is not None:
        ax = axes[3]
        res = search.cv_results_
        order = np.argsort(-res["mean_test_score"])
        labels = [str(res["params"][i]) for i in order]
        ax.errorbar(
            np.arange(len(order)),
            res["mean_test_score"][order],
            yerr=res["std_test_score"][order],
            marker="o", markersize=4, linestyle="none", color="black", capsize=3,
        )
        ax.set_xticks(np.arange(len(order)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=6)
        ax.set_ylabel("CV accuracy")
        ax.set_title(f"grid '{args.grid}' -- {cfg['regime']}", fontsize=9)
        ax.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    if args.save:
        print(f"\nsaved {save_figure(fig, args.save, overwrite=args.overwrite)}")
    else:
        plt.show()


def run_one(cfg: dict, args: argparse.Namespace) -> dict:
    print_header(cfg)
    ds, split, sim_seconds = load_data(cfg)
    print(f"\nsimulated {len(ds)} walks in {sim_seconds:.1f}s")

    search = None
    record: dict = {"regime": cfg["regime"], "channel": cfg["channel"], "n": cfg["n"],
                    "n_samples": cfg["n_samples"], "seed": cfg["seed"],
                    "random_state": cfg["random_state"],
                    "normalize_pmax": cfg["normalize_pmax"],
                    "window_delocalized": list(cfg["window_delocalized"]),
                    "window_localized": list(cfg["window_localized"])}

    if args.grid:
        search, grid_record = run_grid(cfg, split, args.grid, args.cv, args.n_jobs)
        record.update(grid_record)
        clf = search.best_estimator_
    else:
        clf = build_estimator(cfg)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            clf.fit(split[0], split[2])

    record.update(report_fit(clf, split, cfg))

    if args.plot or args.save:
        plot(clf, ds, cfg, search, args)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("config", nargs="?", type=Path,
                        help="optional JSON config; overrides --regime")
    parser.add_argument("--regime", default=DEFAULT_REGIME, help=f"default: {DEFAULT_REGIME}")
    parser.add_argument("--all", action="store_true", help="run every regime, print a summary")
    parser.add_argument("--list", action="store_true", help="list regimes and grids, then exit")
    parser.add_argument("--grid", metavar="NAME", help=f"GridSearchCV over one of: "
                        f"{', '.join(PARAM_GRIDS)}")
    parser.add_argument("--cv", type=int, default=3, help="CV folds for --grid (default 3)")
    parser.add_argument("--n-jobs", type=int, default=-1, help="parallel fits (default all cores)")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--save", metavar="NAME", help="save the plot to figures/NAME.png")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-results", metavar="PATH", type=Path,
                        help="append the run record to a JSON file")
    # Overrides for exploration only; anything you quote belongs in a regime.
    parser.add_argument("--n", type=int)
    parser.add_argument("--n-samples", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--random-state", type=int)
    parser.add_argument("--max-iter", type=int)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--hidden", metavar="A,B,C", help="hidden layer sizes, comma separated")
    parser.add_argument("--no-normalize", action="store_true",
                        help="turn off the P_max = 1 normalization")
    args = parser.parse_args()

    if args.list:
        print("regimes (scripts/train_mlp.py, REGIMES):\n")
        for name, r in REGIMES.items():
            mark = " <- default" if name == DEFAULT_REGIME else ""
            print(f"  {name}{mark}")
            print(f"      channel {r['channel']}, n={r['n']}, "
                  f"samples={r.get('n_samples', REGIME_DEFAULTS['n_samples'])}")
            print(f"      windows {tuple(r['window_delocalized'])} vs "
                  f"{tuple(r['window_localized'])}")
            for line in textwrap.wrap(r.get("note", ""), 66):
                print(f"      {line}")
            print()
        print("grids (src/models/mlp.py, PARAM_GRIDS):\n")
        for name, g in PARAM_GRIDS.items():
            n_combos = int(np.prod([len(v) for v in g.values()]))
            print(f"  {name:<16} {n_combos:>3} combinations over {', '.join(g)}")
        return

    names = list(REGIMES) if args.all else [args.regime]
    records = []
    for name in names:
        records.append(run_one(resolve_regime(name, args.config, args), args))
        print()

    if args.all:
        print("=" * 74)
        print(f"  {'regime':<24} {'channel':<20} {'n':>5} {'train':>7} {'test':>7}")
        for r in records:
            print(f"  {r['regime']:<24} {r['channel']:<20} {r['n']:>5} "
                  f"{r['train_accuracy']:>7.4f} {r['test_accuracy']:>7.4f}")
        print("=" * 74)

    if args.save_results:
        args.save_results.parent.mkdir(parents=True, exist_ok=True)
        existing = (json.loads(args.save_results.read_text())
                    if args.save_results.exists() else [])
        existing.extend(records)
        args.save_results.write_text(json.dumps(existing, indent=2))
        print(f"appended {len(records)} record(s) to {args.save_results}")


if __name__ == "__main__":
    main()
