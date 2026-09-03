#!/usr/bin/env python3
"""Train the SVM on one randomness channel and report what it learned.

Thin entry point: parses args, loads the config, calls into ``src`` (CLAUDE.md
Sec. 8). Phase 1 only -- this trains and scores on the two extreme regimes and
deliberately does not touch the transition regime.

    python3 scripts/train_svm.py configs/svm_discrete_coin.json
    python3 scripts/train_svm.py configs/svm_discrete_coin.json --plot
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.generate import DELOCALIZED, LOCALIZED, make_dataset  # noqa: E402
from models.svm import SVMClassifier  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--plot", action="store_true", help="show weights and examples")
    parser.add_argument("--save", metavar="NAME", help="save the plot to figures/NAME.png")
    parser.add_argument("--overwrite", action="store_true")
    # Overrides for quick exploration. Anything you intend to quote should go
    # in a config file instead, so the run stays reproducible from it alone.
    parser.add_argument("--channel", help="override the config's channel")
    parser.add_argument("--n", type=int, help="override the lattice parameter")
    parser.add_argument("--n-samples", type=int, help="override the sample count")
    parser.add_argument("--seed", type=int, help="override the master seed")
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="turn off the P_max = 1 normalization (Paper A: no effect on the SVM)",
    )
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text())
    for key, value in (
        ("channel", args.channel),
        ("n", args.n),
        ("n_samples", args.n_samples),
        ("seed", args.seed),
    ):
        if value is not None:
            cfg[key] = value
    if args.no_normalize:
        cfg["normalize_pmax"] = False

    print(f"config              {args.config}")
    print(f"channel             {cfg['channel']}")
    print(f"lattice             n={cfg['n']}  ({2 * cfg['n'] + 1} sites, {cfg['parity']})")
    print(f"theta_0             {cfg['theta_0']:.6f}  ({cfg.get('theta_0_expr', '')})")
    print(f"window delocalized  {tuple(cfg['window_delocalized'])}")
    print(f"window localized    {tuple(cfg['window_localized'])}")
    print(f"samples             {cfg['n_samples']}  (seed {cfg['seed']})")
    print(f"P_max normalization {cfg['normalize_pmax']}")
    print()

    print("simulating ...", flush=True)
    ds = make_dataset(
        cfg["n"],
        cfg["channel"],
        tuple(cfg["window_delocalized"]),
        tuple(cfg["window_localized"]),
        n_samples=cfg["n_samples"],
        theta_0=cfg["theta_0"],
        seed=cfg["seed"],
        parity=cfg.get("parity", "odd"),
    )

    X_train, X_test, y_train, y_test = train_test_split(
        ds.X,
        ds.y,
        test_size=cfg["test_size"],
        random_state=cfg["random_state"],
        stratify=ds.y,
    )

    clf = SVMClassifier(
        normalize=cfg["normalize_pmax"],
        alpha=cfg["alpha"],
        max_iter=cfg["max_iter"],
        tol=cfg["tol"],
        random_state=cfg["random_state"],
    )
    clf.fit(X_train, y_train)

    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    print(f"\ntrain accuracy      {train_acc:.4f}  ({len(y_train)} samples)")
    print(f"test accuracy       {test_acc:.4f}  ({len(y_test)} samples)")

    predicted = clf.predict(X_test)
    print("\nconfusion (rows = true, cols = predicted)")
    print("                 pred deloc   pred loc")
    for label, name in ((DELOCALIZED, "true deloc"), (LOCALIZED, "true loc  ")):
        row = [int(np.sum((y_test == label) & (predicted == p))) for p in (0, 1)]
        print(f"    {name}      {row[0]:>8}   {row[1]:>8}")

    decision = clf.decision_function(X_test)
    proba = clf.predict_proba(X_test)[:, DELOCALIZED]
    graded = int(np.sum((proba > 1e-9) & (proba < 1 - 1e-9)))
    print(f"\n|decision| range     [{np.abs(decision).min():.3f}, {np.abs(decision).max():.3f}]")
    print(f"graded predictions   {graded} / {len(proba)}")
    print(
        "  modified_huber saturates to exactly 0/1 outside |decision| >= 1, so on\n"
        "  well-separated extremes almost everything is hard. The graded band is\n"
        "  what Phase 2's confusion point is read from."
    )

    print("\nmost extreme test samples")
    for idx in (np.argmin(decision), np.argmax(decision)):
        print(
            f"  decision={decision[idx]:+9.3f}  P(deloc)={proba[idx]:.4f}  "
            f"true={'deloc' if y_test[idx] == DELOCALIZED else 'loc'}"
        )

    if args.plot or args.save:
        _plot(clf, ds, args)


def _plot(clf: SVMClassifier, ds, args: argparse.Namespace) -> None:
    import matplotlib.pyplot as plt

    from plotting.style import save_figure

    positions = np.arange(-ds.n, ds.n + 1)
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    top.plot(positions, ds.X[ds.y == DELOCALIZED][0], color="blue", label="delocalized")
    top.plot(
        positions,
        ds.X[ds.y == LOCALIZED][0],
        color="red",
        linestyle="dashed",
        label="localized",
    )
    top.set_ylabel("P(x)")
    top.set_title(f"SVM training classes and learned weights, {ds.channel}, N={ds.n}")
    top.legend()

    bottom.plot(positions, clf.weights, color="black", linewidth=0.8)
    bottom.axhline(0.0, color="grey", linewidth=0.5)
    bottom.set_xlabel("x")
    bottom.set_ylabel("SVM weight")
    bottom.set_title("positive weight pushes toward 'localized'", fontsize=9)

    fig.tight_layout()
    if args.save:
        print(f"\nsaved {save_figure(fig, args.save, overwrite=args.overwrite)}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
