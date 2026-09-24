#!/usr/bin/env python3
"""Classical unbiased random walk on 1D -- the diffusive baseline.

Self-contained on purpose: ``src/qw/`` is the quantum machinery and a classical
walk shares none of it. This is a reference curve, not a dependency. It imports
the quantum walk only for the side-by-side figure.

It shows two things: P(x) after N steps (classical single peak vs the quantum
two-peak), and MoI vs N (classical exactly N, quantum ~ N^2). That contrast is
also why a localized quantum walk looks classical (CLAUDE.md Sec. 2.4).

Only sites of the same parity as N are reachable, so plotting every site draws
a comb. This plots the reachable sub-lattice by default; ``--all-sites``
restores the comb of the older reference figures. Plotting choice only -- every
observable is computed on the full lattice.

    python3 scripts/legacy_classical_walk.py [--steps 300] [--mc 20000]
    python3 scripts/legacy_classical_walk.py [--all-sites] [--save-prefix NAME]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binom

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from plotting.style import CHANNEL_STYLE, save_figure  # noqa: E402
from qw.observables import probability  # noqa: E402
from qw.operators import PHI_DEFAULT  # noqa: E402
from qw.walk import lattice_positions, run_walk  # noqa: E402

#: Local, not in plotting.style.CHANNEL_STYLE: that dict is keyed by quantum
#: randomness channel and the classical walk is not one of them.
CLASSICAL_STYLE = {"label": "Classical RW", "color": "black", "linestyle": "dashed"}

#: Hadamard-equivalent coin angle, as in the reference figures.
THETA_HADAMARD: float = np.pi / 4


def classical_exact(n_steps: int, p_right: float = 0.5) -> np.ndarray:
    """Exact P(x) after ``n_steps`` steps, on the same odd lattice as the QW.

    After N steps with k right steps the walker is at x = 2k - N, so
    P(x) = Binomial(N, p)[(x + N) / 2]; wrong-parity sites are exactly 0. No
    sampling noise, and directly comparable to the quantum curve site by site.
    """
    x = lattice_positions(n_steps, "odd")
    prob = np.zeros(x.size, dtype=np.float64)
    reachable = (x.astype(np.int64) + n_steps) % 2 == 0
    k = (x[reachable] + n_steps) / 2
    prob[reachable] = binom.pmf(k, n_steps, p_right)
    return prob


def classical_monte_carlo(
    n_steps: int, n_walkers: int, rng: np.random.Generator, p_right: float = 0.5
) -> np.ndarray:
    """Monte Carlo estimate of the same distribution, as a sanity overlay.

    Converges to :func:`classical_exact` as ``n_walkers`` grows.
    """
    steps = rng.choice(np.array([-1, 1]), size=(n_walkers, n_steps), p=[1 - p_right, p_right])
    final = steps.sum(axis=1)
    counts = np.bincount(final + n_steps, minlength=2 * n_steps + 1)
    return counts.astype(np.float64) / n_walkers


def moment_of_inertia(prob: np.ndarray, positions: np.ndarray) -> float:
    r""":math:`\sum_x x^2 P(x)`. Same definition as ``qw.observables`` (CLAUDE.md Sec. 2.5)."""
    return float(np.sum(positions**2 * prob))


def reachable_mask(n_steps: int) -> np.ndarray:
    """Sites reachable after ``n_steps`` steps: x and N share parity."""
    x = lattice_positions(n_steps, "odd").astype(np.int64)
    return (x + n_steps) % 2 == 0


def figure_distribution(
    n_steps: int, theta: float, n_mc: int, seed: int, all_sites: bool = False
) -> "tuple[object, dict]":
    """Classical vs. pure quantum ``P(x)`` at the same step count."""
    import matplotlib.pyplot as plt

    positions = lattice_positions(n_steps, "odd")
    p_classical = classical_exact(n_steps)

    qw = run_walk(
        n_steps,
        "pure",
        theta_0=theta,
        parity="odd",
        n_steps=n_steps,
        phi1=PHI_DEFAULT,
        phi2=PHI_DEFAULT,
    )
    p_quantum = probability(qw.psi_plus, qw.psi_minus)

    # The mask is for legibility; the observables above use the full lattice.
    keep = np.ones(positions.size, dtype=bool) if all_sites else reachable_mask(n_steps)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        positions[keep],
        p_quantum[keep],
        label=CHANNEL_STYLE["pure"]["label"],
        color=CHANNEL_STYLE["pure"]["color"],
        linestyle=CHANNEL_STYLE["pure"]["linestyle"],
        linewidth=0.9,
    )
    ax.plot(positions[keep], p_classical[keep], linewidth=1.4, **CLASSICAL_STYLE)
    if n_mc:
        rng = np.random.default_rng(seed)
        ax.plot(
            positions[keep],
            classical_monte_carlo(n_steps, n_mc, rng)[keep],
            label=f"Classical RW, MC ({n_mc:,} walkers)",
            color="grey",
            linestyle="none",
            marker=".",
            markersize=2,
            alpha=0.6,
        )
    ax.set_xlabel("x")
    ax.set_ylabel("P(x)")
    ax.set_title(f"Classical vs. Quantum Walk, {n_steps} Steps")
    ax.legend()
    fig.tight_layout()

    stats = {
        "moi_classical": moment_of_inertia(p_classical, positions),
        "moi_quantum": moment_of_inertia(p_quantum, positions),
        "sum_classical": float(p_classical.sum()),
        "sum_quantum": float(p_quantum.sum()),
    }
    return fig, stats


def figure_spreading(
    max_steps: int, theta: float
) -> "tuple[object, dict]":
    """MoI vs. number of steps, log-log: classical slope 1, quantum slope 2."""
    import matplotlib.pyplot as plt

    ns = np.unique(np.round(np.logspace(np.log10(8), np.log10(max_steps), 14)).astype(int))
    moi_classical = np.array(
        [moment_of_inertia(classical_exact(n), lattice_positions(n, "odd")) for n in ns]
    )
    moi_quantum = np.empty(ns.size)
    for i, n in enumerate(ns):
        res = run_walk(int(n), "pure", theta_0=theta, parity="odd", n_steps=int(n))
        moi_quantum[i] = moment_of_inertia(
            probability(res.psi_plus, res.psi_minus), res.positions
        )

    slope_classical = float(np.polyfit(np.log(ns), np.log(moi_classical), 1)[0])
    slope_quantum = float(np.polyfit(np.log(ns), np.log(moi_quantum), 1)[0])

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.loglog(
        ns,
        moi_quantum,
        marker="o",
        markersize=4,
        color=CHANNEL_STYLE["pure"]["color"],
        label=f"Pure QW  (slope {slope_quantum:.3f})",
    )
    ax.loglog(
        ns,
        moi_classical,
        marker="s",
        markersize=4,
        color="black",
        linestyle="dashed",
        label=f"Classical RW  (slope {slope_classical:.3f})",
    )
    ax.set_xlabel("N (steps)")
    ax.set_ylabel(r"MoI $= \sum_x x^2 P(x)$")
    ax.set_title("Ballistic vs. diffusive spreading")
    ax.legend()
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()

    return fig, {
        "ns": ns,
        "moi_classical": moi_classical,
        "moi_quantum": moi_quantum,
        "slope_classical": slope_classical,
        "slope_quantum": slope_quantum,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=300, help="steps for the P(x) figure")
    parser.add_argument(
        "--max-steps", type=int, default=300, help="largest N in the spreading figure"
    )
    parser.add_argument(
        "--theta",
        type=float,
        default=THETA_HADAMARD,
        help="coin angle of the quantum comparison walk (default pi/4, Hadamard)",
    )
    parser.add_argument("--mc", type=int, default=0, help="overlay N Monte Carlo walkers")
    parser.add_argument(
        "--all-sites",
        action="store_true",
        help="plot the unreachable (identically zero) sites too, as the older figures do",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save-prefix",
        metavar="NAME",
        help="save to figures/NAME_distribution.png and figures/NAME_spreading.png",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    fig_dist, stats = figure_distribution(
        args.steps, args.theta, args.mc, args.seed, all_sites=args.all_sites
    )
    print(f"N = {args.steps} steps, theta = {args.theta:.6f}")
    print(f"  sum P(x)   classical {stats['sum_classical']:.12f}   quantum {stats['sum_quantum']:.12f}")
    print(f"  MoI        classical {stats['moi_classical']:12.3f}   quantum {stats['moi_quantum']:12.3f}")
    print(f"  MoI / N    classical {stats['moi_classical'] / args.steps:12.6f}  (exactly 1 for the unbiased walk)")
    print(f"  MoI / N^2  quantum   {stats['moi_quantum'] / args.steps**2:12.6f}")

    fig_spread, scaling = figure_spreading(args.max_steps, args.theta)
    print(f"\nMoI ~ N^alpha over N in [{scaling['ns'][0]}, {scaling['ns'][-1]}]")
    print(f"  classical alpha = {scaling['slope_classical']:.4f}   (expect 1, diffusive)")
    print(f"  quantum   alpha = {scaling['slope_quantum']:.4f}   (expect 2, ballistic)")

    if args.save_prefix:
        print()
        print(f"saved {save_figure(fig_dist, f'{args.save_prefix}_distribution', overwrite=args.overwrite)}")
        print(f"saved {save_figure(fig_spread, f'{args.save_prefix}_spreading', overwrite=args.overwrite)}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
