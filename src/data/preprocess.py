"""P_max = 1 normalisation before a classifier sees a sample (Paper A, Sec. III B 4)."""

from __future__ import annotations

import numpy as np


def normalize_pmax(x: np.ndarray) -> np.ndarray:
    """Divide each distribution by its own maximum, the image-recognition convention.

    ``x`` is ``(n_samples, n_sites)`` or a single ``(n_sites,)`` distribution;
    the shape is preserved. All-zero rows are left alone rather than made NaN.
    Must be applied to both training and inference data.
    """
    arr = np.atleast_2d(np.asarray(x, dtype=np.float64))
    peak = arr.max(axis=1, keepdims=True)
    scaled = np.divide(arr, peak, out=arr.copy(), where=peak > 0)
    return scaled.reshape(np.shape(x))


#: "slice" -- each time slice by its own max; "global" -- the whole image by its
#: single max; "none" -- keep the physical sum_x P(x, t) = 1. None is a paper
#: value: Paper A normalises a single final-time distribution.
EVOLUTION_NORMALIZATIONS: tuple[str, ...] = ("slice", "global", "none")


def normalize_pmax_evolution(p: np.ndarray, mode: str = "slice") -> np.ndarray:
    """P_max = 1 for space-time samples whose last two axes are ``(n_times, n_sites)``.

    "slice" is the default because the peak of P(x, t) decays as the walker
    spreads, so a "global" maximum is set by the first few steps and crushes
    the late-time slices, which is where the signal is.
    """
    if mode not in EVOLUTION_NORMALIZATIONS:
        raise ValueError(
            f"unknown mode {mode!r}; expected one of {EVOLUTION_NORMALIZATIONS}"
        )
    arr = np.asarray(p, dtype=np.float64)
    if arr.ndim < 2:
        raise ValueError(
            f"expected at least 2 axes, the last two being (n_times, n_sites), "
            f"got shape {arr.shape}"
        )
    if mode == "none":
        return arr.copy()

    axes = (-1,) if mode == "slice" else (-2, -1)
    peak = arr.max(axis=axes, keepdims=True)
    return np.divide(arr, peak, out=arr.copy(), where=peak > 0)
