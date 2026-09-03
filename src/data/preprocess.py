"""Normalisation of probability distributions before they reach a classifier.

Paper A, Sec. III B 4 (CLAUDE.md Sec. 6).
"""

from __future__ import annotations

import numpy as np


def normalize_pmax(x: np.ndarray) -> np.ndarray:
    r"""Rescale each sample so that its maximum is 1.

    Physical normalisation is :math:`\sum_x P(x)=1`; the *ML* normalisation is
    :math:`P_{max}=1`, dividing each distribution by its own maximum, matching
    the image-recognition convention (Paper A, Sec. III B 4).

    Paper A found this makes no difference to the SVM but substantially
    reduces variance for the MLP and CNN. It must be applied to **both**
    training and inference data.

    Parameters
    ----------
    x
        Shape ``(n_samples, n_sites)`` or a single ``(n_sites,)`` distribution.

    Returns
    -------
    ndarray
        Same shape, ``float64``. Rows that are identically zero are left
        alone rather than producing NaNs.
    """
    arr = np.atleast_2d(np.asarray(x, dtype=np.float64))
    peak = arr.max(axis=1, keepdims=True)
    scaled = np.divide(arr, peak, out=arr.copy(), where=peak > 0)
    return scaled.reshape(np.shape(x))
