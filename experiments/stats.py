"""Statistics: bootstrap confidence intervals and paired significance tests (numpy only)."""
from __future__ import annotations

from itertools import product
from typing import Sequence

import numpy as np


def bootstrap_ci(
    values: Sequence[float], n_resamples: int = 10000, level: float = 0.95, seed: int = 0
) -> tuple[float, float, float]:
    """Mean and percentile-bootstrap CI of the mean. Returns (mean, lo, hi)."""
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    if x.size == 1:
        return float(x[0]), float(x[0]), float(x[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_resamples, x.size))
    means = x[idx].mean(axis=1)
    a = (1.0 - level) / 2.0
    return float(x.mean()), float(np.quantile(means, a)), float(np.quantile(means, 1.0 - a))


def paired_permutation_test(a: Sequence[float], b: Sequence[float], n_mc: int = 20000, seed: int = 0) -> float:
    """Two-sided paired sign-flip permutation test on mean(a - b).

    Exact enumeration of all 2^n sign flips for n <= 16, Monte Carlo otherwise.
    """
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = d.size
    if n == 0:
        return float("nan")
    obs = abs(d.mean())
    if np.allclose(d, 0.0):
        return 1.0
    if n <= 16:
        signs = np.array(list(product((1.0, -1.0), repeat=n)))
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice((1.0, -1.0), size=(n_mc, n))
    stats = np.abs((signs * d).mean(axis=1))
    return float(np.mean(stats >= obs - 1e-12))


def holm_correction(pvalues: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (same order as the input)."""
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj.tolist()


def paired_comparison(
    a: Sequence[float], b: Sequence[float], n_resamples: int = 10000, seed: int = 0
) -> dict[str, float]:
    """Mean difference a-b with bootstrap CI, permutation p-value and win count."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    mean, lo, hi = bootstrap_ci(d, n_resamples=n_resamples, seed=seed)
    return {
        "mean_diff": mean,
        "ci_lo": lo,
        "ci_hi": hi,
        "p_value": paired_permutation_test(a, b, seed=seed),
        "n": int(d.size),
        "a_lower_count": int(np.sum(d < 0)),
    }
