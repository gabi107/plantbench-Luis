"""Variance-based sensitivity analysis: which variables an index responds to.

A design criterion is a scalar read from a plant, and a fair question to ask of it is how
much of its variation over a design space each variable accounts for.  Sobol' indices
answer that by decomposing the variance of the output: the first-order index of a variable
is the share of the variance explained by that variable alone, and its total index adds
every interaction the variable enters, so the two coincide when the model is additive and
separate when it is not.

The estimators are the ones of Saltelli et al. (2010) for the first-order index and Jansen
(1999) for the total, evaluated on the Saltelli construction of two independent samples and
the k matrices that swap one column between them.  The cost is `n * (k + 2)` evaluations of
the output for k variables.

`f` takes the whole sample at once, as an (m, k) array, and returns m outputs.  That leaves
the caller free to evaluate the points in parallel, which matters when one evaluation is a
design solve.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.stats import qmc


@dataclass
class Indices:
    """Sobol' indices of one output over the variables of a design space."""

    names: list[str]
    first: np.ndarray  # first-order: the variance this variable explains alone
    total: np.ndarray  # total: including every interaction it enters
    variance: float  # of the output over the sample
    n_evaluations: int

    def ranked(self) -> list[tuple[str, float, float]]:
        """(name, first, total) in decreasing total index, the order a reader wants."""
        order = np.argsort(-self.total)
        return [(self.names[i], float(self.first[i]), float(self.total[i])) for i in order]


def _matrices(bounds, n: int, seed: int):
    """The Saltelli construction: two samples, and one per variable that swaps a column."""
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    if not np.all(lo < hi):
        raise ValueError("every bound needs low < high")
    k = len(bounds)
    # One 2k-dimensional low-discrepancy sample split in half keeps A and B independent,
    # which is what the estimators assume.
    u = qmc.Sobol(d=2 * k, scramble=True, seed=seed).random(n)
    A, B = qmc.scale(u[:, :k], lo, hi), qmc.scale(u[:, k:], lo, hi)
    AB = []
    for i in range(k):
        M = A.copy()
        M[:, i] = B[:, i]
        AB.append(M)
    return A, B, AB


def base_size(n: int) -> int:
    """`n` rounded up to a power of two, which is what the Sobol' sequence is balanced for."""
    return 1 << max(0, int(np.ceil(np.log2(max(n, 1)))))


def sample_points(bounds: Sequence[Sequence[float]], n: int = 256,
                  seed: int = 0) -> np.ndarray:
    """The `base_size(n) * (k + 2)` points the analysis evaluates, in the order it reads
    them back: the two samples first, then the swapped matrix of each variable in turn.

    Exposed so that an output too expensive to evaluate in one process -- a closed-loop
    run rather than a design solve -- can be generated as a dataset and read back with
    `indices_from`, which is the same analysis split in two.
    """
    A, B, AB = _matrices(bounds, base_size(n), seed)
    return np.vstack([A, B, *AB])


def indices_from(y: np.ndarray, n: int, k: int,
                 names: Sequence[str] | None = None) -> Indices:
    """The indices of outputs evaluated at `sample_points(bounds, n, seed)`, in that order."""
    n = base_size(n)
    if names is None:
        names = [f"x{i}" for i in range(k)]
    if len(names) != k:
        raise ValueError(f"{len(names)} names for {k} variables")
    y = np.asarray(y, dtype=float).ravel()
    if y.size != n * (k + 2):
        raise ValueError(f"{y.size} outputs for a sample of {n * (k + 2)} points")
    bad = int((~np.isfinite(y)).sum())
    if bad:
        raise ValueError(f"{bad} of {y.size} evaluations are not finite; the indices would "
                         "be meaningless. Bound the space to the region that solves.")

    yA, yB = y[:n], y[n:2 * n]
    yAB = [y[(2 + i) * n:(3 + i) * n] for i in range(k)]
    var = float(np.var(np.concatenate([yA, yB])))
    if var == 0.0:
        raise ValueError("the output does not vary over the sample; there is nothing to "
                         "attribute to any variable")
    first, total = _estimate(yA, yB, yAB, var)
    return Indices(names=list(names), first=first, total=total,
                   variance=var, n_evaluations=int(y.size))


def _estimate(yA, yB, yAB, var):
    first = np.array([float(np.mean(yB * (yi - yA)) / var) for yi in yAB])
    total = np.array([float(np.mean((yA - yi) ** 2) / (2 * var)) for yi in yAB])
    return first, total


def bootstrap_intervals(y: np.ndarray, n: int, k: int, resamples: int = 1000,
                        level: float = 0.95, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Percentile bootstrap intervals of the first-order and total indices.

    The rows of the base sample are resampled with replacement, the same rows from A, B
    and every swapped matrix, as Archer et al. (1997) do.  Returns two (k, 2) arrays of
    lower and upper bounds, for the first-order and the total indices.
    """
    n = base_size(n)
    y = np.asarray(y, dtype=float).ravel()
    if y.size != n * (k + 2):
        raise ValueError(f"{y.size} outputs for a sample of {n * (k + 2)} points")
    blocks = y.reshape(k + 2, n)
    rng = np.random.default_rng(seed)
    first, total = np.empty((resamples, k)), np.empty((resamples, k))
    for b in range(resamples):
        rows = rng.integers(0, n, n)
        yA, yB, *yAB = blocks[:, rows]
        var = float(np.var(np.concatenate([yA, yB])))
        first[b], total[b] = _estimate(yA, yB, yAB, var) if var > 0 else (np.nan, np.nan)
    q = [(1 - level) / 2 * 100, (1 + level) / 2 * 100]
    return (np.nanpercentile(first, q, axis=0).T, np.nanpercentile(total, q, axis=0).T)


def sobol_indices(f: Callable[[np.ndarray], np.ndarray], bounds: Sequence[Sequence[float]],
                  n: int = 256, seed: int = 0, names: Sequence[str] | None = None) -> Indices:
    """First-order and total Sobol' indices of `f` over a box.

    `f` takes the whole sample at once and the output costs `base_size(n) * (k + 2)`
    evaluations of it.
    """
    X = sample_points(bounds, n, seed)
    y = np.asarray(f(X), dtype=float).ravel()
    if y.size != X.shape[0]:
        raise ValueError(f"f returned {y.size} values for {X.shape[0]} points")
    return indices_from(y, n, len(bounds), names)
