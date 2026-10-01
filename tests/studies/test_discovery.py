"""The data mining steps: subspace greedy search finds what two synthetic clusters differ in."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("hdbscan")

from studies.regimes import discovery  # noqa: E402

NAMES = [f"v{j}" for j in range(7)]


def _data(n_ref=300, n_comp=40, seed=3):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n_ref, 7))
    B = rng.standard_normal((n_comp, 7))
    B[:, 2] += 4.0
    B[:, 5] += 2.0
    return np.vstack([A, B]), np.arange(n_ref), np.arange(n_ref, n_ref + n_comp)


def test_sgs_ranks_the_shifted_variables_first():
    X, ref, comp = _data()
    contri = discovery.sgs(X, ref, comp, NAMES)
    assert list(contri)[:2] == ["v2", "v5"]
    assert contri["v2"] > 50 and sum(contri.values()) == pytest.approx(100.0)


def test_sgs_extends_a_subspace_only_by_variables_not_already_in_it():
    """With repeats allowed, a variable that alone separates the clusters took every
    level of the search and all of the contribution; without them the search goes on."""
    rng = np.random.default_rng(3)
    A, B = rng.standard_normal((300, 7)), rng.standard_normal((40, 7))
    B[:, 2] += 6.0
    contri = discovery.sgs(np.vstack([A, B]), np.arange(300), np.arange(300, 340), NAMES)
    assert list(contri)[0] == "v2" and 0 < contri["v2"] < 100
    assert sum(v > 0 for v in contri.values()) > 1


def test_sgs_with_reduction_is_reproducible_by_seed():
    X, ref, comp = _data(n_comp=200)
    a = discovery.sgs(X, ref, comp, NAMES, seed=1)
    b = discovery.sgs(X, ref, comp, NAMES, seed=1)
    assert a == b and list(a)[0] == "v2"


def test_hdbscan_separates_two_blobs_and_reports_a_low_dbi():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 0.3, (200, 2)), rng.normal(5, 0.3, (200, 2))])
    labels, dbi = discovery.cluster_hdbscan(X, 10)
    assert len(set(labels) - {-1}) == 2 and dbi < 0.5


def test_pca_keeps_the_components_for_95_percent():
    rng = np.random.default_rng(0)
    X = np.c_[rng.normal(0, 10, 500), rng.normal(0, 1, 500), rng.normal(0, 0.01, 500)]
    assert discovery.pca_auto_components(X) == 1
