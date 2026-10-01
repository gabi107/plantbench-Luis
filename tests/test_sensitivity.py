"""Sobol' indices: the variance of an output attributed to the variables of a design space."""

from __future__ import annotations

import numpy as np
import pytest

from plantbench.sensitivity import (
    base_size,
    bootstrap_intervals,
    indices_from,
    sample_points,
    sobol_indices,
)

A, B = 7.0, 0.1
# The Ishigami function, whose indices are known in closed form, is the standard check.
ISHIGAMI_FIRST = (0.3139, 0.4424, 0.0)
ISHIGAMI_TOTAL = (0.5576, 0.4424, 0.2437)
ISHIGAMI_BOUNDS = [(-np.pi, np.pi)] * 3


def ishigami(X):
    return np.sin(X[:, 0]) + A * np.sin(X[:, 1]) ** 2 + B * X[:, 2] ** 4 * np.sin(X[:, 0])


def test_indices_match_the_closed_form():
    r = sobol_indices(ishigami, ISHIGAMI_BOUNDS, n=8192, names=["x1", "x2", "x3"])
    assert r.first == pytest.approx(ISHIGAMI_FIRST, abs=0.01)
    assert r.total == pytest.approx(ISHIGAMI_TOTAL, abs=0.01)


def test_a_variable_the_output_ignores_gets_no_total_index():
    """x3 enters Ishigami only through an interaction, and a fourth variable not at all."""
    r = sobol_indices(lambda X: X[:, 0] + 2 * X[:, 1], [(0.0, 1.0)] * 3, n=1024)
    assert r.total[2] == pytest.approx(0.0, abs=0.01)


def test_an_additive_output_has_first_equal_to_total():
    r = sobol_indices(lambda X: X[:, 0] + 2 * X[:, 1], [(0.0, 1.0)] * 2, n=1024)
    assert r.first == pytest.approx(r.total, abs=0.01)
    assert sum(r.first) == pytest.approx(1.0, abs=0.01)


def test_the_cost_is_n_times_k_plus_two_and_n_is_a_power_of_two():
    r = sobol_indices(ishigami, ISHIGAMI_BOUNDS, n=100)  # rounded up to 128
    assert r.n_evaluations == 128 * (3 + 2)


def test_the_same_seed_gives_the_same_indices():
    a = sobol_indices(ishigami, ISHIGAMI_BOUNDS, n=512, seed=3)
    b = sobol_indices(ishigami, ISHIGAMI_BOUNDS, n=512, seed=3)
    assert np.array_equal(a.first, b.first) and np.array_equal(a.total, b.total)


def test_ranked_orders_by_total_index():
    r = sobol_indices(ishigami, ISHIGAMI_BOUNDS, n=2048, names=["x1", "x2", "x3"])
    assert [name for name, _, _ in r.ranked()] == ["x1", "x2", "x3"]


def test_a_design_that_does_not_solve_is_refused_rather_than_averaged():
    def sometimes_fails(X):
        y = X[:, 0].copy()
        y[0] = np.nan
        return y

    with pytest.raises(ValueError, match="not finite"):
        sobol_indices(sometimes_fails, [(0.0, 1.0)], n=64)


def test_an_output_that_does_not_vary_is_refused():
    with pytest.raises(ValueError, match="does not vary"):
        sobol_indices(lambda X: np.ones(len(X)), [(0.0, 1.0)] * 2, n=64)


@pytest.mark.parametrize("bad", [[(1.0, 0.0)], [(0.0, 0.0)]])
def test_bounds_need_low_below_high(bad):
    with pytest.raises(ValueError, match="low < high"):
        sobol_indices(lambda X: X[:, 0], bad, n=64)


def test_names_must_match_the_variables():
    with pytest.raises(ValueError, match="names for"):
        sobol_indices(lambda X: X[:, 0], [(0.0, 1.0)] * 2, n=64, names=["only one"])


# -- the sample and the estimate, split ------------------------------------------------


def test_sample_points_has_the_shape_the_cost_formula_predicts():
    X = sample_points(ISHIGAMI_BOUNDS, n=64, seed=0)
    assert X.shape == (64 * (3 + 2), 3)


def test_a_sample_run_elsewhere_gives_the_same_indices():
    """The split analysis is the whole one: evaluating `sample_points` and handing the
    outputs to `indices_from` must reproduce `sobol_indices` exactly."""
    whole = sobol_indices(ishigami, ISHIGAMI_BOUNDS, n=256, seed=3, names=["x1", "x2", "x3"])
    X = sample_points(ISHIGAMI_BOUNDS, n=256, seed=3)
    split = indices_from(ishigami(X), n=256, k=3, names=["x1", "x2", "x3"])
    assert np.allclose(split.first, whole.first)
    assert np.allclose(split.total, whole.total)


def test_n_is_rounded_to_a_power_of_two_the_same_way_on_both_sides():
    assert base_size(100) == 128
    assert sample_points([(0.0, 1.0)] * 2, n=100).shape[0] == 128 * 4
    indices_from(np.arange(128 * 4, dtype=float), n=100, k=2)


def test_outputs_in_the_wrong_number_are_refused():
    with pytest.raises(ValueError, match="outputs for a sample"):
        indices_from(np.arange(10.0), n=64, k=3)


def test_bootstrap_intervals_hold_the_closed_form_and_narrow_with_n():
    widths = []
    for n in (256, 4096):
        y = ishigami(sample_points(ISHIGAMI_BOUNDS, n=n))
        first, total = bootstrap_intervals(y, n=n, k=3, resamples=400)
        if n == 4096:
            assert np.all((first[:, 0] - 0.02 <= ISHIGAMI_FIRST) & (ISHIGAMI_FIRST <= first[:, 1] + 0.02))
            assert np.all((total[:, 0] - 0.02 <= ISHIGAMI_TOTAL) & (ISHIGAMI_TOTAL <= total[:, 1] + 0.02))
        widths.append(np.mean(total[:, 1] - total[:, 0]))
    assert widths[1] < widths[0] / 2
