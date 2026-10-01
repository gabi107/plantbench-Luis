"""Pareto filtering, normalization, and the pinch target as a bound off design."""

import dataclasses

import numpy as np
import pytest

from plantbench.cases.reactor_separator_recycle import pareto, plant
from plantbench.cases.reactor_separator_recycle.parameters import PlantParameters


def test_nondominated_filter():
    profit = np.array([10.0, 9.0, 8.0, 10.0, 7.0])
    gwp = np.array([5.0, 4.0, 4.5, 6.0, 3.0])
    # (8, 4.5) is beaten by (9, 4.0); (10, 6.0) by (10, 5.0)
    assert list(pareto.nondominated(profit, gwp)) == [True, True, False, False, True]


def test_ties_are_all_kept():
    assert pareto.nondominated(np.array([1.0, 1.0]), np.array([2.0, 2.0])).all()


def test_normalize_puts_best_at_zero():
    v = pareto.normalize(np.array([3.0, 5.0, 4.0]), best=3.0, worst=5.0)
    assert list(v) == pytest.approx([0.0, 1.0, 0.5])
    # for a quantity to maximize, best is the larger value
    v = pareto.normalize(np.array([3.0, 5.0]), best=5.0, worst=3.0)
    assert list(v) == pytest.approx([1.0, 0.0])


@pytest.fixture(scope="module")
def results():
    design = plant.solve_design(PlantParameters(), V_boil=28.0)
    points = [pareto.settle(design, f) for f in (-0.10, 0.0, 0.10)]
    return points, pareto.evaluate_designs(points, design)


def test_throughput_points_settle(results):
    points, _ = results
    for sp in points:
        assert sp.drift < 0.05


def test_pinch_target_bounds_every_network_off_design(results):
    _, res = results
    for fraction in (-0.10, 0.0, 0.10):
        target = {r.pressure: r.evaluation for r in res
                  if r.design == "D2" and r.fraction == fraction}
        for r in res:
            if r.design in ("D2",) or r.fraction != fraction:
                continue
            bound = target[r.pressure]
            assert r.evaluation.gwp_energy >= bound.gwp_energy - 1e-9
            assert r.evaluation.utility_cost >= bound.utility_cost - 1e-9


def test_settle_leaves_the_design_inputs_alone():
    """A disturbance returns the nominal inputs themselves while t precedes its step, so
    a run ending before the step at 30 min is where writing the settled values into what
    the disturbance returned would change the design itself."""
    design = plant.solve_design(PlantParameters(), V_boil=28.0)
    before = dataclasses.asdict(design.u)
    pareto.settle(design, 0.10, t_end=20.0)
    after = dataclasses.asdict(design.u)
    for field, value in before.items():
        assert np.array_equal(value, after[field]), f"settle changed design.u.{field}"


def test_feed_burden_is_common_to_all_designs(results):
    _, res = results
    for fraction in (-0.10, 0.0, 0.10):
        burdens = {round(r.evaluation.gwp_feed, 9) for r in res if r.fraction == fraction}
        assert len(burdens) == 1
