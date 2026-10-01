"""The temperature model must reproduce the volatilities the column already uses."""

import numpy as np
import pytest

from plantbench.cases.reactor_separator_recycle import plant
from plantbench.cases.reactor_separator_recycle.parameters import PlantParameters
from plantbench.units import thermo


@pytest.fixture(scope="module")
def pp():
    return PlantParameters()


@pytest.fixture(scope="module")
def design(pp):
    return plant.solve_design(pp, V_boil=28.0)


def test_volatilities_are_exact_at_any_temperature(pp):
    tp, cp = pp.thermo, pp.column
    for T in (280.0, 330.0, 400.0, 480.0):
        Ps = thermo.psat(T, tp, cp)
        assert Ps / Ps[2] == pytest.approx(np.array(cp.alpha), rel=1e-12)


def test_normal_boiling_points(pp):
    Tb = thermo.normal_boiling_points(pp.thermo, pp.column)
    assert Tb[2] == pytest.approx(400.0, abs=1e-9)
    assert Tb[1] == pytest.approx(359.0, abs=0.05)
    assert Tb[0] == pytest.approx(337.2, abs=0.05)


def test_bubble_and_dew_points_are_consistent(pp):
    tp, cp = pp.thermo, pp.column
    x = np.array([0.2, 0.5, 0.3])
    for P in (0.15, 1.0, 3.0):
        Tb = thermo.bubble_point(x, P, tp, cp)
        assert thermo.psat(Tb, tp, cp) @ x == pytest.approx(P, rel=1e-12)
        y = thermo.psat(Tb, tp, cp) * x / P  # vapor in equilibrium with x
        assert y.sum() == pytest.approx(1.0, rel=1e-12)
        # the dew point of that vapor is the same temperature
        assert thermo.dew_point(y, P, tp, cp) == pytest.approx(Tb, abs=1e-9)
        assert thermo.dew_point(x, P, tp, cp) >= Tb


def test_temperature_levels_reproduce_exploration(pp, design):
    """The 16 September scan that the design ladder was chosen from."""
    t = thermo.column_temperatures(design.x[6:], 0.25, pp.thermo, pp.column)
    assert t["bottoms"] == pytest.approx(349.2, abs=0.05)
    assert t["drum"] == pytest.approx(313.4, abs=0.05)
    t = thermo.column_temperatures(design.x[6:], 1.0, pp.thermo, pp.column)
    assert t["bottoms"] == pytest.approx(399.5, abs=0.05)
    assert t["drum"] == pytest.approx(353.3, abs=0.05)


def test_tray_temperatures_rise_down_the_column(pp, design):
    t = thermo.column_temperatures(design.x[6:], 0.5, pp.thermo, pp.column)
    assert np.all(np.diff(t["trays"]) >= -1e-9)
    assert t["drum"] <= t["trays"][0] + 1e-9
    assert t["trays"][-1] <= t["bottoms"] + 1e-9


def test_reactor_must_be_pressurized(pp, design):
    """At 362.4 K the reactor liquid would boil below 0.87 bar."""
    xr = design.x[1:4] / pp.rho_molar
    P = thermo.bubble_pressure(xr, design.x[4], pp.thermo, pp.column)
    assert P == pytest.approx(0.87, abs=0.01)
