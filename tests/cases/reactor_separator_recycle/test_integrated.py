"""Stage 2 plants: rest state, consistency with the steady-state network, conservation."""

import numpy as np
import pytest

from plantbench.cases.reactor_separator_recycle import hen, integrated, plant, streams
from plantbench.cases.reactor_separator_recycle.parameters import PlantParameters
from plantbench.core import control as ctl

DESIGNS = ("D1", "D3", "D4", "D5")


@pytest.fixture(scope="module")
def design():
    return plant.solve_design(PlantParameters(), V_boil=28.0)


@pytest.mark.parametrize("name", DESIGNS)
def test_stage2_plant_starts_at_rest(design, name):
    integ = integrated.build(design, name)
    S = integrated.stage2(integ)
    y0 = np.concatenate([integ.x, np.zeros(len(S))])
    r = ctl.closed_loop_rhs(0.0, y0, S, integ.u, integ.pp, None, integ.rhs)
    assert np.max(np.abs(r)) < 1e-8


@pytest.mark.parametrize("name", DESIGNS)
def test_material_state_is_the_base_design(design, name):
    """Heat integration does not move the material steady state."""
    integ = integrated.build(design, name)
    same = np.ones(len(design.x), dtype=bool)
    same[5] = False  # the jacket temperature moves when a pump-around shares the duty
    assert integ.x[:len(design.x)][same] == pytest.approx(design.x[same], abs=1e-12)
    assert integ.T_inlet(integ.x) == pytest.approx(design.pp.reactor.T0)


def test_rated_duties_match_the_sized_network(design):
    for name in ("D3", "D4", "D5"):
        integ = integrated.build(design, name)
        sized = hen.design(hen.DESIGNS[name], streams.design_point(design), design.pp)
        assert integ.duties(integ.x) == pytest.approx([e.duty for e in sized.exchangers])


def test_pump_around_and_jacket_remove_the_reactor_heat(design):
    integ = integrated.build(design, "D4")
    rp = design.pp.reactor
    jacket = rp.U * rp.A_ht * (integ.x[4] - integ.x[5])
    Q_pa = integ.duties(integ.x)[integ.pump_around]
    assert jacket + Q_pa == pytest.approx(streams.design_point(design).Q_jacket, rel=1e-9)
    assert jacket == pytest.approx((1 - hen.PUMP_AROUND_SHARE) * (jacket + Q_pa), rel=1e-9)


def test_trim_holds_the_inlet_through_a_feed_temperature_upset(design):
    import dataclasses
    integ = integrated.build(design, "D3")
    S = integrated.stage2(integ)
    T0 = design.pp.thermo.T_storage
    upset = lambda t, u: u if t < 30 else dataclasses.replace(u, T_storage=T0 - 10.0)
    t, X, U = ctl.simulate(integ, S, t_end=300.0, n_points=150, disturbance=upset)
    assert X[integ.n_base, -1] == pytest.approx(design.pp.reactor.T0, abs=0.05)
