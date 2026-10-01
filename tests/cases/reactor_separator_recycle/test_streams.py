"""The stream table conserves energy around the whole plant."""

import pytest

from plantbench.cases.reactor_separator_recycle import plant, streams
from plantbench.cases.reactor_separator_recycle.parameters import PlantParameters

I, A, B = 0, 1, 2


@pytest.fixture(scope="module")
def pp():
    return PlantParameters()


@pytest.fixture(scope="module")
def op(pp):
    return streams.design_point(plant.solve_design(pp, V_boil=28.0))


@pytest.mark.parametrize("P", [0.15, 0.4, 0.8])
def test_plant_energy_balance_closes(pp, op, P):
    """Reaction heat plus heating duties equals cooling duties plus the sensible heat the
    products carry out above the fresh-feed temperature.  Without the column's sensible
    heat term this fails by 6-9 GJ/h, growing with pressure."""
    table = streams.stream_table(op, P, pp)
    heating = sum(s.duty for s in table if not s.is_hot)
    cooling = sum(s.duty for s in table if s.is_hot)

    reacted = op.F_fresh * (1.0 - pp.z_inert) - op.purge * op.xD[A] - op.Bm * op.xB[A]
    reaction = pp.reactor.dH * reacted

    t = streams.temperatures(op, P, pp)
    T0, Tst = pp.thermo.T_storage, pp.thermo.T_product_storage
    carried_out = pp.cp_molar * (op.Bm * (min(t.T_bottoms, Tst) - T0)
                                 + op.purge * (min(t.T_drum, Tst) - T0))

    assert reaction + heating == pytest.approx(cooling + carried_out, rel=1e-6)


def test_sensible_heat_is_positive_and_grows_with_pressure(pp, op):
    q = [streams.column_sensible_heat(op, streams.temperatures(op, P, pp), pp)
         for P in (0.15, 0.4, 0.8)]
    assert 0 < q[0] < q[1] < q[2]
