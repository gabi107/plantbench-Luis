"""Heat-exchanger networks: balances, approaches, and the pinch target as a bound."""

import pytest

from plantbench.cases.reactor_separator_recycle import economics as ec
from plantbench.cases.reactor_separator_recycle import hen, plant, streams
from plantbench.cases.reactor_separator_recycle.parameters import PlantParameters


@pytest.fixture(scope="module")
def pp():
    return PlantParameters()


@pytest.fixture(scope="module")
def op(pp):
    return streams.design_point(plant.solve_design(pp, V_boil=28.0))


@pytest.mark.parametrize("name", ["D3", "D4", "D5"])
def test_exchangers_respect_approach_and_balance(pp, op, name):
    net = hen.DESIGNS[name]
    table = {s.name: s for s in streams.stream_table(op, net.pressure, pp)}
    r = hen.design(net, op, pp)
    assert r.exchangers, "every integrated design recovers some heat"
    for x in r.exchangers:
        h, c = table[x.match.hot], table[x.match.cold]
        assert x.duty == pytest.approx(h.CP * (x.T_hot_in - x.T_hot_out), rel=1e-9)
        assert x.duty == pytest.approx(c.CP * (x.T_cold_out - x.T_cold_in), rel=1e-9)
        assert x.T_hot_in - x.T_cold_out >= pp.thermo.dT_min - 1e-6
        assert x.T_hot_out - x.T_cold_in >= pp.thermo.dT_min - 1e-6


@pytest.mark.parametrize("name", ["D1", "D3", "D4", "D5"])
def test_process_duties_are_conserved(pp, op, name):
    """Recovered heat plus residual duty equals each stream's full duty."""
    net = hen.DESIGNS[name]
    table = streams.stream_table(op, net.pressure, pp)
    r = hen.design(net, op, pp)
    residual = {s.name: s.duty for s in r.residual}
    for s in table:
        taken = sum(x.duty for x in r.exchangers if s.name in (x.match.hot, x.match.cold))
        assert taken + residual.get(s.name, 0.0) == pytest.approx(s.duty, rel=1e-9)


@pytest.mark.parametrize("name", ["D1", "D3", "D4", "D5"])
def test_no_network_beats_the_pinch_target(pp, op, name):
    net = hen.DESIGNS[name]
    table = streams.stream_table(op, net.pressure, pp)
    target = ec.target_loads(table, pp)
    loads = hen.design(net, op, pp).loads
    hot = lambda L: sum(v for k, v in L.items() if "steam" in k)
    cold = lambda L: sum(v for k, v in L.items() if "water" in k)
    assert hot(loads) >= hot(target) * (1 - 1e-9) - 1e-6
    assert cold(loads) >= cold(target) * (1 - 1e-9) - 1e-6


def test_single_exchanger_designs_reach_the_target(pp, op):
    for name in ("D3", "D5"):
        net = hen.DESIGNS[name]
        target = ec.evaluate(op, ec.target_loads(streams.stream_table(op, net.pressure, pp), pp), pp)
        got = ec.evaluate(op, hen.design(net, op, pp).loads, pp)
        assert got.gwp_energy == pytest.approx(target.gwp_energy, rel=0.01)


def test_rating_at_design_reproduces_design_duties(pp, op):
    for name in ("D3", "D4", "D5"):
        sized = hen.design(hen.DESIGNS[name], op, pp)
        rated = hen.rate(sized, op, pp)
        assert [x.duty for x in rated.exchangers] == pytest.approx(
            [x.duty for x in sized.exchangers], rel=1e-6)


def test_pump_around_leaves_the_jacket_its_share(pp, op):
    r = hen.design(hen.DESIGNS["D4"], op, pp)
    pa = next(x for x in r.exchangers if x.match.hot == "reactor heat")
    assert pa.duty <= hen.PUMP_AROUND_SHARE * op.Q_jacket * (1 + 1e-9)
