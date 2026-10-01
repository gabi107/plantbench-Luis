"""Every quantity of the design comparison, computed in one place.

`studies/reactor_separator_recycle/designs.py` prints these to `studies/reactor_separator_recycle/results/designs.txt` and draws the
figures from them, so the write-up, the numbers and the figures come from one run.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np

from plantbench.core import control as ctl
from plantbench.heat import pinch
from plantbench.units import column as col

from . import economics as ec
from . import hen, integrated, pareto, plant, streams
from . import scenarios as sc
from .parameters import PlantParameters

PRESSURES = (0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0)
FINAL_DESIGN = "D3"
GRID_INTENSITIES = (0.1, 0.4, 0.8)  # kg CO2e/kWh
PUMP_AROUND_SHARES = (0.0, 0.3, 0.5, 0.7)
# Reactor temperature loop settings scanned for D4: gains -0.25 to -4, integral times 5 to 160 min.
T_LOOP_GRID = tuple((Kc, tau) for Kc in (-0.25, -0.5, -1.0, -2.0, -4.0)
                    for tau in (5.0, 10.0, 20.0, 40.0, 80.0, 160.0))
SAMPLE = 0.25  # min, fine enough to resolve D4's 6-min oscillation
LATE_WINDOW = 300.0  # min, the end of a run over which residual oscillation is measured


@dataclass
class DesignComparison:
    design: plant.Design
    op: streams.OperatingPoint
    temperatures: list = field(default_factory=list)
    targets: list = field(default_factory=list)
    ladder: dict = field(default_factory=dict)
    points: list = field(default_factory=list)
    pareto: list = field(default_factory=list)
    grid: dict = field(default_factory=dict)
    share: list = field(default_factory=list)
    tuning: list = field(default_factory=list)
    controllability: dict = field(default_factory=dict)
    validation: dict = field(default_factory=dict)


def _evaluate(op, loads, pp):
    return ec.evaluate(op, loads, pp)


def temperature_levels(op, pp, pressures=PRESSURES):
    rows = []
    for P in pressures:
        t = streams.temperatures(op, P, pp)
        rows.append(dict(P=P, T_bottoms=t.T_bottoms, T_drum=t.T_drum, T_dew=t.T_condenser_dew,
                         T_column_feed=t.T_column_feed, T_mix=t.T_mix,
                         T_reactor=t.T_reactor, T_inlet=t.T_reactor_inlet))
    return rows


def targets_against_pressure(op, pp, pressures=PRESSURES):
    """Utility use with no integration (D1) and at the pinch target (D2)."""
    rows = []
    for P in pressures:
        table = streams.stream_table(op, P, pp)
        pt = pinch.problem_table(table, pp.thermo.dT_min)
        direct = _evaluate(op, ec.direct_loads(table, pp), pp)
        target = _evaluate(op, ec.target_loads(table, pp), pp)
        rows.append(dict(P=P, pinch_hot=pt.pinch_hot, QH_min=pt.QH_min, QC_min=pt.QC_min,
                         direct=direct, target=target, streams=table))
    return rows


def design_ladder(op, pp):
    out = {}
    for name, net in hen.DESIGNS.items():
        r = hen.design(net, op, pp)
        table = streams.stream_table(op, net.pressure, pp)
        out[name] = dict(network=net, result=r, evaluation=_evaluate(op, r.loads, pp),
                         target=_evaluate(op, ec.target_loads(table, pp), pp),
                         direct=_evaluate(op, ec.direct_loads(table, pp), pp))
    return out


def grid_sensitivity(design, points):
    """The Pareto evaluation repeated at other grid carbon intensities."""
    out = {}
    for g in GRID_INTENSITIES:
        pp = dataclasses.replace(design.pp,
                                 economics=dataclasses.replace(design.pp.economics, grid_intensity=g))
        d = dataclasses.replace(design, pp=pp)
        out[g] = pareto.evaluate_designs(points, d)
    return out


def pump_around_trade(design, op):
    rows = []
    for share in PUMP_AROUND_SHARES:
        net = hen.vacuum_network(share)
        e = _evaluate(op, hen.design(net, op, design.pp).loads, design.pp)
        integ = integrated.build(design, net)
        ev = ctl.closed_loop_spectrum(integ, integrated.stage2(integ))
        rows.append(dict(share=share, evaluation=e, Fj=integ.u.Fj,
                         damping=ctl.damping_ratio(ev), rightmost=float(ev.real.max())))
    return rows


def temperature_loop_tuning(design):
    """Damping of the least damped mode over a grid of reactor temperature tunings, base plant and D4."""
    integ = integrated.build(design, "D4")
    rows = []
    for Kc, tau in T_LOOP_GRID:
        retune = lambda S: [dataclasses.replace(lp, Kc=Kc, tau_I=tau) if lp.name == "reactor T" else lp
                            for lp in S]
        ev0 = ctl.closed_loop_spectrum(design, retune(sc.plantwide(design)))
        ev4 = ctl.closed_loop_spectrum(integ, retune(integrated.stage2(integ)))
        rows.append(dict(Kc=Kc, tau_I=tau,
                         base_damping=ctl.damping_ratio(ev0), base_rightmost=float(ev0.real.max()),
                         D4_damping=ctl.damping_ratio(ev4), D4_rightmost=float(ev4.real.max())))
    return rows


def _iae(t, v):
    return float(np.trapezoid(np.abs(v - v[0]), t))


def _disturbances(design):
    T0 = design.pp.thermo.T_storage
    return {
        "throughput +10%": (True, None,
                            sc.throughput_step(0.10, sc.THROUGHPUT_HANDLE["effluent fixed + cascade"], 30.0)),
        "fresh feed -10 K": (True, None,
                             lambda t, u: u if t < 30 else dataclasses.replace(u, T_storage=T0 - 10.0)),
    }


def controllability(design, names=("D1", "D3", "D4", "D5"), t_end=1500.0):
    out = {}
    for name in names:
        integ = integrated.build(design, name)
        ev = ctl.closed_loop_spectrum(integ, integrated.stage2(integ))
        entry = dict(damping=ctl.damping_ratio(ev), rightmost=float(ev.real.max()),
                     Fj_design=integ.u.Fj, runs={})
        for label, (cascade, spstep, dist) in _disturbances(design).items():
            S = integrated.stage2(integ, cascade=cascade, T_setpoint_step=spstep)
            t, X, U = ctl.simulate(integ, S, t_end=t_end, n_points=int(t_end / SAMPLE) + 1, disturbance=dist)
            nb = integ.n_base
            xBA = np.array([col.unpack(X[6:, i], design.pp.column)[5][1] for i in range(X.shape[1])])
            late = t >= t_end - LATE_WINDOW
            entry["runs"][label] = dict(
                t=t, T=X[4], CA=X[2], T_in=X[nb], Fj=U["Fj"], xBA=xBA,
                IAE_CA=_iae(t, X[2]), IAE_T_in=_iae(t, X[nb]), IAE_xBA=_iae(t, xBA),
                Fj_at_limit=float(np.mean((U["Fj"] <= 0.05 + 1e-9) | (U["Fj"] >= 6.0 - 1e-9))),
                Fj_min=float(U["Fj"].min()),
                # Peak-to-peak swing over the end of the run: zero for a settled plant, and
                # the direct measure of a limit cycle, which IAE from the start misstates.
                late_swing_T=float(np.ptp(X[4][late])),
                late_swing_Fj=float(np.ptp(U["Fj"][late])))
        out[name] = entry
    return out


def validation(design, names=(FINAL_DESIGN, "D4"), t_end=3000.0):
    """+1 K on the reactor temperature set-point, composition primary in manual.

    Run long enough for the recycle composition to settle, which takes far longer than
    the reactor temperature."""
    out = {}
    for name in names:
        integ = integrated.build(design, name)
        S = integrated.stage2(integ, cascade=False, T_setpoint_step=(30.0, 1.0))
        t, X, U = ctl.simulate(integ, S, t_end=t_end, n_points=int(t_end / SAMPLE) + 1)
        nb = integ.n_base
        cols = [col.unpack(X[6:, i], design.pp.column) for i in range(X.shape[1])]
        xBA = np.array([c[5][1] for c in cols])
        xDB = np.array([c[1][2] for c in cols])  # product B in the distillate, and so the recycle
        duties = {f"{e.match.hot} -> {e.match.cold}": X[nb + 1 + k]
                  for k, e in enumerate(integ.sized.exchangers)}
        out[name] = dict(t=t, T=X[4], CA=X[2], T_in=X[nb], Fj=U["Fj"], trim=U["Q_feed_trim"],
                         V_boil=U["V_boil"], recycle=U["recycle"], production=U["Bm"], xBA=xBA,
                         xDB=xDB, duties=duties)
    return out


def compute(pp: PlantParameters | None = None, dynamics: bool = True) -> DesignComparison:
    pp = PlantParameters() if pp is None else pp
    design = plant.solve_design(pp, V_boil=28.0)
    op = streams.design_point(design)
    ex = DesignComparison(design=design, op=op)
    ex.temperatures = temperature_levels(op, pp)
    ex.targets = targets_against_pressure(op, pp)
    ex.ladder = design_ladder(op, pp)
    ex.points = [pareto.settle(design, f) for f in pareto.DEFAULT_THROUGHPUT]
    ex.pareto = pareto.evaluate_designs(ex.points, design)
    ex.grid = grid_sensitivity(design, ex.points)
    if dynamics:
        ex.share = pump_around_trade(design, op)
        ex.tuning = temperature_loop_tuning(design)
        ex.controllability = controllability(design)
        ex.validation = validation(design)
    return ex
