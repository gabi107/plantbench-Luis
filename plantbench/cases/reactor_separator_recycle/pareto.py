"""Operating points across throughput, the designs evaluated on them, and Pareto fronts.

The operating variable is throughput.  Each operating point is the settled state of the
plant under the reference control structure after a change in the reactor effluent flow,
which sets the production rate, computed once with the dynamic model.  Every design is
then evaluated on the same operating points: D1 and the networks at their own column pressures, and the pinch
target D2 at every pressure on a grid, keeping only the points no other point beats.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from plantbench.core import control as ctl

from . import economics as ec
from . import hen, streams
from . import scenarios as sc
from .plant import Design

DEFAULT_THROUGHPUT = (-0.20, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20, 0.25)
DEFAULT_PRESSURES = (0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5)


@dataclass
class SettledPoint:
    fraction: float
    op: streams.OperatingPoint
    drift: float  # largest state change over the last tenth of the run


def settle(design: Design, fraction: float, t_end: float = 5000.0) -> SettledPoint:
    """Settle the plant after a change in the effluent flow under the reference structure."""
    if fraction == 0.0:
        return SettledPoint(0.0, streams.design_point(design), 0.0)
    S = sc.plantwide(design)
    step = sc.throughput_step(fraction, sc.THROUGHPUT_HANDLE["effluent fixed + cascade"], 30.0)
    t, X, U = ctl.simulate(design, S, t_end=t_end, n_points=200, disturbance=step)
    n = len(t) // 10
    drift = float(np.max(np.abs(X[:, -1] - X[:, -1 - n])))
    # A disturbance returns the nominal inputs themselves while t precedes its step
    # (core.disturbances), so the settled values are written into a copy rather than
    # into whatever came back, which would otherwise be the cached design's own inputs.
    u = replace(step(t[-1], design.u),
                **{f: U[f][-1] for f in
                   ("F_fresh", "Fj", "F_out", "V_boil", "L_reflux", "D", "P", "Bm")})
    op = streams.operating_point(f"{fraction:+.0%}", X[:, -1], u, design.pp)
    return SettledPoint(fraction, op, drift)


@dataclass
class Result:
    design: str
    fraction: float
    pressure: float
    evaluation: ec.Evaluation


def evaluate_designs(points: list[SettledPoint], design: Design,
                     pressures=DEFAULT_PRESSURES) -> list[Result]:
    pp = design.pp
    design_op = streams.design_point(design)
    sized = {k: hen.design(net, design_op, pp) for k, net in hen.DESIGNS.items()}
    out = []
    for sp in points:
        for name, net in hen.DESIGNS.items():
            loads = hen.rate(sized[name], sp.op, pp).loads
            out.append(Result(name, sp.fraction, net.pressure, ec.evaluate(sp.op, loads, pp)))
        for P in pressures:
            try:
                loads = ec.target_loads(streams.stream_table(sp.op, P, pp), pp)
            except ValueError:
                continue
            out.append(Result("D2", sp.fraction, P, ec.evaluate(sp.op, loads, pp)))
    return out


def nondominated(profit: np.ndarray, gwp: np.ndarray) -> np.ndarray:
    """Mask of points no other point beats on both maximizing profit and minimizing GWP."""
    profit, gwp = np.asarray(profit, float), np.asarray(gwp, float)
    keep = np.ones(profit.size, dtype=bool)
    for i in range(profit.size):
        better_or_equal = (profit >= profit[i]) & (gwp <= gwp[i])
        strictly = (profit > profit[i]) | (gwp < gwp[i])
        if np.any(better_or_equal & strictly):
            keep[i] = False
    return keep


def normalize(value: np.ndarray, best: float, worst: float) -> np.ndarray:
    """Map to [0, 1] with 0 the best achievable and 1 the worst, as in the design comparison."""
    span = worst - best
    return np.zeros_like(value) if span == 0 else (np.asarray(value) - best) / span
