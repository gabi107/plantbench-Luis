"""Operating points and the stream table they imply at a given column pressure.

Heat integration here rests on a separation that holds for this plant: under a control
structure that holds its set-points, the material steady state does not depend on where
the heat comes from.  Flows, compositions and the reactor temperature are fixed by the
material balances and the loops; the heat-exchanger network decides only how the
resulting duties are met.  So an operating point is computed once, from the dynamic
model, and every design and column pressure is then evaluated on it algebraically.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from plantbench.heat.streams import Stream
from plantbench.units import column as col
from plantbench.units import thermo

from .parameters import PlantParameters
from .plant import N_REACTOR_STATES, Design, Inputs

I, A, B = 0, 1, 2

# Isothermal duties (condensing, boiling, reaction heat) enter the problem table as
# streams spanning this small temperature interval.
ISOTHERMAL_SPAN = 0.1  # K


@dataclass(frozen=True)
class OperatingPoint:
    """A settled material steady state of the plant, independent of the heat network."""

    label: str
    F_fresh: float  # kmol/min
    recycle: float  # kmol/min
    purge: float  # kmol/min
    D: float  # kmol/min
    Bm: float  # kmol/min
    V_boil: float  # kmol/min
    F_col: float  # kmol/min, reactor effluent = column feed
    T_reactor: float  # K
    Q_jacket: float  # kJ/min removed by the jacket
    z_column: np.ndarray  # column state, for temperatures
    zF: np.ndarray  # column feed composition
    xD: np.ndarray  # distillate composition
    xB: np.ndarray  # bottoms composition


def operating_point(label: str, x: np.ndarray, u: Inputs, pp: PlantParameters) -> OperatingPoint:
    """Build an operating point from a plant state and the inputs acting on it."""
    rp = pp.reactor
    n_col = col.n_states(pp.column)
    z = np.asarray(x[N_REACTOR_STATES:N_REACTOR_STATES + n_col], dtype=float)
    _, xD, _, _, _, xB = col.unpack(z, pp.column)
    T, Tj = float(x[4]), float(x[5])
    return OperatingPoint(
        label=label,
        F_fresh=float(u.F_fresh),
        recycle=float(u.D - u.P),
        purge=float(u.P),
        D=float(u.D),
        Bm=float(u.Bm),
        V_boil=float(u.V_boil),
        F_col=float(u.F_out * pp.rho_molar),
        T_reactor=T,
        Q_jacket=float(rp.U * rp.A_ht * (T - Tj)),
        z_column=z,
        zF=np.asarray(x[1:4], dtype=float) / pp.rho_molar,
        xD=np.array(xD, dtype=float),
        xB=np.array(xB, dtype=float),
    )


def design_point(design: Design) -> OperatingPoint:
    return operating_point("design", design.x, design.u, design.pp)


def _sensible(name, Ts, Tt, CP) -> Stream | None:
    if abs(Ts - Tt) < 1e-9 or CP <= 0:
        return None
    return Stream(name, Ts, Tt, CP)


def _isothermal(name, T, duty, hot: bool) -> Stream | None:
    if duty <= 0:
        return None
    span = ISOTHERMAL_SPAN
    Ts, Tt = (T + span / 2, T - span / 2) if hot else (T - span / 2, T + span / 2)
    return Stream(name, Ts, Tt, duty / span)


@dataclass(frozen=True)
class Temperatures:
    """The temperature levels an operating point takes on at column pressure P."""

    P: float
    T_reactor: float
    T_reactor_inlet: float
    T_mix: float  # fresh feed and recycle, before any preheat
    T_drum: float
    T_condenser_dew: float
    T_bottoms: float
    T_column_feed: float  # bubble point of the reactor effluent at column pressure


def temperatures(op: OperatingPoint, P: float, pp: PlantParameters) -> Temperatures:
    tp, cp = pp.thermo, pp.column
    t = thermo.column_temperatures(op.z_column, P, tp, cp)
    F_in = op.F_fresh + op.recycle
    T_mix = (op.F_fresh * tp.T_storage + op.recycle * t["drum"]) / F_in
    return Temperatures(
        P=P,
        T_reactor=op.T_reactor,
        T_reactor_inlet=pp.reactor.T0,
        T_mix=T_mix,
        T_drum=t["drum"],
        T_condenser_dew=t["condenser_dew"],
        T_bottoms=t["bottoms"],
        T_column_feed=float(thermo.bubble_point(op.zF, P, tp, cp)),
    )


def column_sensible_heat(op: OperatingPoint, t: Temperatures, pp: PlantParameters) -> float:
    """kJ/min the reboiler supplies beyond lambda*V, which closes the column energy balance.

    Constant molar overflow carries the same vapor through every stage, which on its own
    makes the reboiler and condenser duties equal.  But the feed enters at its bubble
    point and the products leave at theirs, the bottoms hotter and the distillate colder,
    so Q_R - Q_C = D h_D + B h_B - F h_F.  With the condenser at lambda*V, the difference
    is the reboiler's.
    """
    return pp.cp_molar * (op.D * (t.T_drum - t.T_column_feed)
                          + op.Bm * (t.T_bottoms - t.T_column_feed))


def stream_table(op: OperatingPoint, P: float, pp: PlantParameters) -> list[Stream]:
    """Every heating and cooling duty the plant needs at column pressure P.

    Hot and cold is decided by the temperatures, not assumed: at high column pressure the
    reactor effluent must be heated to its bubble point rather than cooled, and the
    recycle can arrive hotter than the reactor inlet it is mixed into.
    """
    tp, cpm, lam = pp.thermo, pp.cp_molar, pp.column.lambda_vap
    t = temperatures(op, P, pp)
    F_in = op.F_fresh + op.recycle
    Tst = tp.T_product_storage

    candidates = [
        _sensible("reactor feed", t.T_mix, t.T_reactor_inlet, F_in * cpm),
        _sensible("reactor effluent", t.T_reactor, t.T_column_feed, op.F_col * cpm),
        _isothermal("reactor heat", t.T_reactor, op.Q_jacket, hot=True),
        # A total condenser takes the overhead vapor from its dew point to the bubble
        # point of the same composition; the latent heat is spread over that interval.
        # The interval is kept at least ISOTHERMAL_SPAN wide so the stream stays a hot
        # stream even when a transient leaves the distillate nearly a single component.
        Stream("condenser", max(t.T_condenser_dew, t.T_drum + ISOTHERMAL_SPAN), t.T_drum,
               lam * op.V_boil / max(t.T_condenser_dew - t.T_drum, ISOTHERMAL_SPAN)),
        _isothermal("reboiler", t.T_bottoms, lam * op.V_boil + column_sensible_heat(op, t, pp),
                    hot=False),
        # Products go to storage no hotter than Tst; a colder stream is left as it is.
        _sensible("bottoms product", t.T_bottoms, min(t.T_bottoms, Tst), op.Bm * cpm),
        _sensible("purge", t.T_drum, min(t.T_drum, Tst), op.purge * cpm),
    ]
    return [s for s in candidates if s is not None]
