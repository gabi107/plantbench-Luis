"""Heat-exchanger networks: declared as data, sized at a design point, rated off design.

A network is an ordered list of matches between named process streams.  Order matters:
each exchanger sees what the exchangers before it on the same stream have left, so a
feed preheated first against a cool stream and then against a hot one is described by
listing the cool match first.  Whatever duty remains on a stream after its exchangers is
met by utilities, exactly as in the no-integration case.

Each exchanger is counter-current.  At the design point its duty is the largest the two
streams allow at the minimum approach, and that fixes its UA.  Away from the design
point the same UA is rated by the effectiveness-NTU method, and a bypass holds the duty
to what the process needs where the exchanger could do more.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from plantbench.heat.streams import Stream
from plantbench.units.parameters import ColumnParameters

from . import economics as ec
from .parameters import PlantParameters
from .streams import OperatingPoint, stream_table


@dataclass(frozen=True)
class Match:
    hot: str
    cold: str
    # Largest share of the hot stream's duty the match may take.  Used for the reactor
    # heat, where the jacket must keep part of the duty as the stabilizing loop.
    max_share_of_hot: float = 1.0


@dataclass(frozen=True)
class Network:
    name: str
    description: str
    pressure: float  # bar, column pressure the network is designed for
    matches: tuple[Match, ...] = ()


@dataclass
class Exchanger:
    match: Match
    duty: float  # kJ/min
    UA: float  # kJ/(min K)
    T_hot_in: float
    T_hot_out: float
    T_cold_in: float
    T_cold_out: float


@dataclass
class NetworkResult:
    network: Network
    exchangers: list[Exchanger]
    residual: list[Stream]
    loads: dict[str, float] = field(default_factory=dict)

    @property
    def recovered(self) -> float:
        return sum(e.duty for e in self.exchangers)


class _StreamState:
    """The unprocessed part of a stream, consumed from its supply end."""

    def __init__(self, s: Stream):
        self.s = s
        self.T = s.T_supply
        self.used = 0.0

    @property
    def remaining(self) -> float:
        return abs(self.s.T_target - self.T) * self.s.CP

    def take(self, Q: float) -> float:
        dT = Q / self.s.CP
        self.T = self.T - dT if self.s.is_hot else self.T + dT
        self.used += Q
        return self.T

    def residual(self) -> Stream | None:
        if self.remaining <= 1e-6 * max(1.0, self.s.duty):
            return None
        return Stream(self.s.name, self.T, self.s.T_target, self.s.CP)


def _lmtd(dT1: float, dT2: float) -> float:
    if abs(dT1 - dT2) < 1e-9:
        return dT1
    return (dT1 - dT2) / np.log(dT1 / dT2)


def _effectiveness(UA: float, C_hot: float, C_cold: float) -> float:
    Cmin, Cmax = min(C_hot, C_cold), max(C_hot, C_cold)
    NTU, Cr = UA / Cmin, Cmin / Cmax
    if Cr < 1e-6:
        return 1.0 - np.exp(-NTU)
    if abs(1.0 - Cr) < 1e-9:
        return NTU / (1.0 + NTU)
    e = np.exp(-NTU * (1.0 - Cr))
    return (1.0 - e) / (1.0 - Cr * e)


def _states(streams: list[Stream], network: Network) -> dict[str, _StreamState]:
    """Stream states by name, refusing a network whose matches contradict the streams."""
    by_name = {s.name: _StreamState(s) for s in streams}
    for m in network.matches:
        for name, want_hot in ((m.hot, True), (m.cold, False)):
            st = by_name.get(name)
            if st is not None and st.s.is_hot != want_hot:
                raise ValueError(f"{network.name}: {name} is not a "
                                 f"{'hot' if want_hot else 'cold'} stream here")
    return by_name


def design(network: Network, op: OperatingPoint, pp: PlantParameters) -> NetworkResult:
    """Size every exchanger at the design operating point and the network's pressure."""
    dT = pp.thermo.dT_min
    streams = stream_table(op, network.pressure, pp)
    st = _states(streams, network)
    exchangers = []
    for m in network.matches:
        if m.hot not in st or m.cold not in st:
            continue
        h, c = st[m.hot], st[m.cold]
        Th, Tc = h.T, c.T
        approach_limit = min(h.s.CP, c.s.CP) * (Th - Tc - dT)
        share_limit = m.max_share_of_hot * h.s.duty - h.used
        Q = max(0.0, min(approach_limit, h.remaining, c.remaining, share_limit))
        if Q <= 1e-6:
            continue
        Th_out, Tc_out = h.take(Q), c.take(Q)
        UA = Q / _lmtd(Th - Tc_out, Th_out - Tc)
        exchangers.append(Exchanger(m, Q, UA, Th, Th_out, Tc, Tc_out))
    residual = [r for r in (s.residual() for s in st.values()) if r is not None]
    return NetworkResult(network, exchangers, residual, ec.direct_loads(residual, pp))


def rate_exchangers(sized: NetworkResult, streams: list[Stream],
                    pp: PlantParameters) -> tuple[list[Exchanger], dict[str, _StreamState]]:
    """Rate each sized exchanger on a stream set, UA held at its design value.

    Returns one exchanger per sized exchanger, in the same order, with zero duty where the
    streams no longer allow any exchange, so callers can index duties by position.
    """
    # Off design a stream can change side, as the reactor effluent does when the column
    # runs above about 1 bar.  A match whose streams have done so exchanges nothing,
    # rather than invalidating the whole network.
    st = {s.name: _StreamState(s) for s in streams}
    out = []
    for ex in sized.exchangers:
        m = ex.match
        if (m.hot not in st or m.cold not in st
                or not st[m.hot].s.is_hot or st[m.cold].s.is_hot):
            out.append(Exchanger(m, 0.0, ex.UA, np.nan, np.nan, np.nan, np.nan))
            continue
        h, c = st[m.hot], st[m.cold]
        Th, Tc = h.T, c.T
        Q = 0.0
        if Th > Tc:
            eps = _effectiveness(ex.UA, h.s.CP, c.s.CP)
            Q = eps * min(h.s.CP, c.s.CP) * (Th - Tc)
            Q = max(0.0, min(Q, h.remaining, c.remaining,
                             m.max_share_of_hot * h.s.duty - h.used))
        out.append(Exchanger(m, Q, ex.UA, Th, h.take(Q), Tc, c.take(Q)))
    return out, st


def rate(sized: NetworkResult, op: OperatingPoint, pp: PlantParameters) -> NetworkResult:
    """Rate a sized network at another operating point, UA held at its design value.

    Duty is the counter-current effectiveness times the largest possible exchange,
    reduced by bypass to no more than the process needs and to the match's share limit.
    """
    streams = stream_table(op, sized.network.pressure, pp)
    exchangers, st = rate_exchangers(sized, streams, pp)
    residual = [r for r in (s.residual() for s in st.values()) if r is not None]
    return NetworkResult(sized.network, exchangers, residual, ec.direct_loads(residual, pp))


# ------------------------------------------------------------------------------------
# The designs of the design comparison
# ------------------------------------------------------------------------------------
#
# Chosen from the stream tables and pinch targets at the design operating point.  D2 is
# the pinch-analysis target itself, a bound rather than a network, and has no entry here.
#
#   D3 and D5 each reach the minimum-utility target at their pressures with a single
#   exchanger, and so cost the same at design.  They differ in the loop they close: D3
#   feeds reactor heat back into the reactor inlet, D5 carries the column's overhead
#   heat to it.
#
#   D4 is the only practical network below the mid-pressure GWP target.  It runs the
#   column under vacuum so that reactor heat can reach the reboiler, and pays for a
#   refrigerated condenser.  A bottoms-to-feed match was tried and rejected: it pushes the
#   effluent's remaining cooling below the reach of cooling water.

PUMP_AROUND_SHARE = 0.7  # of the jacket duty; the jacket keeps the rest to stabilize the reactor

DESIGNS = {
    "D1": Network("D1", "no heat integration", ColumnParameters().P_base),
    "D3": Network("D3", "feed-effluent exchanger", 0.4,
                  (Match("reactor effluent", "reactor feed"),)),
    "D4": Network("D4", "vacuum column, reactor heat to reboiler", 0.15,
                  (Match("reactor effluent", "reboiler"),
                   Match("reactor heat", "reboiler", PUMP_AROUND_SHARE),
                   Match("reactor effluent", "reactor feed"))),
    "D5": Network("D5", "condenser heat to reactor feed", 0.8,
                  (Match("condenser", "reactor feed"),)),
}


def vacuum_network(share: float) -> Network:
    """D4 with a different pump-around share; D4 itself is `vacuum_network(PUMP_AROUND_SHARE)`."""
    matches = [Match("reactor effluent", "reboiler")]
    if share > 0:
        matches.append(Match("reactor heat", "reboiler", share))
    matches.append(Match("reactor effluent", "reactor feed"))
    return Network("D4", f"vacuum column, pump-around share {share}", DESIGNS["D4"].pressure,
                   tuple(matches))
