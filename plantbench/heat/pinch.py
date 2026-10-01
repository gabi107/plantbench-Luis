"""Pinch analysis: the problem-table algorithm, utility targets and composite curves.

Streams are shifted by half the minimum approach, hot down and cold up, so that any
temperature interval in the shifted scale can pass heat from its hot streams to its cold
streams without violating the approach.  Cascading each interval's surplus from the top
gives the minimum hot utility (the largest deficit that has to be made up) and, by the
overall balance, the minimum cold utility.

Utilities at several temperature levels are placed on the heat cascade.  Hot utility
injected at a level can only replace heat that would otherwise flow down through every
boundary at or above that level, so the most a level can supply is the smallest cascade
flow at or above it.  Cold utility is placed symmetrically below.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .streams import Stream


@dataclass(frozen=True)
class UtilityLevel:
    name: str
    T: float  # K, the utility's temperature for placement
    hot: bool


@dataclass
class ProblemTable:
    dT_min: float
    boundaries: np.ndarray  # shifted temperatures, descending
    surplus: np.ndarray  # kJ/min per interval between consecutive boundaries
    flows: np.ndarray  # cascade heat flow at each boundary, feasible cascade
    QH_min: float
    QC_min: float
    pinch_shifted: float | None  # None when no heat crosses zero (threshold problem)

    @property
    def pinch_hot(self) -> float | None:
        return None if self.pinch_shifted is None else self.pinch_shifted + self.dT_min / 2

    @property
    def pinch_cold(self) -> float | None:
        return None if self.pinch_shifted is None else self.pinch_shifted - self.dT_min / 2

    def flow_at(self, Ts: float) -> float:
        """Cascade heat flow at shifted temperature Ts, interpolated between boundaries."""
        b, f = self.boundaries[::-1], self.flows[::-1]  # ascending for np.interp
        return float(np.interp(Ts, b, f))


def _shifted(s: Stream, dT_min: float) -> tuple[float, float]:
    d = -dT_min / 2 if s.is_hot else dT_min / 2
    return s.T_supply + d, s.T_target + d


def problem_table(streams: list[Stream], dT_min: float,
                  extra_boundaries: tuple[float, ...] = ()) -> ProblemTable:
    """Run the problem-table algorithm.

    `extra_boundaries` are shifted temperatures to include as interval edges without
    changing any interval's surplus.  Utility levels are passed here so that the
    cascade flow is evaluated exactly at each of them.
    """
    shifted = [(s, *_shifted(s, dT_min)) for s in streams]
    temps = {t for _, a, b in shifted for t in (a, b)}
    lo, hi = min(temps), max(temps)
    temps |= {t for t in extra_boundaries if lo <= t <= hi}
    bounds = np.array(sorted(temps, reverse=True))

    surplus = np.zeros(len(bounds) - 1)
    for k in range(len(bounds) - 1):
        top, bot = bounds[k], bounds[k + 1]
        for s, a, b in shifted:
            s_hi, s_lo = max(a, b), min(a, b)
            overlap = max(0.0, min(top, s_hi) - max(bot, s_lo))
            if overlap > 0:
                surplus[k] += (1.0 if s.is_hot else -1.0) * s.CP * overlap

    cascade = np.concatenate([[0.0], np.cumsum(surplus)])
    QH = max(0.0, -cascade.min())
    flows = cascade + QH
    QC = float(flows[-1])
    # A pinch exists only when both utilities are needed; otherwise the problem is a
    # threshold problem and the zero-flow boundary is just its top or bottom.
    k = int(np.argmin(flows))
    pinch = float(bounds[k]) if QH > 0 and QC > 0 else None
    return ProblemTable(dT_min, bounds, surplus, flows, float(QH), QC, pinch)


def place_utilities(pt: ProblemTable, levels: list[UtilityLevel]) -> dict[str, float]:
    """Split the minimum utility targets across utility levels, cheapest placement first.

    Hot levels are filled from the lowest temperature upward, cold levels from the
    highest downward, each taking as much as the cascade allows at its temperature.  A
    level's placement temperature is shifted like a stream: hot utility down by half the
    approach, cold utility up.
    """
    half = pt.dT_min / 2

    def cap_hot(Ts):
        # Largest hot utility deliverable at shifted temperature Ts or above.
        if Ts >= pt.boundaries[0]:
            return pt.QH_min
        return min([pt.flow_at(Ts)] + [f for b, f in zip(pt.boundaries, pt.flows) if b >= Ts])

    def cap_cold(Ts):
        # Largest cold utility absorbable at shifted temperature Ts or below.
        if Ts <= pt.boundaries[-1]:
            return pt.QC_min
        return min([pt.flow_at(Ts)] + [f for b, f in zip(pt.boundaries, pt.flows) if b <= Ts])

    out: dict[str, float] = {}
    for hot, target, cap, key, shift in (
        (True, pt.QH_min, cap_hot, lambda u: u.T, -half),
        (False, pt.QC_min, cap_cold, lambda u: -u.T, +half),
    ):
        placed = 0.0
        for u in sorted((u for u in levels if u.hot == hot), key=key):
            cumulative = min(cap(u.T + shift), target)
            out[u.name] = max(0.0, cumulative - placed)
            placed = max(placed, cumulative)
        if target > 1e-9 and placed < target * (1 - 1e-9):
            side = "hot" if hot else "cold"
            raise ValueError(f"{side} utility levels cannot meet the {side} utility target: "
                             f"{placed:.1f} of {target:.1f} kJ/min")
    return out


def composite_curve(streams: list[Stream], hot: bool) -> tuple[np.ndarray, np.ndarray]:
    """Enthalpy and temperature of the hot or cold composite, enthalpy from zero."""
    sel = [s for s in streams if s.is_hot == hot]
    if not sel:
        return np.array([0.0]), np.array([0.0])
    temps = sorted({t for s in sel for t in (s.T_supply, s.T_target)})
    H = [0.0]
    for lo, hi in zip(temps[:-1], temps[1:]):
        CP = sum(s.CP for s in sel if min(s.T_supply, s.T_target) <= lo
                 and max(s.T_supply, s.T_target) >= hi)
        H.append(H[-1] + CP * (hi - lo))
    return np.array(H), np.array(temps)


def grand_composite(pt: ProblemTable) -> tuple[np.ndarray, np.ndarray]:
    """Cascade heat flow against shifted temperature."""
    return pt.flows.copy(), pt.boundaries.copy()
