"""Utilities, operating profit and global warming potential.

Operating profit follows §19.4:

    profit = product value - raw material cost - utility cost

The environmental objective is global warming potential (GWP), cradle to gate.  It has
two parts that behave differently.  The burden of producing the fresh feed depends only
on how much feed the plant consumes, which the material balance fixes; at a given
production rate it is the same for every heat-exchanger network.  The emissions of
raising steam, generating electricity and circulating cooling water are what heat
integration changes.  Both are computed, and the comparison between designs rests on the
second.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from plantbench.heat import pinch
from plantbench.heat.streams import Stream

from .parameters import EconomicParameters, PlantParameters
from .streams import OperatingPoint

GJ_PER_KWH = 0.0036
KJ_PER_MIN_TO_GJ_PER_H = 60.0 / 1e6


@dataclass(frozen=True)
class Utility:
    name: str
    T: float  # K, placement temperature
    hot: bool
    cost: float  # $/GJ delivered or removed
    gwp: float  # kg CO2e/GJ delivered or removed


def utilities(ep: EconomicParameters) -> list[Utility]:
    """The utility set, with prices and emission factors derived from the assumptions."""
    steam_cost = [ep.gas_price / ep.boiler_efficiency + a for a in ep.steam_adder]
    steam_gwp = ep.gas_emission_factor / ep.boiler_efficiency
    elec_cost = ep.price_electricity / GJ_PER_KWH  # $/GJ of electricity
    elec_gwp = ep.grid_intensity / GJ_PER_KWH  # kg/GJ of electricity
    cw_gwp = ep.cw_pumping * ep.grid_intensity
    # A chiller removes 1 GJ with 1/COP GJ of electricity and rejects both to cooling water.
    reject = 1.0 + 1.0 / ep.chiller_cop
    chw_cost = elec_cost / ep.chiller_cop + reject * ep.cw_price
    chw_gwp = elec_gwp / ep.chiller_cop + reject * cw_gwp
    names = ("LP steam", "MP steam", "HP steam")
    out = [Utility(n, T, True, c, steam_gwp) for n, T, c in zip(names, ep.steam_T, steam_cost)]
    out.append(Utility("cooling water", ep.cw_T, False, ep.cw_price, cw_gwp))
    out.append(Utility("chilled water", ep.chw_T, False, chw_cost, chw_gwp))
    return out


def _empty_loads(ep: EconomicParameters) -> dict[str, float]:
    return {u.name: 0.0 for u in utilities(ep)}


def direct_loads(streams: list[Stream], pp: PlantParameters) -> dict[str, float]:
    """Utility loads when every stream is met by utilities on its own.

    A stream to be heated takes the coldest steam that clears its outlet by the minimum
    approach.  A stream to be cooled gives the part of its duty released above the
    cooling-water limit to cooling water and the rest to chilled water.
    """
    ep, dT = pp.economics, pp.thermo.dT_min
    util = utilities(ep)
    loads = _empty_loads(ep)
    steam = sorted((u for u in util if u.hot), key=lambda u: u.T)
    cw = next(u for u in util if u.name == "cooling water")
    chw = next(u for u in util if u.name == "chilled water")
    for s in streams:
        if not s.is_hot:
            level = next((u for u in steam if u.T >= s.T_target + dT), None)
            if level is None:
                raise ValueError(f"no steam level hot enough for {s.name} at {s.T_target:.1f} K")
            loads[level.name] += s.duty
        else:
            T_cw, T_chw = cw.T + dT, chw.T + dT
            if s.T_target < T_chw - 1e-9:
                raise ValueError(f"{s.name} must be cooled below chilled-water reach")
            above = s.CP * max(0.0, s.T_supply - max(s.T_target, T_cw))
            loads[cw.name] += above
            loads[chw.name] += s.duty - above
    return loads


def target_loads(streams: list[Stream], pp: PlantParameters) -> dict[str, float]:
    """Utility loads at the pinch-analysis minimum, placed on the cheapest levels."""
    ep, dT = pp.economics, pp.thermo.dT_min
    util = utilities(ep)
    levels = [pinch.UtilityLevel(u.name, u.T, u.hot) for u in util]
    extra = tuple(u.T - dT / 2 if u.hot else u.T + dT / 2 for u in util)
    pt = pinch.problem_table(streams, dT, extra_boundaries=extra)
    return pinch.place_utilities(pt, levels)


@dataclass
class Evaluation:
    """Objectives and their parts at one operating point, per hour."""

    revenue: float  # $/h
    feed_cost: float  # $/h
    utility_cost: float  # $/h
    gwp_energy: float  # t CO2e/h
    gwp_feed: float  # t CO2e/h
    loads: dict[str, float] = field(default_factory=dict)  # kJ/min

    @property
    def profit(self) -> float:
        return self.revenue - self.feed_cost - self.utility_cost

    @property
    def gwp_total(self) -> float:
        return self.gwp_energy + self.gwp_feed

    @property
    def hot_utility(self) -> float:
        return sum(v for k, v in self.loads.items() if "steam" in k)

    @property
    def cold_utility(self) -> float:
        return sum(v for k, v in self.loads.items() if "water" in k)


def evaluate(op: OperatingPoint, loads: dict[str, float], pp: PlantParameters) -> Evaluation:
    ep = pp.economics
    kg_per_h = pp.molar_mass * 60.0  # kmol/min -> kg/h
    by_name = {u.name: u for u in utilities(ep)}
    utility_cost = sum(q * KJ_PER_MIN_TO_GJ_PER_H * by_name[n].cost for n, q in loads.items())
    gwp_energy = sum(q * KJ_PER_MIN_TO_GJ_PER_H * by_name[n].gwp for n, q in loads.items())
    return Evaluation(
        revenue=op.Bm * kg_per_h * ep.price_product,
        feed_cost=op.F_fresh * kg_per_h * ep.price_feed,
        utility_cost=utility_cost,
        gwp_energy=gwp_energy / 1000.0,
        gwp_feed=op.F_fresh * kg_per_h * ep.feed_burden / 1000.0,
        loads=dict(loads),
    )
