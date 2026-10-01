"""Flowsheet-level and economic parameters of `reactor_separator_recycle`, the reactor / column / recycle plant.

The unit-operation parameter sets it composes (reactor, column, thermodynamics) live in
`plantbench.units.parameters`.  Units are minutes, cubic meters, kmol and kelvin.
"""

from __future__ import annotations

from dataclasses import dataclass

from plantbench.units.parameters import ColumnParameters, ReactorParameters, ThermoParameters


@dataclass(frozen=True)
class EconomicParameters:
    """Prices, utility data and emission factors.  Every value here is an assumption.

    The chemical is notional, so none of these can be looked up; they are chosen to be
    of the order found for a bulk organic intermediate made with natural-gas steam and
    grid electricity, and each enters the results only through the formulas in
    `economics.py`, so a reader can substitute their own.
    """

    # -- products and raw material ----------------------------------------------
    price_product: float = 1.10  # $/kg B
    price_feed: float = 0.95  # $/kg fresh A
    feed_burden: float = 2.0  # kg CO2e per kg fresh A, cradle to plant gate

    # -- steam, raised in a natural-gas boiler ----------------------------------
    gas_price: float = 4.0  # $/GJ of fuel
    boiler_efficiency: float = 0.85
    gas_emission_factor: float = 56.1  # kg CO2/GJ fuel (natural gas)
    # Distribution and water treatment, higher for higher pressure.
    steam_adder: tuple[float, float, float] = (1.0, 1.5, 2.5)  # $/GJ for LP, MP, HP
    steam_T: tuple[float, float, float] = (417.0, 457.0, 523.0)  # K, about 4, 11, 40 bar

    # -- electricity -------------------------------------------------------------
    price_electricity: float = 0.08  # $/kWh
    grid_intensity: float = 0.40  # kg CO2e/kWh

    # -- cooling -------------------------------------------------------------------
    cw_T: float = 308.0  # K, mean of a 303 -> 313 K cooling water circuit
    cw_price: float = 0.35  # $/GJ removed
    cw_pumping: float = 0.4  # kWh of electricity per GJ removed
    chw_T: float = 283.0  # K, mean of a 278 -> 288 K chilled water circuit
    chiller_cop: float = 4.0


@dataclass(frozen=True)
class PlantParameters:
    """Flowsheet-level parameters for the reactor / column / recycle plant."""

    reactor: ReactorParameters = ReactorParameters()
    column: ColumnParameters = ColumnParameters()

    # Constant liquid molar density.  The reactor is written in concentration units,
    # as the reference gives it; the column is molar.  This is the conversion between
    # them, and it is chosen so that C_A0 = 5 kmol/m3 at the reactor inlet corresponds
    # to a mole fraction below one, leaving room for the product and the inert that
    # the recycle carries back.
    rho_molar: float = 5.60  # kmol/m3

    # Fresh feed: reactant with a small inert impurity.  0.5 mol% is what the
    # flowsheet can carry: the inert leaves only in the purge, so a richer feed
    # forces a larger purge and the purge takes reactant with it.
    z_inert: float = 0.005

    # Reactor level control.  The reference assumes constant volume and
    # invites relaxing it so that the flows can be used for level control, which is
    # what a plantwide structure needs.
    V_nominal: float = 22.82  # m3, level set-point

    thermo: ThermoParameters = ThermoParameters()
    economics: EconomicParameters = EconomicParameters()

    @property
    def cp_molar(self) -> float:
        """Liquid molar heat capacity, kJ/(kmol K).

        Follows from quantities already fixed: the reactor's mass heat capacity and
        density (the reference) and the constant molar density above, which together imply
        a molar mass of rho / rho_molar = 268 kg/kmol.
        """
        return self.reactor.rho * self.reactor.cp / self.rho_molar

    @property
    def molar_mass(self) -> float:
        """kg/kmol, the same for all three species under constant molar density."""
        return self.reactor.rho / self.rho_molar
