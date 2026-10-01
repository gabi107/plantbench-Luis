"""Process streams for heat integration.

A stream is a supply temperature, a target temperature and a heat-capacity flow rate.
Whether it is hot or cold follows from the two temperatures.  Pinch targeting and
network sizing act on lists of these; which streams a plant has is the case's business.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Stream:
    """A process stream for heat integration: supply and target temperature, CP."""

    name: str
    T_supply: float  # K
    T_target: float  # K
    CP: float  # kJ/(min K)

    @property
    def is_hot(self) -> bool:
        return self.T_supply > self.T_target

    @property
    def duty(self) -> float:
        """kJ/min, always positive."""
        return abs(self.T_supply - self.T_target) * self.CP
