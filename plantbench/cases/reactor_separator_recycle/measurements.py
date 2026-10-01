"""Measurements of `reactor_separator_recycle`: functions of the plant state and inputs a loop can control.

Each takes (x, u, pp) and returns a float, which is the signature `core.control.Loop`
expects of its `measure`.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from plantbench.units import column as col
from plantbench.units import thermo

from .parameters import PlantParameters
from .plant import N_REACTOR_STATES, Inputs

I, A, B = 0, 1, 2




def m_reactor_volume(x, u, pp):
    return x[0]


def m_reactor_T(x, u, pp):
    return x[4]


def m_reactor_CA(x, u, pp):
    return x[1 + A]


def m_drum_level(x, u, pp):
    return col.unpack(x[N_REACTOR_STATES:], pp.column)[0]


def m_base_level(x, u, pp):
    return col.unpack(x[N_REACTOR_STATES:], pp.column)[4]


def m_bottoms_A(x, u, pp):
    return col.unpack(x[N_REACTOR_STATES:], pp.column)[5][A]


def m_tray_T(tray: int, P: float) -> Callable[[np.ndarray, Inputs, PlantParameters], float]:
    """Temperature of `tray` (1 = top) at column pressure P: the bubble point of its liquid."""

    def measure(x, u, pp):
        _, _, _, xs, _, _ = col.unpack(x[N_REACTOR_STATES:], pp.column)
        return float(thermo.bubble_point(xs[tray - 1], P, pp.thermo, pp.column))

    return measure


def m_recycle(x, u, pp):
    return u.D - u.P


def inert_inventory(x, pp) -> tuple[float, float]:
    """Moles of inert held in the reactor and in the column.

    The inert has no reaction term and, with the purge shut, no exit, so its total is
    the quantity Step 7 asks to be checked.  Watching its mole fraction instead is
    misleading once a holdup starts to run away, because the fraction can fall while
    the inventory rises.
    """
    reactor = x[0] * x[1 + I]  # m3 * kmol/m3
    MD, xD, M, xs, MB, xB = col.unpack(x[N_REACTOR_STATES:], pp.column)
    column = MD * xD[I] + float((M * xs[:, I]).sum()) + MB * xB[I]
    return reactor, column
