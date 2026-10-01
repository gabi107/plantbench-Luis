"""A first-order toy plant, so the control layer and the case interface are tested
without any particular case.

    dx/dt = (K m + d - x) / tau

`m` is the manipulated input, `d` a disturbance input, `z` an array-valued input and
`note` an input that is None, so input recording is exercised on every kind of field.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from plantbench.core import control as ctl
from plantbench.core.case import Case, Config
from plantbench.core.control import Loop

K, TAU = 2.0, 5.0


@dataclass
class ToyInputs:
    m: float = 1.0
    d: float = 0.0
    z: np.ndarray = field(default_factory=lambda: np.array([0.25, 0.75]))
    note: float | None = None


@dataclass
class ToyParameters:
    K: float = K
    tau: float = TAU


@dataclass
class ToyDesign:
    x: np.ndarray
    u: ToyInputs
    pp: ToyParameters
    derived: ClassVar[dict] = {"m plus d": lambda u: u.m + u.d}

    def rhs(self, t, x, u, pp):
        return np.array([(pp.K * u.m + u.d - x[0]) / pp.tau])


def toy_design(pp: ToyParameters | None = None, m: float = 1.0) -> ToyDesign:
    pp = pp or ToyParameters()
    return ToyDesign(x=np.array([pp.K * m]), u=ToyInputs(m=m), pp=pp)


def measure_x(x, u, pp):
    return float(x[0])


def toy_loop(**kw) -> Loop:
    return Loop("level", measure_x, "m", Kc=0.5, tau_I=2.0, **kw)


def _make_design(config: Config):
    from plantbench.core.case import override
    return toy_design(override(ToyParameters(), config.params),
                      m=float(config.options["m_design"]))


def _make_structure(design, config: Config):
    return ctl.bias_from_design([toy_loop()], design)


TOY_CASE = Case(
    id="toy",
    title="First-order toy plant",
    summary="One state, one PI loop.",
    options={"m_design": 1.0},
    structures=("PI",),
    default_structure="PI",
    make_design=_make_design,
    make_structure=_make_structure,
    disturbances={},
    measurements=lambda design: {"x": measure_x},
    state_names=lambda design: ["x"],
)
