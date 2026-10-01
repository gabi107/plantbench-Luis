"""Set-points and tuning are configuration, not code.

    .venv/bin/python examples/02_setpoints_and_tuning.py

A +2 K set-point step on the reactor temperature, under three tunings of the loop.  The
damping ratio of the least damped closed-loop mode is computed from the linearized plant
and the overshoot from the nonlinear run.  They need not agree: the least damped mode is
not necessarily the one a set-point step excites.
"""

import numpy as np

import plantbench as pb
from plantbench.core import control as ctl

case = pb.load_case("jacketed_cstr")

print(f"{'Kc':>6} {'tau_I':>6} {'damping':>8} {'overshoot, K':>13}")
for Kc, tau_I in ((-0.5, 20.0), (-1.0, 20.0), (-2.0, 5.0)):
    config = case.config("temperature PI",
                         tuning={"reactor T": {"Kc": Kc, "tau_I": tau_I}},
                         setpoint_steps={"reactor T": [(10.0, 2.0)]})
    zeta = ctl.damping_ratio(pb.build(case, config).spectrum())
    tr = pb.run(case, config, t_end=300.0, dt=0.5)
    T = tr.y["T"]
    overshoot = max(np.max(T) - (T[0] + 2.0), 0.0)
    print(f"{Kc:6.2f} {tau_I:6.1f} {zeta:8.3f} {overshoot:13.3f}")
