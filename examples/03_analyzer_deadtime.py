"""A non-ideality is a configuration: analyzer deadtime on `reactor_separator_recycle`'s composition loop.

    .venv/bin/python examples/03_analyzer_deadtime.py

The reference plant of `reactor_separator_recycle` measures the reactor composition instantly.  A
chromatograph has a cycle time.  Its deadtime enters as a Pade approximation, so the
closed loop stays an ODE and its eigenvalues remain computable; the reference results
are unchanged because an unconfigured instrument adds no states.
"""

import plantbench as pb
from plantbench.core import control as ctl

case = pb.load_case("reactor_separator_recycle")

print(f"{'deadtime, min':>14} {'states':>7} {'rightmost, 1/min':>17} {'damping':>8}")
for deadtime in (0.0, 2.0, 5.0, 10.0):
    instruments = {"reactor C_A": {"deadtime": deadtime, "order": 2}} if deadtime else {}
    setup = pb.build(case, case.config(instruments=instruments))
    ev = setup.spectrum()
    n = len(setup.design.x) + ctl.n_augmented(setup.structure)
    print(f"{deadtime:14.1f} {n:7d} {ev.real.max():+17.4f} {ctl.damping_ratio(ev):8.3f}")
