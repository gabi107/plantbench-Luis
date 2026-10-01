"""Run a case: the reference plant, then the same disturbance with the loop open.

    .venv/bin/python examples/01_run_case.py

`jacketed_cstr` is the reactor of `reactor_separator_recycle` on its unstable middle branch.  A feed at 340 K, 3 K
hotter than design, carries it to the high-temperature branch in open loop; the
temperature loop holds it.
"""

import plantbench as pb

case = pb.load_case("jacketed_cstr")
hotter = {"name": "feed_temperature", "value": 340.0, "t": 10.0}

for structure in ("open loop", "temperature PI"):
    config = case.config(structure, disturbance=hotter)
    tr = pb.run(case, config, t_end=600.0, dt=1.0)
    T = tr.y["T"]
    print(f"{structure:16s} T {T[0]:.2f} -> {T[-1]:.2f} K, coolant {tr.u['Fj'][0]:.3f} -> "
          f"{tr.u['Fj'][-1]:.3f} m3/min  ({tr.wall_time:.2f} s)")
