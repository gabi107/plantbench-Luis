"""Produce every number quoted in the write-up, in one run.

    python -m studies.reactor_separator_recycle.reference [--out DIR]

Writes results/reference.txt (or DIR/reference.txt) as well as printing, so the document can
be checked against a single artifact rather than against a set of remembered runs.
"""

from __future__ import annotations

import argparse
import io
import pathlib
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from plantbench.cases.reactor_separator_recycle import measurements as meas
from plantbench.cases.reactor_separator_recycle import plant
from plantbench.cases.reactor_separator_recycle import scenarios as sc
from plantbench.cases.reactor_separator_recycle.parameters import PlantParameters
from plantbench.core import control as ctl
from plantbench.units import column as col
from plantbench.units import cstr, thermo
from plantbench.units.parameters import (
    REFERENCE_EIGENVALUES,
    REFERENCE_STEADY_STATE,
    reference_parameters,
)

T_END = 5000.0
N_POINTS = 500
OUT = io.StringIO()


def say(s=""):
    print(s, flush=True)
    OUT.write(s + "\n")


def rightmost_eigenvalue(design, structure):
    y0 = np.concatenate([design.x, np.zeros(len(structure))])
    f = lambda y: ctl.closed_loop_rhs(0.0, y, structure, design.u, design.pp, None, design.rhs)
    n = len(y0)
    J = np.empty((n, n))
    for j in range(n):
        h = 1e-7 * max(1.0, abs(y0[j]))
        yp, ym = y0.copy(), y0.copy()
        yp[j] += h
        ym[j] -= h
        J[:, j] = (f(yp) - f(ym)) / (2 * h)
    return np.linalg.eigvals(J).real.max()


def main(argv=None):
    import dataclasses

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=pathlib.Path,
                        default=pathlib.Path(__file__).resolve().parent / "results",
                        help="directory for reference.txt")
    args = parser.parse_args(argv)

    p = reference_parameters()

    say("=" * 78)
    say("1.  REACTOR AGAINST THE REFERENCE (ROMAGNOLI AND PALAZOGLU, 2020)")
    say("=" * 78)
    ss = cstr.steady_state(p)
    lam = np.sort(cstr.eigenvalues(ss, p).real)
    say(f"steady state   C_A {ss[0]:.4f}   T {ss[1]:.3f}   T_j {ss[2]:.3f}")
    say("reference      C_A {CA:.4f}   T {T:.3f}   T_j {Tj:.3f}".format(
        **REFERENCE_STEADY_STATE))
    # Round-off below 1e-12 is printed as a bound, so the file does not depend on the last bit.
    r = np.max(np.abs(cstr.rhs_constant_volume(0, ss, p)))
    say(f"residual       {'< 1e-12' if r < 1e-12 else f'{r:.1e}'}")
    say(f"eigenvalues    {lam[2]:+.4f}  {lam[1]:+.4f}  {lam[0]:+.4f}")
    say(f"reference      {REFERENCE_EIGENVALUES[0]:+.4f}  "
        f"{REFERENCE_EIGENVALUES[1]:+.4f}  {REFERENCE_EIGENVALUES[2]:+.4f}")
    say(f"branches, K    {np.round(cstr.steady_state_branches(p), 2)}")
    say(f"k at T_s       {cstr.rate_constant(ss[1], p):.4f} 1/min")
    say(f"conversion     {(p.CA0 - ss[0]) / p.CA0:.1%}")

    say()
    say("=" * 78)
    say("2.  PLANTWIDE DESIGN")
    say("=" * 78)
    design = plant.solve_design(PlantParameters(), V_boil=28.0)
    say(design.summary())
    cp = design.pp.column
    _, _, M, _, _, _ = col.unpack(design.x[6:], cp)
    L = col.weir_flow(M, cp)
    say(f"internal liquid  above feed {L[0]:.3f}   below feed {L[-1]:.3f} kmol/min")

    say()
    say("=" * 78)
    say("3.  REACTOR TEMPERATURE LOOP: STABILIZING GAIN")
    say("=" * 78)
    taus = (2.0, 3.2, 5.0, 10.0, 20.0)
    say("rightmost closed-loop eigenvalue of the whole plant")
    say("      Kc | " + "  ".join(f"tauI={t:<5.1f}" for t in taus))
    for Kc in (-0.10, -0.30, -0.60, -1.00, -2.00):
        row = []
        for tau in taus:
            S = sc.recycle_free(design)
            S[0] = dataclasses.replace(S[0], Kc=Kc, tau_I=tau)
            e = rightmost_eigenvalue(design, S)
            row.append("stable    " if e < 1e-6 else f"{e:+.4f}   ")
        say(f"  {Kc:6.2f} | " + " ".join(row))

    say()
    say("=" * 78)
    say("4.  THROUGHPUT CHANGES")
    say("=" * 78)
    say("throughput moved at the fresh feed (recycle free) or at the effluent flow (others)")
    say(f"{'structure':26s} {'step':>6} {'production':>11} {'recycle':>10} "
        f"{'ampl':>7} {'C_A':>8} {'T':>8} {'V_boil':>7} {'x_B,A peak':>11} {'x_B,A end':>10}")
    rows = {}
    for name in ("recycle free", "effluent fixed", "effluent fixed + cascade"):
        S = sc.STRUCTURES[name](design)
        h = sc.THROUGHPUT_HANDLE[name]
        for frac in (0.05, 0.10, 0.20, 0.25):
            try:
                t, X, U = ctl.simulate(
                    design, S, t_end=T_END, n_points=N_POINTS,
                    disturbance=sc.throughput_step(frac, h, 30.0),
                )
            except Exception as exc:  # noqa: BLE001
                say(f"{name:26s} {frac:+6.0%}   integration failed: "
                    f"{str(exc)[:40]}")
                continue
            MD = col.unpack(X[6:, -1], cp)[0]
            xBA = np.array([col.unpack(X[6:, i], cp)[5][1] for i in range(X.shape[1])])
            if MD > 10 * cp.drum_holdup:
                say(f"{name:26s} {frac:+6.0%}   no steady state: drum holdup "
                    f"{MD:.0f} kmol against a design {cp.drum_holdup:.0f}")
                continue
            # A run that ends with a valve on its limit has lost that loop: whatever it
            # settles to is not an operating point of the structure.
            limits = [(lp, U[lp.mv][-1]) for lp in S
                      if isinstance(lp, ctl.Loop) and not lp.mv.startswith("sp:")
                      and (U[lp.mv][-1] <= lp.lo + 1e-6 or U[lp.mv][-1] >= lp.hi - 1e-6)]
            if limits:
                which = ", ".join(f"{lp.name} ({lp.mv} {v:.2f})" for lp, v in limits)
                say(f"{name:26s} {frac:+6.0%}   loop at a limit: {which}; reactor "
                    f"{X[4, -1]:.2f} K, bottoms x_A {xBA[-1]:.3f}")
                continue
            prod = U["Bm"][-1] / U["Bm"][0] - 1
            R = U["recycle"][-1] / U["recycle"][0] - 1
            rows[(name, frac)] = (prod, R)
            say(f"{name:26s} {frac:+6.0%} {prod:+11.2%} {R:+10.2%} "
                f"{R / prod:6.2f}x {X[2, -1]:8.4f} {X[4, -1]:8.2f} "
                f"{U['V_boil'][-1]:7.2f} {xBA.max():11.2e} {xBA[-1]:10.2e}")

    say()
    say("=" * 78)
    say("5.  FIXING THE DISTILLATE")
    say("=" * 78)
    S = sc.distillate_fixed(design)
    t, X, U = ctl.simulate(design, S, t_end=T_END, n_points=N_POINTS,
                           disturbance=sc.feed_step(0.10, 30.0))
    MD = np.array([col.unpack(X[6:, i], cp)[0] for i in range(X.shape[1])])
    say(f"10% fresh feed increase, {T_END:.0f} min")
    say(f"reflux drum holdup {MD[0]:.2f} -> {MD[-1]:.2f} kmol, still rising: "
        f"{MD[-1] > MD[-2]}")
    say(f"reflux at its limit: {U['L_reflux'][-1]:.2f} kmol/min")

    say()
    say("=" * 78)
    say("6.  INERT INVENTORY")
    say("=" * 78)
    S = sc.plantwide(design)
    # The purge-closed case has no steady state to reach; 3000 min is long enough to see
    # the inert displace the reactant and the reactor lose its temperature control.
    for label, dist, horizon in (
        ("purge open, +50% feed inert", sc.inert_step(0.5, 30.0), T_END),
        ("purge closed", sc.purge_closed(30.0), 3000.0),
    ):
        t, X, U = ctl.simulate(design, S, t_end=horizon, n_points=N_POINTS,
                               disturbance=dist)
        xI = np.array([col.unpack(X[6:, i], cp)[1][0] for i in range(X.shape[1])])
        inv = np.array([sum(meas.inert_inventory(X[:, i], design.pp))
                        for i in range(X.shape[1])])
        say(f"{label}")
        at3000 = xI[min(np.searchsorted(t, 3000.0), len(t) - 1)]
        say(f"    inert mole fraction in recycle  {xI[0]:.4f} -> {xI.max():.4f} "
            f"(peak) -> {xI[-1]:.4f} (end, {t[-1]:.0f} min); {at3000:.4f} at 3000 min")
        line = (f"    inert inventory, kmol           {inv[0]:.2f} -> {inv[-1]:.2f}  "
                f"(accumulated {inv[-1] - inv[0]:.2f}")
        if "closed" in label:
            # Only with the purge shut does everything fed have to stay, so only then
            # is the comparison against the amount fed meaningful.
            # The fresh feed holds reactor level, so it moves: integrate it from the
            # moment the purge shuts.
            after = t >= 30.0
            fed = np.trapezoid(U["F_fresh"][after] * design.pp.z_inert, t[after])
            line += f", fed {fed:.2f}"
        say(line + ")")
        say(f"    reflux drum holdup, kmol        "
            f"{col.unpack(X[6:, 0], cp)[0]:.2f} -> "
            f"{col.unpack(X[6:, -1], cp)[0]:.2f}")
        if "closed" in label:
            for tk in (600.0, 1500.0, 3000.0):
                i = min(np.searchsorted(t, tk), len(t) - 1)
                say(f"    at {t[i]:4.0f} min: production {U['Bm'][i]:.3f} kmol/min "
                    f"({U['Bm'][i] / U['Bm'][0] - 1:+.0%}), fresh feed {U['F_fresh'][i]:.3f}, "
                    f"reactor {X[4, i]:.2f} K, jacket flow {U['Fj'][i]:.3f} m3/min")
            at_limit = np.where(U["Fj"] <= 0.05 + 1e-6)[0]
            if at_limit.size:
                say(f"    jacket flow first at its 0.05 m3/min lower limit at {t[at_limit[0]]:.0f} min")

    say()
    say("=" * 78)
    say("7.  PRODUCT QUALITY: TRAY TEMPERATURE LOOP")
    say("=" * 78)
    tray = sc.TEMPERATURE_TRAY
    temps = thermo.column_temperatures(design.x[6:], cp.P_base, design.pp.thermo, cp)["trays"]
    steps = np.diff(temps)
    say(f"column pressure {cp.P_base} bar; tray {tray} (1 = top, feed on {cp.feed_tray})")
    say(f"tray temperatures {temps[tray - 2]:.2f} / {temps[tray - 1]:.2f} / {temps[tray]:.2f} K "
        f"(trays {tray - 1}-{tray + 1}); steepest step below the feed tray: "
        f"trays {cp.feed_tray + 1 + int(np.argmax(steps[cp.feed_tray:]))}-"
        f"{cp.feed_tray + 2 + int(np.argmax(steps[cp.feed_tray:]))}, {steps[cp.feed_tray:].max():.2f} K")
    say(f"tuning Kc {sc.TRAY_T_TUNING['Kc']} kmol/min per K, tau_I {sc.TRAY_T_TUNING['tau_I']} min; "
        f"reflux ratioed to the column feed at {design.u.L_reflux / design.u.F_out:.4f} kmol/min per m3/min")
    _, xD0, _, _, _, xB0 = col.unpack(design.x[6:], cp)
    say(f"design: bottoms x_A {xB0[1]:.2e}, distillate x_B {xD0[2]:.4f}")
    ev = ctl.closed_loop_spectrum(design, sc.plantwide(design))
    say(f"effluent fixed + cascade: rightmost eigenvalue {ev.real.max():+.4f}, "
        f"least damping {ctl.damping_ratio(ev):.3f}")

    path = args.out / "reference.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(OUT.getvalue())
    print(f"\nwritten to {path}")


if __name__ == "__main__":
    main()
