"""The damping ratio of `reactor_separator_recycle` against the analyzer deadtime, and its Sobol' indices.

    python -m studies.design.analysis [--workers N] [--out DIR]
    python -m studies.design.verify --workers N
    python -m studies.design.analysis [--workers N] [--out DIR]

Writes results/results_design.txt.  Two questions are asked of the design space of the
vacuum network D4.  Section 1 sweeps the deadtime of the reactor composition analyzer at
the design points the analytical method enumerated and reads the least damping ratio of
the closed loop linearized there.  Section 2 computes the Sobol' indices of that damping
ratio, and of the realized outcome, over the design variables.  The damping ratio costs a
design solve and is computed here.  The outcome costs a closed-loop run per point: the
first pass writes the sample to results/sobol-outcome.json, verify.py runs it into
data/design-sobol, and the second pass reads it back.

The outcome criterion is that of the regimes study, imported unchanged.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np

from studies.regimes import dataset

HERE = Path(__file__).resolve().parent
# A design is reliable when its worst outcome over the upsets is settled or saturated:
# the plant holds an operating point, with or without a loop on a limit.
RELIABLE = ("settled", "saturated")

OUT = io.StringIO()


def say(s: str = "") -> None:
    print(s, flush=True)
    OUT.write(s + "\n")


def heading(title: str) -> None:
    say()
    say("=" * 88)
    say(title)
    say("=" * 88)


# ------------------------------------------------------------------------------------
# 1.  The damping ratio when the analyzer is not ideal
# ------------------------------------------------------------------------------------

# The designs the deadtime is swept over: the vacuum network at the four pump-around
# shares the analytical method enumerated, and the network it selected.
SWEEP_PLANTS = [("D4", 0.0), ("D4", 0.3), ("D4", 0.5), ("D4", 0.7), ("D3", None)]
SWEEP_DEADTIMES = (0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0)


def _zeta(job) -> tuple[str, float, float, float, float, int]:
    """The damping ratio and rightmost eigenvalue of one plant at one analyzer deadtime."""
    import plantbench as pb
    from plantbench.core import control as ctl
    network, share, deadtime = job
    options = {"network": network}
    if share is not None:
        options["pump_around_share"] = share
    setup = pb.build(dataset.CASE, dataset.CASE.config(
        structure="effluent fixed + cascade", options=options,
        instruments={"reactor C_A": {"order": 2, "deadtime": deadtime}}))
    ev = setup.spectrum()
    return network, share, deadtime, ctl.damping_ratio(ev), float(ev.real.max()), ev.size


def section_deadtime(store, workers: int = 1):
    heading("1.  THE DAMPING RATIO WHEN THE ANALYZER IS NOT IDEAL")
    say("The analytical method priced operability by the least damping ratio of the closed "
        "loop\nlinearized at the design point, holding every measurement ideal.  The "
        "linearization carries\nwhatever instrument the configuration names, so the same "
        "criterion can be read with the\ncomposition analyzer in place.")
    jobs = [(n, s, d) for n, s in SWEEP_PLANTS for d in SWEEP_DEADTIMES]
    if workers <= 1:
        rows = [_zeta(j) for j in jobs]
    else:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(_zeta, jobs))
    say(f"\n  {'plant':16s}" + "".join(f"{f'{d:g} min':>10s}" for d in SWEEP_DEADTIMES)
        + "   closed-loop states")
    swept = {}
    for network, share in SWEEP_PLANTS:
        sel = [r for r in rows if r[0] == network and r[1] == share]
        label = network if share is None else f"{network} share {share:g}"
        swept[label] = np.array([r[3] for r in sel])
        say(f"  {label:16s}" + "".join(f"{r[3]:10.4f}" for r in sel)
            + f"   {sel[0][5]} to {sel[-1][5]}")
    say("  (the rightmost eigenvalue moves from "
        f"{min(r[4] for r in rows):.6f} to {max(r[4] for r in rows):.6f} 1/min over the "
        "whole sweep, so\n   what the deadtime changes is damping rather than stability)")
    store["sweep_deadtime"] = np.array(SWEEP_DEADTIMES)
    for label, z in swept.items():
        store[f"sweep_zeta_{label}"] = z


# ------------------------------------------------------------------------------------
# 2.  Sensitivity over the design variables
# ------------------------------------------------------------------------------------

# The two rankings compared are computed over one input set, so that they
# can be compared: the design variables, over the box the vacuum family samples.  The
# damping ratio costs a design solve and is computed here; the realized outcome costs a
# closed-loop run each and is generated separately, from the sample written below.
SOBOL_BOUNDS = {
    "pump-around share": (0.0, 0.7),
    "column pressure, bar": (0.15, 0.275),
    "reactor T loop Kc": (-4.0, -0.25),
    "reactor T loop tau_I, min": (5.0, 160.0),
    "analyzer deadtime, min": (0.0, 10.0),
}
SOBOL_DATA = "data/design-sobol"
# The upset the sample is run under, as the specifications write it: the one the analytical
# method separated the designs on.  Holding it fixed keeps the sample a sample of designs.
SOBOL_DISTURBANCE = {"name": "throughput", "fraction": 0.10, "t": 30.0}
SOBOL_UPSET = "throughput +10%"


def _sobol_config(row) -> dict:
    share, pressure, Kc, tau_I, deadtime = (float(v) for v in row)
    return {"structure": "effluent fixed + cascade",
            "options": {"network": "D4", "pump_around_share": share},
            "params": {"column.P_network": pressure},
            "tuning": {"reactor T": {"Kc": Kc, "tau_I": tau_I}},
            "instruments": {"reactor C_A": {"order": 2, "deadtime": deadtime}},
            "disturbance": dict(SOBOL_DISTURBANCE)}


def _sobol_zeta(row) -> float:
    import plantbench as pb
    from plantbench.core import control as ctl
    setup = pb.build(dataset.CASE, dataset.CASE.config(**_sobol_config(row)))
    return ctl.damping_ratio(setup.spectrum())


def section_sensitivity(store, out_dir: Path, n: int = 256, workers: int = 1):
    heading("2.  SENSITIVITY OVER THE DESIGN VARIABLES")
    from plantbench import sensitivity
    names = list(SOBOL_BOUNDS)
    bounds = [SOBOL_BOUNDS[k] for k in names]
    base = sensitivity.base_size(n)
    say(f"Sobol' indices over the vacuum family's box, n = {base}, so "
        f"{base * (len(names) + 2)} evaluations of each\noutput.  The upset is held at "
        f"{SOBOL_UPSET}, so that the sample varies the design and not what it is\nasked "
        "to reject.")

    def zeta(sample):
        if workers <= 1:
            return np.array([_sobol_zeta(r) for r in sample])
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as pool:
            return np.array(list(pool.map(_sobol_zeta, list(sample), chunksize=8)))

    say("\n(a) the damping ratio at the design point, the analytical method's criterion;\n"
        "the analyzer deadtime enters the linearization as its second-order Pade "
        "approximation (section 1)")
    y_zeta = zeta(sensitivity.sample_points(bounds, n=n, seed=0))
    idx = sensitivity.indices_from(y_zeta, n=n, k=len(names), names=names)
    _print_indices(idx, y_zeta, n)

    say("\n(b) the realized outcome, as the indicator that a run does not hold an "
        "operating point")
    written = _sobol_sample_file(bounds, n, out_dir)
    say(f"  {written} configurations written to sobol-outcome.json.  Run them with:")
    say("    python -m studies.design.verify --workers 8")
    _sobol_outcome(bounds, names, n, store)
    store.update(sobol_zeta_first=idx.first, sobol_zeta_total=idx.total,
                 sobol_names=np.array(names))


def _print_indices(idx, y: np.ndarray, n: int, resamples: int = 1000) -> None:
    """The indices in decreasing total, each with its 95% bootstrap interval."""
    from plantbench import sensitivity
    first, total = sensitivity.bootstrap_intervals(y, n=n, k=len(idx.names),
                                                   resamples=resamples, seed=0)
    say(f"  {'design variable':28s}{'first order':>13s}{'total':>9s}"
        f"    95% bootstrap intervals ({resamples} resamples), first; total")
    for name, f, t in idx.ranked():
        i = idx.names.index(name)
        say(f"  {name:28s}{f:13.3f}{t:9.3f}    [{first[i, 0]:.3f}, {first[i, 1]:.3f}]; "
            f"[{total[i, 0]:.3f}, {total[i, 1]:.3f}]")


def _sobol_sample_file(bounds, n: int, out_dir: Path) -> int:
    """Write the configurations the outcome's Sobol' sample needs, in its own order."""
    from plantbench import sensitivity
    rows = sensitivity.sample_points(bounds, n=n, seed=0)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sobol-outcome.json").write_text(json.dumps(
        {"case": "reactor_separator_recycle", "upset": SOBOL_UPSET, "n": n,
         "configs": [_sobol_config(r) for r in rows]}, indent=1))
    return len(rows)


def _sobol_outcome(bounds, names, n: int, store):
    """The outcome's indices, once the sample of `SOBOL_DATA` has been run."""
    from plantbench import sensitivity
    if not (Path(SOBOL_DATA) / "runs.jsonl").exists():
        say(f"  {SOBOL_DATA} does not exist yet, so the second ranking is not computed here")
        return
    rows = sensitivity.sample_points(bounds, n=n, seed=0)
    sample = dataset.load(SOBOL_DATA, workers=1)
    verified = {r.run_id: r for r in sample}
    wall = np.array([r.wall_time for r in sample])
    say(f"  {SOBOL_DATA}: {len(sample)} runs, {wall.sum() / 3600:.2f} h of integration, "
        f"median {np.median(wall):.1f} s, max {wall.max():.0f} s")
    from plantbench import datagen
    y = []
    for row in rows:
        key = datagen.run_id(dataset.CASE, _sobol_config(row))
        if key not in verified:
            say(f"  {SOBOL_DATA} is missing {key}; the sample is incomplete")
            return
        y.append(0.0 if verified[key].outcome in RELIABLE else 1.0)
    idx = sensitivity.indices_from(np.array(y), n=n, k=len(names), names=names)
    _print_indices(idx, np.array(y), n)
    store.update(sobol_outcome_first=idx.first, sobol_outcome_total=idx.total)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=1,
                        help="processes used for the design solves")
    parser.add_argument("--out", type=Path, default=HERE / "results")
    parser.add_argument("--sobol-n", type=int, default=256,
                        help="base sample size of the sensitivity analysis")
    args = parser.parse_args(argv)
    from plantbench import datasets
    stamp = datasets.stamp(SOBOL_DATA) if Path(SOBOL_DATA).exists() else None
    say("The damping ratio of reactor_separator_recycle against the analyzer deadtime, and its Sobol' indices")
    if stamp is not None:
        for line in datasets.stamp_lines(stamp):
            say(line)
    store: dict = {}
    args.out.mkdir(parents=True, exist_ok=True)
    section_deadtime(store, args.workers)
    section_sensitivity(store, args.out, n=args.sobol_n, workers=args.workers)
    np.savez_compressed(args.out / "analysis.npz", **store)
    (args.out / "results_design.txt").write_text(OUT.getvalue())
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
