"""The figures of the ACC 2027 paper, drawn at the conference's column widths.

    python -m studies.acc_figures --designs-cache DIR [--out DIR]

In the style of studies/paper_style.py at the ACC venue, written to figures/acc.  The
control structure is drawn by studies/reactor_separator_recycle/flowsheets.py, the closed-loop responses from
the designs.pkl cache that studies/reactor_separator_recycle/designs.py writes to its --cache directory, and the
damping-ratio panel from studies/regimes/results/analysis.npz.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from studies import paper_style as ps

ROOT = Path(__file__).resolve().parents[1]

# One line style per design, so that the responses read in grayscale.
CONTROL_STYLE = {"D1": dict(lw=2.4, ls="-"), "D3": dict(lw=1.3, ls="--"),
                 "D5": dict(lw=1.0, ls=(0, (1, 1.2))), "D4": dict(lw=0.7, ls="-")}


def _control_line(ax, t, y, name):
    ax.plot(t, y, color=ps.DESIGN_COLOR[name], label=name, **CONTROL_STYLE[name])


def fig_control_structure() -> None:
    from plantbench.cases.reactor_separator_recycle.parameters import PlantParameters
    from studies.reactor_separator_recycle import flowsheets as fs

    fs._paper()  # applies the style at the venue already set, then the drawing's own sizes
    scale = ps.WIDTH["double"] / ps.VENUES["default"]["WIDTH"]["double"]
    fs.FS, fs.FS_SMALL, fs.FS_TAG, fs.FS_NOTE = (round(s * scale, 1)
                                                for s in (fs.FS, fs.FS_SMALL, fs.FS_TAG, fs.FS_NOTE))
    fig, ax = fs.new_axes()
    fs.draw_plant(ax, PlantParameters().column.P_base)
    fs.controls(ax)
    fs.save(fig, "control_structure")


def fig_controllability(ex) -> None:
    fig, axes = ps.figure("single", 3.0, nrows=2, ncols=2)
    zoom = 60.0
    order = ("D1", "D3", "D5", "D4")
    for name in order:
        run = ex.controllability[name]["runs"]["throughput +10%"]
        t = run["t"]
        late = t >= t[-1] - zoom
        _control_line(axes[0, 0], t, run["T"], name)
        _control_line(axes[0, 1], t, run["Fj"], name)
        _control_line(axes[1, 0], t[late], run["T"][late], name)
        _control_line(axes[1, 1], t[late], run["Fj"][late], name)
    labels = ("Reactor temperature (K)", r"Jacket flow (m$^3$ min$^{-1}$)") * 2
    for ax, lab, letter in zip(axes.ravel(), labels, "abcd"):
        ax.set_ylabel(lab)
        ax.set_xlabel("Time (min)")
        ps.panel(ax, letter)
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="outside upper center", ncol=4)
    ps.save(fig, "controllability")


def fig_damping(results: Path, data: Path) -> None:
    from studies.regimes import figures as rf

    d = np.load(results / "analysis.npz", allow_pickle=False)
    rf.check(d, data)
    fig, ax = ps.figure("single", 1.9)
    outcomes = d["run_outcome"]
    for o in ps.OUTCOME_ORDER:
        m = outcomes == o
        if m.any():
            ax.scatter(d["run_damping"][m], d["run_wall"][m], s=5, marker=ps.OUTCOME_MARKER[o],
                       c=ps.OUTCOME_COLOR[o], linewidths=0, rasterized=True)
    ax.set_yscale("log")
    ax.set_xlabel(r"Least damping ratio $\zeta$ at the design point")
    ax.set_ylabel("Wall time of the run (s)")
    ax.legend(handles=rf._legend_handles(set(outcomes)), loc="upper right", handletextpad=0.3)
    ps.save(fig, "damping")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--designs-cache", type=Path, required=True,
                        help="the --cache directory of studies.reactor_separator_recycle.designs, holding designs.pkl")
    parser.add_argument("--results", type=Path, default=ROOT / "studies/regimes/results")
    parser.add_argument("--data", type=Path, default=Path("data/regimes"))
    parser.add_argument("--out", type=Path, help="directory for the figures; figures/acc by default")
    args = parser.parse_args(argv)
    ps.use("acc")
    if args.out is not None:
        ps.FIGURES = args.out
    ps.apply()
    ex = pickle.loads((args.designs_cache / "designs.pkl").read_bytes())
    fig_controllability(ex)
    fig_damping(args.results, args.data)
    fig_control_structure()
    print(f"figures in {ps.FIGURES}")


if __name__ == "__main__":
    main()
