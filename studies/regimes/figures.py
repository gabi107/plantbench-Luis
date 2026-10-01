"""Figures of the regimes study, drawn from results/analysis.npz in the style of studies/paper_style.py.

    .venv/bin/python -m studies.regimes.figures [--results DIR] [--out DIR] [--data DIR]

Figures carry no titles; captions name them.  Color and marker encode
the late outcome together (studies/paper_style.py), and the variables are named as the
text names them rather than by their identifiers in the case.  Written as PDF and EPS to
figures/.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D

from studies import paper_style as ps

HERE = Path(__file__).resolve().parent
METHODS = ("PCA", "t-SNE", "PaCMAP")

# The signals of dataset.VARIABLES as the figures name them.
NAMES = {
    "reactor T": "reactor temperature", "reactor C_A": r"reactor $\mathit{C}_\mathrm{A}$",
    "reactor V": "reactor holdup", "drum level": "reflux drum level",
    "base level": "column base level", "bottoms x_A": "A in the bottoms",
    "tray 17 T": "tray 17 temperature", "recycle": "recycle flow",
    "feed T": "reactor inlet temperature", "Fj": "jacket coolant flow",
    "F_fresh": "fresh feed flow", "V_boil": "boil-up", "L_reflux": "reflux flow",
    "D": "distillate flow", "Bm": "bottoms flow", "Q_feed_trim": "feed trim duty",
}
SIZE = {"settled": 1.0, "unsettled": 2.0, "limit cycle": 2.5}  # rare classes drawn larger


def _legend_handles(present):
    return [Line2D([], [], marker=ps.OUTCOME_MARKER[o], linestyle="", markersize=3.5,
                   markeredgewidth=0, color=ps.OUTCOME_COLOR[o], label=o)
            for o in ps.OUTCOME_ORDER if o in present]


def _scatter(ax, E, outcomes, scale=1.0):
    """One layer per outcome, rare on top, rasterized inside the vector file."""
    for o in ps.OUTCOME_ORDER:
        m = outcomes == o
        if m.any():
            ax.scatter(E[m, 0], E[m, 1], s=SIZE[o] * scale, marker=ps.OUTCOME_MARKER[o],
                       c=ps.OUTCOME_COLOR[o], linewidths=0, rasterized=True)


def _blank(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("left", "bottom"):
        ax.spines[side].set_color(ps.GRID)


def _caption(ax, text):
    """A short name for the panel, beside its letter above the axes, clear of the data."""
    ax.text(0.02, 1.02, text, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=ps.SMALL, color=ps.INK2)


def fig_embeddings(d, out: Path):
    fig, axes = ps.figure("double", 4.2, nrows=2, ncols=3)
    for row, rep in enumerate(("global", "local")):
        outcomes = d[f"{rep}_outcome"]
        for col, method in enumerate(METHODS):
            ax = axes[row, col]
            _scatter(ax, d[f"{rep}_{method}"], outcomes)
            _blank(ax)
            _caption(ax, f"{method}, {rep} representation")
    for ax, letter in zip(axes.ravel(), "abcdef"):
        ps.panel(ax, letter)
    fig.legend(handles=_legend_handles(set(d["global_outcome"])), loc="outside upper center",
               ncol=3, handletextpad=0.3, columnspacing=1.5)
    ps.save(fig, "r1_late_embeddings", out)


def _dominant_cluster(d, rep: str, outcome: str):
    """The cluster, among those whose majority is `outcome` and that SGS compared, holding
    the most samples of that outcome."""
    labels, outcomes = d[f"{rep}_PaCMAP_clusters"], d[f"{rep}_outcome"]
    best, best_n = None, 0
    for key in d.files:
        if not key.startswith(f"{rep}_sgs_") or key.endswith(("_label", "_outcome")):
            continue
        if str(d[f"{key}_outcome"]) != outcome:
            continue
        c = int(key.rsplit("_", 1)[1])
        n = int(np.sum((labels == c) & (outcomes == outcome)))
        if n > best_n:
            best, best_n = key, n
    return best


def _describe(label: str, rep: str) -> str:
    """What the compared cluster holds, from the label the analysis stored."""
    m = re.fullmatch(r"cluster \d+ \(([^,]+), (.+)\) against cluster \d+", label)
    if m is None:
        return rep.capitalize()
    outcome, upset = m.groups()
    return f"{rep.capitalize()}: {outcome.replace(' ', '-')} cluster, {upset}"


def fig_sgs(d, out: Path, n_top: int = 5):
    """SGS contributions for the cluster holding the most limit-cycle samples in each
    representation, and for the local cluster holding the most unsettled samples."""
    variables = list(d["variables"])
    chosen = [(_dominant_cluster(d, "global", "limit cycle"), "global"),
              (_dominant_cluster(d, "local", "limit cycle"), "local"),
              (_dominant_cluster(d, "local", "unsettled"), "local")]
    chosen = [(k, rep) for k, rep in chosen if k is not None]
    if not chosen:
        return
    fig, axes = ps.figure("single", 1.25 * len(chosen) + 0.3, nrows=len(chosen), sharex=True,
                          squeeze=False)
    for ax, (k, rep), letter in zip(axes[:, 0], chosen, "abc"):
        v = np.nan_to_num(d[k])
        order = [i for i in np.argsort(-v)[:n_top] if v[i] > 0][::-1]
        outcome = str(d[f"{k}_outcome"])
        rows = range(n_top - len(order), n_top)  # from the top, so bars keep one height
        ax.barh(rows, v[order], height=0.6, color=ps.OUTCOME_COLOR[outcome])
        ax.set_yticks(rows, [NAMES.get(variables[i], variables[i]) for i in order])
        ax.set_ylim(-0.5, n_top - 0.5)
        ax.tick_params(axis="y", length=0)
        for i, j in zip(rows, order):
            ax.text(v[j] + 1.5, i, f"{v[j]:.0f}", va="center", color=ps.INK2,
                    fontsize=ps.SMALL)
        ax.set_xlim(0, 100.0)
        _caption(ax, _describe(str(d[f"{k}_label"]), rep))
        ps.panel(ax, letter)
    axes[-1, 0].set_xlabel("Contribution (%)")
    ps.save(fig, "r2_sgs", out)


def fig_cost_and_early(d, out: Path):
    fig, axes = ps.figure("double", 2.5, ncols=2)
    ax = axes[0]
    outcomes = d["run_outcome"]
    for o in ps.OUTCOME_ORDER:
        m = outcomes == o
        if m.any():
            ax.scatter(d["run_damping"][m], d["run_wall"][m], s=4, marker=ps.OUTCOME_MARKER[o],
                       c=ps.OUTCOME_COLOR[o], linewidths=0, rasterized=True)
    ax.set_yscale("log")
    ax.set_xlabel("Least damping ratio of the linearized closed loop")
    ax.set_ylabel("Wall time of the run (s)")
    ax.legend(handles=_legend_handles(set(outcomes)), loc="upper right", handletextpad=0.3)
    ps.panel(ax, "a")

    ax = axes[1]
    _scatter(ax, d["early_PaCMAP"], d["early_outcome"], scale=3.0)
    _blank(ax)
    _caption(ax, "PaCMAP, early window")
    ps.panel(ax, "b")
    ps.save(fig, "r3_cost_and_early", out)


def check(d, data_dir: Path) -> None:
    """Refuse to draw one dataset's figures from another dataset's analysis."""
    from plantbench import datasets
    if not (data_dir / "manifest.json").exists():
        print(f"{data_dir}: not on disk, so the analysis is drawn unchecked")
        return
    if "provenance" not in d:
        print("the analysis carries no stamp, so it is drawn unchecked")
        return
    why = datasets.check_stamp(json.loads(str(d["provenance"])), data_dir)
    if why is not None:
        raise SystemExit(f"analysis.npz was {why}\nregenerate it with "
                         f".venv/bin/python -m studies.regimes.analysis {data_dir}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=HERE / "results")
    parser.add_argument("--out", type=Path, default=ps.FIGURES)
    parser.add_argument("--data", type=Path, default=Path("data/regimes"),
                        help="the dataset the analysis should be of")
    args = parser.parse_args(argv)
    ps.apply()
    d = np.load(args.results / "analysis.npz", allow_pickle=False)
    check(d, args.data)
    fig_embeddings(d, args.out)
    fig_sgs(d, args.out)
    fig_cost_and_early(d, args.out)
    print(f"figures in {args.out}")


if __name__ == "__main__":
    main()
