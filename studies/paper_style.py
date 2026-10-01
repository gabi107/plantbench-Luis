"""The style of the study figures, drawn at the width they are printed at.

    from studies import paper_style as ps
    ps.apply()
    fig, ax = ps.figure("single", height=2.2)
    ...
    ps.save(fig, "r3_cost_and_early")

The default venue prints a single column at 3.33 in and a double column at 6.5 in, with
lettering no smaller than 4.5 pt and lines no thinner than 0.5 pt in print, in Helvetica
or Arial, and with color never carrying a distinction alone, at a contrast of 3:1 for
marks and 4.5:1 for text.  Every figure is therefore drawn at the width it is
printed at, so that the sizes set here are the sizes printed, and saved as a vector PDF
with its fonts embedded as TrueType, beside an EPS copy for the submission system.

Color does two jobs, each with its own slots so that neither is read as the other: a
design (D1, D3, D4, D5, and the D2 target) and a late outcome (settled, unsettled, limit
cycle).  Both sets were checked with the dataviz validator for all-pairs separation under
the three color-vision deficiencies and for 3:1 contrast on white; settled is a neutral
gray on purpose, the reference the other two outcomes are read against.  Every design
also has its own marker and line style, and every outcome its own marker, so that each
figure reads in grayscale.

Mathematical text is upright by default, so that numbers and units in a label are set
upright; a variable is written as $\\mathit{C}_\\mathrm{A}$.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGURES = Path(__file__).resolve().parents[1] / "figures"

# in; 6.5 is a letter page's text width at 1 in margins.
WIDTH = {"single": 3.33, "double": 6.5}
FONT = 7.0  # pt, labels and ticks; 4.5 pt is the floor
SMALL = 6.0  # pt, legends and annotations
LINE = 1.0  # pt, data lines; 0.5 pt is the floor
THIN = 0.6  # pt, axes, ticks and reference lines

INK, INK2, MUTED = "#0b0b0b", "#3d3c39", "#6b6a65"  # 19.8, 10.9 and 5.4:1 on white
GRID = "#d9d8d2"  # recessive, never carries data

# Designs: validated all-pairs (worst CVD delta E 9.1, normal-vision 16.3, all >= 3:1).
DESIGN_COLOR = {"D1": "#2a78d6", "D3": "#eb6834", "D4": "#4a3aa7", "D5": "#1b9e77",
                "D2": INK}
DESIGN_MARKER = {"D1": "o", "D3": "s", "D4": "^", "D5": "D", "D2": "*"}
DESIGN_LINE = {"D1": "-", "D3": "--", "D4": "-.", "D5": ":", "D2": "-"}

# Outcomes: validated all-pairs (worst CVD delta E 12.4, normal-vision 16.1, all >= 3:1).
OUTCOME_COLOR = {"settled": "#939390", "unsettled": "#9a6400", "limit cycle": "#b8277a",
                 "saturated": "#939390"}
OUTCOME_MARKER = {"settled": "o", "unsettled": "^", "limit cycle": "s", "saturated": "o"}
OUTCOME_ORDER = ("settled", "unsettled", "limit cycle")  # drawn in this order, rare on top

# Series that are not designs or outcomes (the cases of the `reactor_separator_recycle` figures), in fixed order.
SERIES_COLOR = ("#2a78d6", "#eb6834", "#4a3aa7", "#1b9e77")
SERIES_LINE = ("-", "--", "-.", ":")
SERIES_MARKER = ("o", "s", "^", "D")


# The ACC proceedings (ieeeconf.cls, letter paper): \columnwidth 245.72 pt and \textwidth
# 505.89 pt, the widths the conference paper includes its figures at.
VENUES = {
    "default": dict(WIDTH=WIDTH, FONT=FONT, SMALL=SMALL, FIGURES=FIGURES),
    "acc": dict(WIDTH={"single": 3.40, "double": 7.0}, FONT=8.0, SMALL=7.0,
                FIGURES=FIGURES / "acc"),
}


def use(venue: str) -> None:
    """Set the widths, font sizes and output directory of a venue; call before apply()."""
    globals().update(VENUES[venue])


def apply() -> None:
    """Set matplotlib's defaults to the study style."""
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold", "mathtext.default": "regular",
        "font.size": FONT, "axes.labelsize": FONT, "xtick.labelsize": FONT,
        "ytick.labelsize": FONT, "legend.fontsize": SMALL, "axes.titlesize": FONT,
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.edgecolor": INK2, "axes.linewidth": THIN,
        "xtick.major.width": THIN, "ytick.major.width": THIN,
        "xtick.minor.width": THIN * 0.8, "ytick.minor.width": THIN * 0.8,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.direction": "out", "ytick.direction": "out",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "grid.color": GRID, "grid.linewidth": THIN * 0.8,
        "axes.formatter.useoffset": False, "axes.formatter.limits": (-4, 5),
        "axes.prop_cycle": plt.cycler(color=SERIES_COLOR, linestyle=SERIES_LINE),
        "lines.linewidth": LINE, "lines.markersize": 3.5, "patch.linewidth": THIN,
        "legend.frameon": False, "legend.handlelength": 2.2, "legend.borderaxespad": 0.3,
        "figure.dpi": 150, "savefig.dpi": 600, "savefig.bbox": "standard",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def figure(width: str = "single", height: float = 2.2, **subplots):
    """A figure at its printed width, laid out so that nothing is cropped at save time."""
    return plt.subplots(figsize=(WIDTH[width], height), layout="constrained", **subplots)


def panel(ax, letter: str) -> None:
    """The panel letter, bold, above the top-left corner of the axes."""
    ax.text(0.0, 1.02, f"({letter})", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=FONT, fontweight="bold", color=INK)


def save(fig, name: str, out: Path | None = None) -> list[Path]:
    """Write `name`.pdf for the LaTeX build and `name`.eps for the submission system."""
    out = FIGURES if out is None else Path(out)
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / f"{name}.pdf", out / f"{name}.eps"]
    for p in paths:
        fig.savefig(p, metadata=None if p.suffix == ".eps" else {"CreationDate": None})
    plt.close(fig)
    return paths
