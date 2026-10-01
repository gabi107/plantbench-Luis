"""Flowsheet and control-structure drawings of `reactor_separator_recycle`.

    flowsheet                the plant, with design stream values
    control_structure        the reference regulatory structure
    control_structure_D3     the same structure on the integrated plant (D3)

All three share one geometry so that they can be compared directly.  The stream values in
the flowsheet come from the design solve, not from typed-in numbers.  Each drawing is
written as PNG and SVG to figures/reactor_separator_recycle.

    python -m studies.reactor_separator_recycle.flowsheets [--print]

With --print the same drawings are made at a 6.5 in double-column width, in the study
style (studies/paper_style.py), with text sized for that width and stream values to two
decimals, and written as PDF and EPS to figures/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib import patheffects as pe

from plantbench.cases.reactor_separator_recycle import hen, plant, streams
from plantbench.cases.reactor_separator_recycle.parameters import PlantParameters
from plantbench.cases.reactor_separator_recycle.scenarios import TEMPERATURE_TRAY

FIG = Path(__file__).resolve().parents[2] / "figures" / "reactor_separator_recycle"

LW = 1.4  # process lines
LW_SIGNAL = 0.9
DASH = (0, (4, 2.5))
FS = 9  # equipment labels
FS_SMALL = 8  # stream values, utilities, legend
FS_TAG = 8  # instrument tags
FS_NOTE = 7  # controller annotations
DEC = 3  # decimals of a stream value
FIGSIZE = (12.0, 6.6)
PAPER = False  # set by --print
R_HX = 0.32  # exchanger circle
R_TAG = 0.27  # instrument bubble
JACKET_GAP = 0.17

# Geometry shared by every drawing.  Units are arbitrary; the axes keep an equal aspect.
X_LEFT = -0.9  # fresh feed enters here
Y_FEED = 6.5  # mixer -> reactor feed line
Y_RECYCLE = 10.4
X_MIX = 0.9
X_FEHE = 2.5  # integrated plant only
X_BYPASS = (1.5, 3.3)  # split and join around the FEHE
Y_BYPASS = 7.3
X_HEATER = 4.4
REACTOR = dict(x0=6.3, y0=3.2, w=1.8, h=2.4)
X_REACTOR_FEED = 6.7
X_EFFLUENT_OUT = 7.2
Y_EFFLUENT_LOW = 1.2
Y_EFFLUENT_HIGH = 8.3  # integrated plant: FEHE outlet to the column
Y_JACKET_IN = REACTOR["y0"] + 0.15
Y_JACKET_OUT = REACTOR["y0"] + 1.6
X_JACKET_SUPPLY = 9.9
X_INSTR_REACTOR = 8.95  # reactor temperature and composition controllers
X_RISER = 10.3  # column feed riser
COLUMN = dict(x=11.7, y0=2.3, y1=8.0, w=0.9)
Y_COLUMN_FEED = 5.15
X_REFLUX_TURN = 12.6
Y_REFLUX = 6.9
Y_REFLUX_IN = 7.6
CONDENSER = (13.3, 9.3)
Y_CW_SUPPLY = 7.95
DRUM = dict(x=14.9, y0=7.3, y1=7.9, w=1.4)
X_SPLIT = 16.5
Y_DISTILLATE = 6.0
REBOILER = (13.1, 1.6)
Y_REBOILER_FEED = 1.0
Y_BOILUP_IN = 2.8
X_STEAM_SUPPLY = 14.25
Y_BOTTOMS = 0.4
X_RIGHT = 18.1


def column_left():
    return COLUMN["x"] - COLUMN["w"] / 2


def column_right():
    return COLUMN["x"] + COLUMN["w"] / 2


def reactor_right():
    return REACTOR["x0"] + REACTOR["w"]


# ----------------------------------------------------------------------------------------
# Symbols
# ----------------------------------------------------------------------------------------


def _arrowhead(ax, pts, lw, head_length, head_width, z):
    (x0, y0), (x1, y1) = pts[-2], pts[-1]
    ax.annotate("", xy=(x1, y1), xytext=(x1 - 0.02 * (x1 - x0), y1 - 0.02 * (y1 - y0)),
                arrowprops=dict(arrowstyle=f"-|>,head_length={head_length},head_width={head_width}",
                                color="black", lw=lw, shrinkA=0, shrinkB=0),
                zorder=z)


def line(ax, pts, arrow=False, halo=False, z=2.0):
    """Process line.  With `halo`, a white casing breaks any process line it crosses."""
    xs, ys = zip(*pts)
    kw = {}
    if halo:
        kw["path_effects"] = [pe.Stroke(linewidth=6, foreground="white"), pe.Normal()]
    ax.plot(xs, ys, color="black", lw=LW, solid_capstyle="butt", zorder=z, **kw)
    if arrow:
        _arrowhead(ax, pts, LW, 0.55, 0.22, z)


def signal(ax, pts, arrow=False):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color="black", lw=LW_SIGNAL, ls=DASH, zorder=4)
    if arrow:
        _arrowhead(ax, pts, LW_SIGNAL, 0.45, 0.18, 4)


def node(ax, x, y):
    ax.add_patch(patches.Circle((x, y), 0.06, color="black", zorder=3))


def exchanger(ax, x, y, vertical_coil=True):
    """Circle with a coil across it; the coil carries the second stream."""
    ax.add_patch(patches.Circle((x, y), R_HX, fc="white", ec="black", lw=LW, zorder=3))
    n, a = 5, 0.13
    offsets = [(i + 1) * 2 * R_HX / (n + 1) - R_HX for i in range(n)]
    wiggle = [a if i % 2 else -a for i in range(n)]
    if vertical_coil:
        pts = [(x, y - R_HX)] + [(x + w, y + o) for o, w in zip(offsets, wiggle)] + [(x, y + R_HX)]
    else:
        pts = [(x - R_HX, y)] + [(x + o, y + w) for o, w in zip(offsets, wiggle)] + [(x + R_HX, y)]
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color="black", lw=1.0, zorder=3.1)


def valve(ax, x, y, vertical=False, actuator="up"):
    """Control valve: a bowtie on the line and a diaphragm actuator.  Returns the point
    where a controller output connects."""
    s, h = 0.17, 0.12
    if vertical:
        body = [(x - h, y + s), (x + h, y + s), (x - h, y - s), (x + h, y - s)]
    else:
        body = [(x - s, y + h), (x - s, y - h), (x + s, y + h), (x + s, y - h)]
    ax.add_patch(patches.Polygon(body, closed=True, fc="white", ec="black", lw=1.1, zorder=3))
    stem, dome = 0.28, 0.14
    dx, dy = {"up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0)}[actuator]
    tip = (x + dx * stem, y + dy * stem)
    ax.plot([x, tip[0]], [y, tip[1]], color="black", lw=1.0, zorder=3)
    theta = {"up": (0, 180), "down": (180, 360), "left": (90, 270), "right": (270, 90)}[actuator]
    ax.add_patch(patches.Wedge(tip, dome, *theta, fc="white", ec="black", lw=1.0, zorder=3))
    return (tip[0] + dx * dome, tip[1] + dy * dome)


def instrument(ax, x, y, tag):
    ax.add_patch(patches.Circle((x, y), R_TAG, fc="white", ec="black", lw=1.0, zorder=5))
    ax.text(x, y, tag, ha="center", va="center", fontsize=FS_TAG, zorder=6)


def text(ax, x, y, s, ha="center", va="center", fontsize=None):
    fontsize = FS if fontsize is None else fontsize
    ax.text(x, y, s, ha=ha, va=va, fontsize=fontsize, zorder=6, linespacing=1.15)


# ----------------------------------------------------------------------------------------
# Plant
# ----------------------------------------------------------------------------------------


def draw_feed_train(ax, integrated: bool):
    """Fresh feed, mixer, feed-effluent exchanger and its bypass (integrated plant only),
    heater, reactor inlet."""
    line(ax, [(X_LEFT, Y_FEED), (X_MIX, Y_FEED)])
    node(ax, X_MIX, Y_FEED)
    if integrated:
        x_split, x_join = X_BYPASS
        line(ax, [(X_MIX, Y_FEED), (X_FEHE - R_HX, Y_FEED)])
        line(ax, [(X_FEHE + R_HX, Y_FEED), (X_HEATER - R_HX, Y_FEED)])
        line(ax, [(x_split, Y_FEED), (x_split, Y_BYPASS), (x_join, Y_BYPASS), (x_join, Y_FEED)])
        node(ax, x_split, Y_FEED)
        node(ax, x_join, Y_FEED)
        exchanger(ax, X_FEHE, Y_FEED)
        text(ax, X_FEHE - 0.42, Y_FEED - 0.5, "FEHE", ha="right")
        text(ax, X_HEATER, Y_FEED - 0.55, "Trim\nheater", va="top")
    else:
        line(ax, [(X_MIX, Y_FEED), (X_HEATER - R_HX, Y_FEED)])
        text(ax, X_HEATER, Y_FEED - 0.55, "Preheater", va="top")
    exchanger(ax, X_HEATER, Y_FEED)
    line(ax, [(X_HEATER + R_HX, Y_FEED), (X_REACTOR_FEED, Y_FEED),
              (X_REACTOR_FEED, REACTOR["y0"] + REACTOR["h"])], arrow=True)
    line(ax, [(X_HEATER, 7.6), (X_HEATER, Y_FEED + R_HX)], arrow=True)
    text(ax, X_HEATER - 0.12, 7.62, "Steam", ha="right", fontsize=FS_SMALL)
    text(ax, X_LEFT, Y_FEED - 0.2, "Fresh feed", ha="left", va="top")


def draw_reactor(ax):
    r, g = REACTOR, JACKET_GAP
    x0, y0, w, h = r["x0"], r["y0"], r["w"], r["h"]
    ax.add_patch(patches.FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=0.25",
                                        fc="white", ec="black", lw=LW, zorder=3))
    # Jacket over the lower part of the wall.
    ax.plot([x0 - g, x0 - g, x0 + w + g, x0 + w + g], [y0 + 1.8, y0 - g, y0 - g, y0 + 1.8],
            color="black", lw=LW, zorder=3)
    # Agitator.
    xa = x0 + w / 2 + 0.25
    ax.plot([xa, xa], [y0 + 0.55, y0 + h + 0.55], color="black", lw=1.0, zorder=3.1)
    ax.plot([xa - 0.3, xa + 0.3], [y0 + 0.55, y0 + 0.55], color="black", lw=2.2, zorder=3.1)
    ax.add_patch(patches.Circle((xa, y0 + h + 0.72), 0.17, fc="white", ec="black", lw=1.0,
                                zorder=3.1))
    text(ax, xa, y0 + h + 0.72, "M", fontsize=FS_NOTE)
    text(ax, x0 + 0.5, y0 + h - 0.45, "CSTR")
    # Jacket coolant: in at the bottom right, out at the top left.
    line(ax, [(X_JACKET_SUPPLY, Y_JACKET_IN), (reactor_right() + g, Y_JACKET_IN)], arrow=True)
    line(ax, [(x0 - g, Y_JACKET_OUT), (x0 - 0.75, Y_JACKET_OUT)], arrow=True)
    # The print size's larger text would reach the column feed riser at the screen position.
    text(ax, (reactor_right() + X_JACKET_SUPPLY) / 2 + (0.0 if PAPER else 0.3),
         Y_JACKET_IN - 0.22, "Jacket coolant", va="top", fontsize=FS_SMALL)


def draw_effluent(ax, integrated: bool):
    """Reactor effluent to the column feed stage, through the FEHE in the integrated plant.
    The FEHE outlet leg crosses the bypass, so it carries a halo."""
    start = [(X_EFFLUENT_OUT, REACTOR["y0"] - JACKET_GAP), (X_EFFLUENT_OUT, Y_EFFLUENT_LOW)]
    column_in = [(X_RISER, Y_COLUMN_FEED), (column_left(), Y_COLUMN_FEED)]
    if not integrated:
        line(ax, start + [(X_RISER, Y_EFFLUENT_LOW)] + column_in, arrow=True, z=2.5)
        return
    line(ax, start + [(X_FEHE, Y_EFFLUENT_LOW), (X_FEHE, Y_FEED - R_HX)], arrow=True, z=2.5)
    line(ax, [(X_FEHE, Y_FEED + R_HX), (X_FEHE, Y_EFFLUENT_HIGH), (X_RISER, Y_EFFLUENT_HIGH)]
         + column_in, arrow=True, halo=True, z=2.5)


def stage_y(stage: int) -> float:
    """Height of a stage (1 = top) on the drawn column."""
    c = COLUMN
    return c["y1"] - 0.45 - (stage - 1) / 29 * (c["y1"] - c["y0"] - 0.9)


def draw_column(ax, pressure, label_y):
    c = COLUMN
    x0 = column_left()
    ax.add_patch(patches.FancyBboxPatch((x0, c["y0"]), c["w"], c["y1"] - c["y0"],
                                        boxstyle="round,pad=0,rounding_size=0.3",
                                        fc="white", ec="black", lw=LW, zorder=3))
    n = 12
    for i in range(n):
        y = c["y0"] + 0.45 + i * (c["y1"] - c["y0"] - 0.9) / (n - 1)
        xs = [x0, x0 + 0.6 * c["w"]] if i % 2 else [x0 + 0.4 * c["w"], x0 + c["w"]]
        ax.plot(xs, [y, y], color="black", lw=0.7, zorder=3.1)
    text(ax, column_right() + 0.2, label_y, f"30 stages\nfeed on stage 15\n{pressure:g} bar", ha="left")


def draw_overhead(ax):
    c, d = COLUMN, DRUM
    xc, yc = CONDENSER
    line(ax, [(c["x"], c["y1"]), (c["x"], yc), (xc - R_HX, yc)])
    exchanger(ax, xc, yc)
    line(ax, [(xc + R_HX, yc), (d["x"], yc), (d["x"], d["y1"])], arrow=True)
    ax.add_patch(patches.FancyBboxPatch((d["x"] - d["w"] / 2, d["y0"]), d["w"], d["y1"] - d["y0"],
                                        boxstyle="round,pad=0,rounding_size=0.28",
                                        fc="white", ec="black", lw=LW, zorder=3))
    text(ax, xc - 0.42, yc + 0.42, "Condenser", ha="right")
    text(ax, d["x"] + d["w"] / 2 + 0.1, d["y1"] + 0.25, "Reflux drum", ha="left")
    # Cooling water through the condenser.
    line(ax, [(xc, Y_CW_SUPPLY), (xc, yc - R_HX)], arrow=True)
    line(ax, [(xc, yc + R_HX), (xc, yc + 0.62)], arrow=True)
    if PAPER:  # below the supply, clear of the pressure valve on the line
        text(ax, xc, Y_CW_SUPPLY - 0.08, "Cooling\nwater", va="top", fontsize=FS_SMALL)
    else:
        text(ax, xc + 0.18, Y_CW_SUPPLY + 0.1, "Cooling\nwater", ha="left", fontsize=FS_SMALL)
    # Reflux.
    xr = d["x"] - 0.4
    line(ax, [(xr, d["y0"]), (xr, Y_REFLUX), (X_REFLUX_TURN, Y_REFLUX), (X_REFLUX_TURN, Y_REFLUX_IN),
              (column_right(), Y_REFLUX_IN)], arrow=True)
    # Distillate to the splitter, then purge and recycle.
    xd = d["x"] + 0.4
    line(ax, [(xd, d["y0"]), (xd, Y_DISTILLATE), (X_SPLIT, Y_DISTILLATE)])
    node(ax, X_SPLIT, Y_DISTILLATE)
    line(ax, [(X_SPLIT, Y_DISTILLATE), (X_RIGHT, Y_DISTILLATE)], arrow=True)
    line(ax, [(X_SPLIT, Y_DISTILLATE), (X_SPLIT, Y_RECYCLE), (X_MIX, Y_RECYCLE),
              (X_MIX, Y_FEED + 0.06)], arrow=True)
    text(ax, X_RIGHT + 0.1, Y_DISTILLATE, "Purge", ha="left")


def draw_bottoms(ax):
    c = COLUMN
    xb, yb = REBOILER
    line(ax, [(c["x"], c["y0"]), (c["x"], Y_BOTTOMS), (X_RIGHT, Y_BOTTOMS)], arrow=True)
    node(ax, c["x"], Y_REBOILER_FEED)
    line(ax, [(c["x"], Y_REBOILER_FEED), (xb, Y_REBOILER_FEED), (xb, yb - R_HX)], arrow=True)
    line(ax, [(xb, yb + R_HX), (xb, Y_BOILUP_IN), (column_right(), Y_BOILUP_IN)], arrow=True)
    exchanger(ax, xb, yb, vertical_coil=False)
    line(ax, [(X_STEAM_SUPPLY, yb), (xb + R_HX, yb)], arrow=True)
    line(ax, [(xb - R_HX, yb), (xb - 0.62, yb)], arrow=True)
    text(ax, X_STEAM_SUPPLY + 0.07, yb, "Steam", ha="left", fontsize=FS_SMALL)
    text(ax, xb + 0.4, yb + 0.45, "Reboiler", ha="left")
    text(ax, X_RIGHT + 0.1, Y_BOTTOMS, "Product B", ha="left")


def draw_plant(ax, pressure, integrated=False, recycle_label="Recycle", column_label_y=3.95):
    draw_feed_train(ax, integrated)
    draw_reactor(ax)
    draw_effluent(ax, integrated)
    draw_column(ax, pressure, column_label_y)
    draw_overhead(ax)
    draw_bottoms(ax)
    text(ax, 8.5, Y_RECYCLE + 0.2, recycle_label, va="bottom")


# ----------------------------------------------------------------------------------------
# Stream values and control structure
# ----------------------------------------------------------------------------------------


def stream_values(ax, op, pp):
    rp = pp.reactor
    r = REACTOR
    reflux = op.V_boil - op.D  # total condenser, constant molar overflow
    small = dict(fontsize=FS_SMALL)
    text(ax, X_LEFT, Y_FEED - 0.55, f"{op.F_fresh:.{DEC}f} kmol/min\n{pp.thermo.T_storage:.0f} K\n"
         f"A + {100 * pp.z_inert:.1f} mol% I", ha="left", va="top", **small)
    text(ax, (X_HEATER + X_REACTOR_FEED) / 2 + 0.15, Y_FEED + 0.15,
         f"{op.F_col:.{DEC}f} kmol/min\n{rp.T0:.2f} K", va="bottom", **small)
    text(ax, r["x0"] - JACKET_GAP - 0.15, r["y0"] + 0.7, f"{op.T_reactor:.2f} K\n{rp.V:.2f} m³",
         ha="right", **small)
    text(ax, (reactor_right() + X_JACKET_SUPPLY) / 2 + (0.0 if PAPER else 0.3),
         Y_JACKET_IN - 0.55, f"{rp.Tj0:.1f} K",
         va="top", **small)
    text(ax, (X_EFFLUENT_OUT + X_RISER) / 2, Y_EFFLUENT_LOW + 0.15, f"{op.F_col:.{DEC}f} kmol/min",
         va="bottom", **small)
    text(ax, column_right() + 0.15, Y_BOILUP_IN + 0.2, f"Boil-up\n{op.V_boil:.{DEC}f} kmol/min",
         ha="left", va="bottom", **small)
    text(ax, X_REFLUX_TURN + 0.15, Y_REFLUX - 0.15, f"Reflux\n{reflux:.{DEC}f} kmol/min",
         ha="left", va="top", **small)
    text(ax, DRUM["x"] + 1.05, Y_DISTILLATE - 0.15, f"Distillate\n{op.D:.{DEC}f} kmol/min",
         va="top", **small)
    text(ax, X_RIGHT + 0.1, Y_DISTILLATE - 0.3, f"{op.purge:.{DEC}f} kmol/min", ha="left", va="top",
         **small)
    text(ax, X_RIGHT + 0.1, Y_BOTTOMS + 0.3, f"{op.Bm:.{DEC}f} kmol/min", ha="left", va="bottom",
         **small)


def controls(ax, integrated=False):
    r, c = REACTOR, COLUMN

    # Reactor level on the fresh feed, measured at the top of the vessel.
    tip = valve(ax, 0.05, Y_FEED)
    x_lc, y_lc = reactor_right() - 0.3, 8.95
    instrument(ax, x_lc, y_lc, "LC")
    signal(ax, [(x_lc, r["y0"] + r["h"]), (x_lc, y_lc - R_TAG)])
    signal(ax, [(x_lc - R_TAG, y_lc), (0.05, y_lc), tip], arrow=True)

    # Reactor inlet temperature on the heater steam, in split range with the FEHE bypass
    # in the integrated plant.
    tip = valve(ax, X_HEATER, 7.15, vertical=True, actuator="right")
    x_tc, y_tc = 5.55, 7.35
    instrument(ax, x_tc, y_tc, "TC")
    signal(ax, [(x_tc, y_tc - R_TAG), (x_tc, Y_FEED)])
    signal(ax, [(x_tc - R_TAG, y_tc - 0.1), tip], arrow=True)
    if integrated:
        x_bypass_valve = 1.95
        tip = valve(ax, x_bypass_valve, Y_BYPASS)
        signal(ax, [(x_tc, y_tc + R_TAG), (x_tc, 7.95), (x_bypass_valve, 7.95), tip], arrow=True)
        text(ax, x_tc + R_TAG + 0.06, y_tc + 0.12, "split\nrange", ha="left", va="bottom",
             fontsize=FS_NOTE)

    # Reactor temperature on the jacket coolant; the composition controller sets its
    # set-point.
    x = X_INSTR_REACTOR
    y_t, y_c = 4.55, 5.75
    tip = valve(ax, x, Y_JACKET_IN)
    instrument(ax, x, y_t, "TC")
    signal(ax, [(x - R_TAG, y_t), (reactor_right(), y_t)])
    signal(ax, [(x, y_t - R_TAG), tip], arrow=True)
    instrument(ax, x, y_c, "CC")
    signal(ax, [(x - R_TAG, y_c), (reactor_right(), y_c)])
    signal(ax, [(x, y_c - R_TAG), (x, y_t + R_TAG)], arrow=True)

    # The effluent flow is fixed and sets the production rate.
    y_f = 2.25
    x_v = 4.9 if integrated else 9.4
    tip = valve(ax, x_v, Y_EFFLUENT_LOW)
    instrument(ax, x_v, y_f, "FC")
    signal(ax, [(x_v, y_f - R_TAG), tip], arrow=True)
    text(ax, x_v - R_TAG - 0.08, y_f, "production\nrate", ha="right", fontsize=FS_NOTE)

    # Column pressure on the condenser cooling water.
    xc, _ = CONDENSER
    y_p = 8.45
    tip = valve(ax, xc, y_p, vertical=True, actuator="left")
    x_p = 12.4
    instrument(ax, x_p, y_p, "PC")
    signal(ax, [(x_p - R_TAG, y_p), (c["x"], y_p)])
    signal(ax, [(x_p + R_TAG, y_p), tip], arrow=True)

    # Reflux on flow control.
    x_r = 13.65
    tip = valve(ax, x_r, Y_REFLUX, actuator="down")
    instrument(ax, x_r, 6.0, "FFC")
    signal(ax, [(x_r, 6.0 + R_TAG), tip], arrow=True)
    text(ax, x_r + R_TAG + 0.08, 6.0, "ratio to\ncolumn feed", ha="left", fontsize=FS_NOTE)

    # Reflux drum level on the distillate.
    xd = DRUM["x"] + 0.4
    tip = valve(ax, xd, 6.5, vertical=True, actuator="right")
    x_l = 16.05
    instrument(ax, x_l, 7.6, "LC")
    signal(ax, [(x_l - R_TAG, 7.6), (DRUM["x"] + DRUM["w"] / 2, 7.6)])
    signal(ax, [(x_l, 7.6 - R_TAG), (x_l, 6.5), tip], arrow=True)

    # Purge on flow control.
    x_purge = 17.3
    tip = valve(ax, x_purge, Y_DISTILLATE)
    instrument(ax, x_purge, 6.85, "FC")
    signal(ax, [(x_purge, 6.85 - R_TAG), tip], arrow=True)

    # Column base level on the bottoms.
    tip = valve(ax, c["x"], 0.7, vertical=True, actuator="left")
    x_b = 10.8
    instrument(ax, x_b, 2.6, "LC")
    signal(ax, [(x_b + R_TAG, 2.6), (column_left(), 2.6)])
    signal(ax, [(x_b, 2.6 - R_TAG), (x_b, 0.7), tip], arrow=True)

    # Stripping-tray temperature on the reboiler steam.
    _, yb = REBOILER
    x_sv = 13.8
    tip = valve(ax, x_sv, yb, actuator="down")
    x_t, y_t = column_right() + 0.4, stage_y(TEMPERATURE_TRAY)
    instrument(ax, x_t, y_t, "TC")
    text(ax, x_t + R_TAG + 0.08, y_t + 0.12, f"tray {TEMPERATURE_TRAY}", ha="left", va="bottom", fontsize=FS_NOTE)
    signal(ax, [(x_t - R_TAG, y_t), (column_right(), y_t)])
    x_route = 15.5
    signal(ax, [(x_t + R_TAG, y_t), (x_route, y_t), (x_route, 0.95), (x_sv, 0.95), tip], arrow=True)

    legend(ax)


def legend(ax):
    y = 0.5
    text(ax, X_LEFT, y, "FC flow   FFC flow ratio   LC level   PC pressure   TC temperature   CC composition",
         ha="left", fontsize=FS_SMALL)
    signal(ax, [(X_LEFT, y - 0.4), (X_LEFT + 0.8, y - 0.4)])
    text(ax, X_LEFT + 0.95, y - 0.4, "controller signal", ha="left", fontsize=FS_SMALL)


# ----------------------------------------------------------------------------------------


def new_axes():
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_xlim(-1.1, 19.4)
    ax.set_ylim(-0.1, 11.1)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def save(fig, name):
    if PAPER:
        from studies import paper_style as ps
        ps.FIGURES.mkdir(parents=True, exist_ok=True)
        for ext in ("pdf", "eps"):
            fig.savefig(ps.FIGURES / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02,
                        metadata={"CreationDate": None} if ext == "pdf" else None)
        plt.close(fig)
        print("wrote", name)
        return
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print("wrote", name)


def _paper() -> None:
    """Text and lines sized for 6.5 in, the width the drawings are printed at."""
    global PAPER, FS, FS_SMALL, FS_TAG, FS_NOTE, LW, LW_SIGNAL, DEC, FIGSIZE
    from studies import paper_style as ps
    ps.apply()
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=["black"])  # every line solid
    PAPER = True
    FS, FS_SMALL, FS_TAG, FS_NOTE = 6.5, 5.5, 5.0, 5.0
    LW, LW_SIGNAL = 0.9, 0.6
    DEC = 2
    FIGSIZE = (ps.WIDTH["double"], ps.WIDTH["double"] * 6.6 / 12.0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--print", action="store_true", dest="print_",
                        help="draw at print width as PDF and EPS into figures/")
    if parser.parse_args(argv).print_:
        _paper()
    pp = PlantParameters()
    design = plant.solve_design(pp, V_boil=28.0)
    op = streams.design_point(design)

    P_base = pp.column.P_base
    fig, ax = new_axes()
    # Without the tray temperature controller the column label can sit higher, clear of
    # the boil-up value.
    draw_plant(ax, P_base, recycle_label=f"Recycle {op.recycle:.{DEC}f} kmol/min", column_label_y=4.6)
    stream_values(ax, op, pp)
    save(fig, "flowsheet")

    fig, ax = new_axes()
    draw_plant(ax, P_base)
    controls(ax)
    save(fig, "control_structure")

    fig, ax = new_axes()
    draw_plant(ax, hen.DESIGNS["D3"].pressure, integrated=True)
    controls(ax, integrated=True)
    save(fig, "control_structure_D3")


if __name__ == "__main__":
    main()
