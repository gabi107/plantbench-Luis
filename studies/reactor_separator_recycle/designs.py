"""The design comparison of `reactor_separator_recycle`: numbers to results/designs.txt, figures to figures/reactor_separator_recycle.

    python -m studies.reactor_separator_recycle.designs [--cached] [--out DIR] [--cache DIR] [--figures DIR]

Takes several minutes: the throughput points are settled with the dynamic model, and the
controllability comparison simulates each design under two disturbances.  The computed
comparison is cached as designs.pkl in --cache (cache/ by default), which
studies/acc_figures.py draws from; designs.txt is written to --out (results/ by default).
"""

from __future__ import annotations

import argparse
import io
import pathlib
import pickle
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

from plantbench.cases.reactor_separator_recycle import comparison, hen, pareto, streams
from plantbench.cases.reactor_separator_recycle import economics as ec
from plantbench.heat import pinch
from plantbench.units import thermo

FIG = pathlib.Path(__file__).resolve().parents[2] / "figures" / "reactor_separator_recycle"
RESULTS = pathlib.Path(__file__).resolve().parent / "results"
CACHE = pathlib.Path(__file__).resolve().parents[2] / "cache"
GJH = ec.KJ_PER_MIN_TO_GJ_PER_H
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "savefig.bbox": "tight",
                     "axes.formatter.useoffset": False})
# Figures carry no titles: captions name them.
CO2E = r"CO$_2$e"
EXCHANGER_LABELS = {
    "reactor effluent -> reactor feed": "FEHE",
    "reactor effluent -> reboiler": "effluent to reboiler",
    "reactor heat -> reboiler": "pump-around to reboiler",
    "condenser -> reactor feed": "condenser to reactor feed",
}
PRESSURE_TICKS = (0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0)


def pressure_axis(ax):
    ax.set_xscale("log")
    ax.set_xticks(PRESSURE_TICKS)
    ax.set_xticklabels([f"{p:g}" for p in PRESSURE_TICKS])
    ax.minorticks_off()
    ax.set_xlabel("column pressure, bar")


COLORS = {"D1": "#7f7f7f", "D2": "#000000", "D3": "#1f77b4", "D4": "#d62728", "D5": "#2ca02c"}
OUT = io.StringIO()


def say(s=""):
    print(s, flush=True)
    OUT.write(s + "\n")


def header(s):
    say()
    say("=" * 88)
    say(s)
    say("=" * 88)


# ------------------------------------------------------------------------------------
# Report
# ------------------------------------------------------------------------------------


def report(ex: comparison.DesignComparison):
    pp, op = ex.design.pp, ex.op
    ep, tp = pp.economics, pp.thermo

    header("ASSUMED DATA")
    Tb = thermo.normal_boiling_points(tp, pp.column)
    say(f"normal boiling points  inert {Tb[0]:.1f} K   reactant {Tb[1]:.1f} K   product {Tb[2]:.1f} K")
    say(f"molar mass {pp.molar_mass:.1f} kg/kmol   liquid heat capacity {pp.cp_molar:.1f} kJ/(kmol K)")
    say(f"fresh feed delivered at {tp.T_storage:.0f} K; products stored at {tp.T_product_storage:.0f} K; "
        f"dT_min {tp.dT_min:.0f} K")
    say(f"prices  product ${ep.price_product:.2f}/kg   feed ${ep.price_feed:.2f}/kg   "
        f"electricity ${ep.price_electricity:.3f}/kWh   gas ${ep.gas_price:.2f}/GJ")
    say(f"GWP     feed burden {ep.feed_burden:.1f} kg/kg   gas {ep.gas_emission_factor:.1f} kg/GJ fuel   "
        f"boiler efficiency {ep.boiler_efficiency:.2f}   grid {ep.grid_intensity:.2f} kg/kWh   "
        f"chiller COP {ep.chiller_cop:.1f}")
    for u in ec.utilities(ep):
        say(f"   {u.name:14s} {u.T:5.0f} K   ${u.cost:5.2f}/GJ   {u.gwp:6.2f} kg CO2e/GJ")
    xr = ex.design.x[1:4] / pp.rho_molar
    say(f"reactor liquid bubble pressure at {op.T_reactor:.1f} K: "
        f"{thermo.bubble_pressure(xr, op.T_reactor, tp, pp.column):.2f} bar")

    header("TEMPERATURE LEVELS AGAINST COLUMN PRESSURE (design operating point)")
    say(f"{'P bar':>6} {'bottoms':>8} {'drum':>7} {'dew':>7} {'col feed':>9} {'mix':>7}   "
        f"reactor {op.T_reactor:.2f} K, inlet {pp.reactor.T0:.2f} K")
    for r in ex.temperatures:
        say(f"{r['P']:6.2f} {r['T_bottoms']:8.1f} {r['T_drum']:7.1f} {r['T_dew']:7.1f} "
            f"{r['T_column_feed']:9.1f} {r['T_mix']:7.1f}")

    say("\ncolumn sensible heat, supplied by the reboiler beyond lambda*V (GJ/h): " + ", ".join(
        f"{P} bar {streams.column_sensible_heat(op, streams.temperatures(op, P, pp), pp) * GJH:.2f}"
        for P in (0.15, 0.4, 0.8, 1.0)))

    for name in ("D3", "D4"):
        P = hen.DESIGNS[name].pressure
        header(f"STREAM TABLE AT {P} bar ({name})")
        row = next(r for r in ex.targets if r["P"] == P)
        for s in row["streams"]:
            say(f"   {s.name:18s} {'hot ' if s.is_hot else 'cold'} {s.T_supply:7.1f} -> {s.T_target:7.1f} K   "
                f"{s.duty * GJH:6.1f} GJ/h")
        say(f"   pinch (hot side) {row['pinch_hot']:.2f} K   QH_min {row['QH_min'] * GJH:.1f} GJ/h   "
            f"QC_min {row['QC_min'] * GJH:.1f} GJ/h")

    header("UTILITY TARGETS AGAINST COLUMN PRESSURE: D1 no integration, D2 pinch target")
    say(f"{'P':>5} | {'D1 $/h':>7} {'D1 t/h':>7} | {'D2 $/h':>7} {'D2 t/h':>7} {'steam GJ/h':>10} "
        f"{'CW':>6} {'ChW':>6} | {'GWP saving':>10}")
    for r in ex.targets:
        dd, tt = r["direct"], r["target"]
        steam = sum(v for k, v in tt.loads.items() if "steam" in k) * GJH
        say(f"{r['P']:5.2f} | {dd.utility_cost:7.0f} {dd.gwp_energy:7.3f} | {tt.utility_cost:7.0f} "
            f"{tt.gwp_energy:7.3f} {steam:10.1f} {tt.loads['cooling water'] * GJH:6.1f} "
            f"{tt.loads['chilled water'] * GJH:6.1f} | {1 - tt.gwp_energy / dd.gwp_energy:9.0%}")

    header("DESIGNS AT THE DESIGN OPERATING POINT")
    say(f"production {op.Bm * pp.molar_mass * 60 / 1000:.2f} t/h; feed burden "
        f"{ex.ladder['D1']['evaluation'].gwp_feed:.1f} t CO2e/h, the same for every design")
    for name, entry in ex.ladder.items():
        net, r, e, tgt = entry["network"], entry["result"], entry["evaluation"], entry["target"]
        say(f"\n{name}  {net.description}, column at {net.pressure} bar, {len(r.exchangers)} exchanger(s)")
        for x in r.exchangers:
            say(f"   {x.match.hot:17s} -> {x.match.cold:13s} {x.duty * GJH:6.1f} GJ/h  UA {x.UA:9,.0f} kJ/(min K)  "
                f"hot {x.T_hot_in:.1f} -> {x.T_hot_out:.1f}  cold {x.T_cold_in:.1f} -> {x.T_cold_out:.1f}")
        loads = ", ".join(f"{k} {v * GJH:.1f}" for k, v in e.loads.items() if v > 1)
        say(f"   utilities (GJ/h): {loads}")
        say(f"   profit ${e.profit:,.0f}/h   utilities ${e.utility_cost:,.0f}/h   energy GWP {e.gwp_energy:.3f} t/h   "
            f"| pinch target at {net.pressure} bar: ${tgt.utility_cost:,.0f}/h, {tgt.gwp_energy:.3f} t/h")
    e3, e4 = ex.ladder["D3"]["evaluation"], ex.ladder["D4"]["evaluation"]
    say(f"\nD4 against D3: {e3.profit - e4.profit:+,.0f} $/h of profit given up for "
        f"{e3.gwp_energy - e4.gwp_energy:+.3f} t CO2e/h avoided, "
        f"{(e3.profit - e4.profit) / (e3.gwp_energy - e4.gwp_energy):,.0f} $/t")

    header("PARETO EVALUATION ACROSS THROUGHPUT (effluent fixed + cascade; effluent flow moved)")
    base = ex.op.Bm
    for sp in ex.points:
        say(f"   effluent {sp.fraction:+.0%}: production {sp.op.Bm:.3f} kmol/min ({sp.op.Bm / base - 1:+.1%}), "
            f"recycle {sp.op.recycle:.3f}, reactor {sp.op.T_reactor:.2f} K, settling drift {sp.drift:.1e}")
    say(f"\n{'design':6s} {'thru':>6} {'P':>5} {'profit $/h':>11} {'utilities':>9} {'GWP_E t/h':>9} {'feed t/h':>9}")
    for r in ex.pareto:
        if r.design == "D2":
            continue
        e = r.evaluation
        say(f"{r.design:6s} {r.fraction:+6.0%} {r.pressure:5.2f} {e.profit:11,.0f} {e.utility_cost:9.0f} "
            f"{e.gwp_energy:9.3f} {e.gwp_feed:9.1f}")
    d2 = [r for r in ex.pareto if r.design == "D2"]
    keep = pareto.nondominated(np.array([r.evaluation.profit for r in d2]),
                               np.array([r.evaluation.gwp_energy for r in d2]))
    say("\nD2 (pinch target over all pressures), non-dominated points:")
    for r, k in zip(d2, keep):
        if k:
            say(f"   {r.fraction:+.0%} at {r.pressure} bar: profit {r.evaluation.profit:,.0f}  GWP {r.evaluation.gwp_energy:.3f}")
    practical = [r for r in ex.pareto if r.design != "D2"]
    keep = pareto.nondominated(np.array([r.evaluation.profit for r in practical]),
                               np.array([r.evaluation.gwp_energy for r in practical]))
    say("\npractical designs, non-dominated points:")
    for r, k in zip(practical, keep):
        if k:
            say(f"   {r.design} {r.fraction:+.0%}: profit {r.evaluation.profit:,.0f}  GWP {r.evaluation.gwp_energy:.3f}")

    header("GRID CARBON INTENSITY: D4 AGAINST D3 AT DESIGN THROUGHPUT")
    for g, res in ex.grid.items():
        pick = {r.design: r.evaluation for r in res if r.fraction == 0.0 and r.design in ("D3", "D4")}
        dp = pick["D3"].profit - pick["D4"].profit
        dg = pick["D3"].gwp_energy - pick["D4"].gwp_energy
        cost = f"{dp / dg:,.0f} $/t" if dg > 0 else "D4 emits more"
        say(f"   grid {g:.1f} kg/kWh: D3 {pick['D3'].gwp_energy:.3f} t/h, D4 {pick['D4'].gwp_energy:.3f} t/h, "
            f"D4 gives up {dp:,.0f} $/h for {dg:+.3f} t/h  -> {cost}")

    if ex.share:
        header("D4: PUMP-AROUND SHARE AGAINST THE REACTOR TEMPERATURE LOOP (K_c -1, tau_I 20 min)")
        for r in ex.share:
            e = r["evaluation"]
            say(f"   share {r['share']:.1f}: jacket flow {r['Fj']:.3f} m3/min   ${e.utility_cost:,.0f}/h   "
                f"GWP {e.gwp_energy:.3f} t/h   damping {r['damping']:.3f}   rightmost {r['rightmost']:+.4f}")

        header("REACTOR TEMPERATURE LOOP TUNING: damping of the least damped mode")
        for r in ex.tuning:
            say(f"   Kc {r['Kc']:5.2f}  tau_I {r['tau_I']:5.1f}:  base plant {r['base_damping']:+.3f} "
                f"({r['base_rightmost']:+.4f})   D4 {r['D4_damping']:+.3f} ({r['D4_rightmost']:+.4f})")
        stable = [r for r in ex.tuning if r["D4_rightmost"] < 0]
        best = max(stable, key=lambda r: r["D4_damping"])
        say(f"   {len(ex.tuning)} settings, {len(stable)} stable for D4; best D4 damping {best['D4_damping']:.3f} "
            f"at Kc {best['Kc']}, tau_I {best['tau_I']} (base plant {best['base_damping']:.3f} there)")

        header("CONTROLLABILITY, STAGE 2 STRUCTURES")
        for name, entry in ex.controllability.items():
            say(f"{name}: damping {entry['damping']:.3f}  rightmost {entry['rightmost']:+.4f}  "
                f"design jacket flow {entry['Fj_design']:.3f} m3/min")
            for label, run in entry["runs"].items():
                say(f"   {label:17s} IAE C_A {run['IAE_CA']:6.2f} kmol min/m3   IAE inlet T {run['IAE_T_in']:6.2f} K min   "
                    f"IAE bottoms x_A {run['IAE_xBA']:.2e}   jacket at a limit {run['Fj_at_limit']:.0%}   "
                    f"min jacket flow {run['Fj_min']:.3f}   swing over last 300 min: "
                    f"T {run['late_swing_T']:.3f} K, jacket {run['late_swing_Fj']:.3f} m3/min   "
                    f"bottoms x_A peak {run['xBA'].max():.2e}")
                if run["late_swing_T"] > 1.0:
                    late = run["t"] >= run["t"][-1] - 300.0
                    tl, Tl = run["t"][late], run["T"][late]
                    peaks = tl[1:-1][(Tl[1:-1] > Tl[:-2]) & (Tl[1:-1] >= Tl[2:]) & (Tl[1:-1] > Tl.mean())]
                    say(f"      last 300 min: reactor T {Tl.min():.2f}-{Tl.max():.2f} K, "
                        f"oscillation period {np.diff(peaks).mean():.1f} min")

        header("DYNAMIC VALIDATION: reactor temperature set-point +1 K at 30 min")
        for name, v in ex.validation.items():
            t, T = v["t"], v["T"]
            final = T[-1]
            outside = np.where((t > 30) & (np.abs(T - final) > 0.05))[0]
            settle = t[outside[-1]] - 30 if outside.size else 0.0
            say(f"{name}: reactor T {T[0]:.2f} -> {final:.2f} K, peak {T.max():.2f} K, settles within 0.05 K "
                f"after {settle:.0f} min")
            say(f"   C_A {v['CA'][0]:.4f} -> {v['CA'][-1]:.4f}   recycle {v['recycle'][0]:.4f} -> {v['recycle'][-1]:.4f}   "
                f"production {v['production'][0]:.4f} -> {v['production'][-1]:.4f} kmol/min")
            say(f"   reactor inlet {v['T_in'].min():.2f}-{v['T_in'].max():.2f} K   jacket flow {v['Fj'].min():.3f}-{v['Fj'].max():.3f}   "
                f"boil-up {v['V_boil'][0]:.3f} -> {v['V_boil'][-1]:.3f}")
            late = t >= t[-1] - 300.0
            drift = np.polyfit(t[late], v["recycle"][late], 1)[0] * 1000.0
            say(f"   distillate x_B {v['xDB'][0]:.4f} -> {v['xDB'][-1]:.4f}   product B recycled "
                f"{v['recycle'][0] * v['xDB'][0]:.3f} -> {v['recycle'][-1] * v['xDB'][-1]:.3f} kmol/min   "
                f"recycle drift over the last 300 min {drift:+.4f} kmol/min per 1000 min   "
                f"bottoms x_A {v['xBA'][0]:.2e} -> {v['xBA'][-1]:.2e} (peak {v['xBA'].max():.2e})   "
                f"run {t[-1]:.0f} min")
            for k, q in v["duties"].items():
                say(f"   {k:40s} {q[0] * GJH:6.2f} -> {q[-1] * GJH:6.2f} GJ/h  (range {q.min() * GJH:.2f}-{q.max() * GJH:.2f})")


# ------------------------------------------------------------------------------------
# Figures
# ------------------------------------------------------------------------------------


def fig_temperatures(ex):
    pp = ex.design.pp
    rows = ex.temperatures
    P = [r["P"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    for key, label in (("T_bottoms", "bottoms (reboiler)"), ("T_dew", "condenser, dew point"),
                       ("T_drum", "condenser, bubble point"), ("T_column_feed", "column feed bubble point")):
        ax.plot(P, [r[key] for r in rows], marker=".", label=label)
    Tr, T0, dT = ex.op.T_reactor, pp.reactor.T0, pp.thermo.dT_min
    for level, text, style, va, dy in (
            (Tr - dT, r"reactor temperature $-\,\Delta T_{min}$", "--", "bottom", 1.0),
            (T0 + dT, r"reactor inlet $+\,\Delta T_{min}$", "-.", "top", -1.0),
            (pp.economics.cw_T + dT, r"cooling water $+\,\Delta T_{min}$", ":", "bottom", 1.0)):
        ax.axhline(level, color="k", ls=style, lw=0.8)
        ax.text(P[-1], level + dy, text, ha="right", va=va, fontsize=7)
    # The column pressures of the designs.
    for P_design, text in ((0.15, "D4"), (0.4, "D3"), (0.8, "D1, D5")):
        ax.axvline(P_design, color="0.6", lw=0.6)
        ax.text(P_design, 1.0, text, transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=7, color="0.3")
    pressure_axis(ax)
    ax.set_ylabel("temperature, K")
    ax.legend(fontsize=7, loc="upper left")
    fig.savefig(FIG / "temperature_levels.png")
    plt.close(fig)


def fig_composites(ex):
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.2))
    for col_i, name in enumerate(("D4", "D3")):
        P = hen.DESIGNS[name].pressure
        row = next(r for r in ex.targets if r["P"] == P)
        table = row["streams"]
        Hh, Th = pinch.composite_curve(table, hot=True)
        Hc, Tc = pinch.composite_curve(table, hot=False)
        ax = axes[0, col_i]
        ax.plot(Hh * GJH, Th, color="tab:red", label="hot composite")
        ax.plot((Hc + row["QC_min"]) * GJH, Tc, color="tab:blue", label="cold composite")
        ax.set_title(f"{P} bar ({name}): $Q_{{H,min}}$ {row['QH_min'] * GJH:.1f} GJ/h, "
                     f"$Q_{{C,min}}$ {row['QC_min'] * GJH:.1f} GJ/h", fontsize=8)
        ax.set_xlabel("enthalpy, GJ/h")
        ax.set_ylabel("temperature, K")
        ax.legend(fontsize=7, loc="lower right")
        if name == "D3":
            # No hot stream reaches the reboiler here, so the composites meet end to end
            # rather than across a gap of dT_min; the reboiler is met by steam alone.
            reb = next(st for st in table if st.name == "reboiler")
            H_reb = (Hc[-1] + row["QC_min"]) * GJH - reb.duty * GJH
            ax.annotate("reboiler: no process heat\nabove it, all hot utility",
                        xy=(H_reb, reb.T_target), xytext=(H_reb + 8, reb.T_target - 25),
                        fontsize=7, ha="left", arrowprops=dict(arrowstyle="->", lw=0.7))
        pt = pinch.problem_table(table, ex.design.pp.thermo.dT_min)
        H, T = pinch.grand_composite(pt)
        ax = axes[1, col_i]
        ax.plot(H * GJH, T, color="k")
        ax.set_xlabel("net heat flow, GJ/h")
        ax.set_ylabel("shifted temperature, K")
        ax.set_title(f"{P} bar ({name}): grand composite curve", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "composite_curves.png")
    plt.close(fig)


def fig_pareto(ex):
    results = ex.pareto
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    profit = np.array([r.evaluation.profit for r in results])
    gwp = np.array([r.evaluation.gwp_energy for r in results])
    f1 = pareto.normalize(profit, best=profit.max(), worst=profit.min())
    f2 = pareto.normalize(gwp, best=gwp.min(), worst=gwp.max())
    for name in ("D1", "D3", "D4", "D5"):
        idx = [i for i, r in enumerate(results) if r.design == name]
        ax.plot(f1[idx], f2[idx], "o", ms=3, color=COLORS[name], alpha=0.3)
        keep = pareto.nondominated(profit[idx], gwp[idx])
        front = sorted((i for i, k in zip(idx, keep) if k), key=lambda i: f1[i])
        ax.plot(f1[front], f2[front], "-o", ms=3, color=COLORS[name], label=name)
    idx = [i for i, r in enumerate(results) if r.design == "D2"]
    keep = pareto.nondominated(profit[idx], gwp[idx])
    front = sorted((i for i, k in zip(idx, keep) if k), key=lambda i: f1[i])
    ax.plot(f1[front], f2[front], "k--", lw=1.0, label="D2 pinch target")
    ax.set_xlabel("economic objective (0 = highest profit)")
    ax.set_ylabel("environmental objective (0 = lowest GWP)")
    ax.legend(fontsize=7)
    fig.savefig(FIG / "pareto.png")
    plt.close(fig)


# D1, D3 and D5 respond almost identically, so they are told apart by line style.
STYLE = {"D1": dict(color=COLORS["D1"], lw=2.6), "D3": dict(color=COLORS["D3"], lw=1.4, ls="--"),
         "D5": dict(color=COLORS["D5"], lw=1.2, ls=":"), "D4": dict(color=COLORS["D4"], lw=0.5)}


def fig_controllability(ex):
    fig, axes = plt.subplots(3, 2, figsize=(8.6, 7.6))
    zoom = 60.0
    for name in ("D4", "D1", "D3", "D5"):
        run = ex.controllability[name]["runs"]["throughput +10%"]
        t = run["t"]
        late = t >= t[-1] - zoom
        axes[0, 0].plot(t, run["CA"], label=name, **STYLE[name])
        axes[0, 1].plot(t, run["xBA"], label=name, **STYLE[name])
        axes[1, 0].plot(t, run["Fj"], label=name, **STYLE[name])
        axes[1, 1].plot(t, run["T"], label=name, **STYLE[name])
        axes[2, 0].plot(t[late], run["Fj"][late], label=name, **STYLE[name])
        axes[2, 1].plot(t[late], run["T"][late], label=name, **STYLE[name])
    labels = (r"reactor $C_A$, kmol/m$^3$", "A in the bottoms, mole fraction",
              "jacket coolant flow, m$^3$/min", "reactor temperature, K",
              f"jacket coolant flow, last {zoom:.0f} min", f"reactor temperature, last {zoom:.0f} min")
    for ax, lab in zip(axes.ravel(), labels):
        ax.set_ylabel(lab)
        ax.set_xlabel("time, min")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "controllability.png")
    plt.close(fig)


def fig_validation(ex):
    early = 200.0
    for name in (comparison.FINAL_DESIGN, "D4"):
        v = ex.validation[name]
        t = v["t"]
        e = t <= early
        fig, axes = plt.subplots(4, 2, figsize=(8.6, 9.2))
        axes[0, 0].plot(t[e], v["T"][e]); axes[0, 0].set_ylabel("reactor temperature, K")
        axes[0, 1].plot(t[e], v["Fj"][e]); axes[0, 1].set_ylabel("jacket coolant flow, m$^3$/min")
        axes[1, 0].plot(t[e], v["T_in"][e]); axes[1, 0].set_ylabel("reactor inlet temperature, K")
        for k, q in v["duties"].items():
            axes[1, 1].plot(t, q * GJH, label=EXCHANGER_LABELS[k])
        axes[1, 1].set_ylabel("exchanger duty, GJ/h")
        axes[1, 1].legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.0, 1.0))
        axes[2, 0].plot(t, v["CA"]); axes[2, 0].set_ylabel(r"reactor $C_A$, kmol/m$^3$")
        axes[2, 1].plot(t, v["recycle"]); axes[2, 1].set_ylabel("recycle, kmol/min")
        axes[3, 0].plot(t, v["xDB"]); axes[3, 0].set_ylabel("B in the recycle, mole fraction")
        axes[3, 1].plot(t, v["xBA"]); axes[3, 1].set_ylabel("A in the bottoms, mole fraction")
        for ax in axes.ravel():
            ax.set_xlabel("time, min")
        fig.tight_layout()
        fig.savefig(FIG / f"validation_{name}.png")
        plt.close(fig)


def main(argv=None):
    global FIG
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=pathlib.Path, default=RESULTS,
                        help="directory for designs.txt")
    parser.add_argument("--cache", type=pathlib.Path, default=CACHE,
                        help="directory for designs.pkl")
    parser.add_argument("--figures", type=pathlib.Path, default=FIG,
                        help="directory for the figures")
    parser.add_argument("--cached", action="store_true",
                        help="report from the cached run instead of recomputing")
    args = parser.parse_args(argv)
    FIG = args.figures
    FIG.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    cache = args.cache / "designs.pkl"
    if args.cached and cache.exists():
        ex = pickle.loads(cache.read_bytes())
    else:
        ex = comparison.compute()
        cache.write_bytes(pickle.dumps(ex))
    report(ex)
    (args.out / "designs.txt").write_text(OUT.getvalue())
    for f in (fig_temperatures, fig_composites, fig_pareto, fig_controllability, fig_validation):
        f(ex)
        print("wrote", f.__name__, flush=True)


if __name__ == "__main__":
    main()
