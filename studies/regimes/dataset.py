"""Reading the regimes dataset: each run's signals, its design-space coordinates and its
outcome."""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import plantbench as pb
from plantbench import datagen, task
from plantbench.core import control as ctl

from .labels import outcome

# The signals analyzed: the case's measurements, and the manipulated variables of the
# feedback loops.  Inputs that the upsets themselves move (the effluent flow, the feed
# composition and temperature) are left out, so that the analysis sees the plant's
# response rather than the disturbance that caused it.
MEASURED = ["reactor T", "reactor C_A", "reactor V", "drum level", "base level",
            "bottoms x_A", "tray 17 T", "recycle", "feed T"]
MANIPULATED = ["Fj", "F_fresh", "V_boil", "L_reflux", "D", "Bm", "Q_feed_trim"]
VARIABLES = MEASURED + MANIPULATED

CASE = pb.load_case("reactor_separator_recycle")


@dataclass
class Run:
    run_id: str
    config: dict
    features: dict
    wall_time: float
    t: np.ndarray
    X: np.ndarray  # (n_times, n_variables), in the order of VARIABLES
    outcome: str
    plant: str
    upset: str

    @property
    def deadtime(self) -> float:
        return float(self.config["instruments"]["reactor C_A"]["deadtime"])

    @property
    def Kc(self) -> float:
        return float(self.config["tuning"]["reactor T"]["Kc"])

    @property
    def tau_I(self) -> float:
        return float(self.config["tuning"]["reactor T"]["tau_I"])

    @property
    def network(self) -> str:
        return str(self.config["options"]["network"])

    @property
    def share(self) -> float:
        """The pump-around share; 0 for a network that has no pump-around."""
        return float(self.config["options"].get("pump_around_share") or 0.0)

    @property
    def pressure(self) -> float:
        """Column pressure in bar: the one sampled, or the network's own if unsampled."""
        P = self.config.get("params", {}).get("column.P_network")
        return float(P) if P is not None else float(_design_pressure(limit_key(self.config)))


def plant_label(config: dict) -> str:
    opts = config["options"]
    if opts.get("pump_around_share") is None:
        return opts["network"]
    return f"{opts['network']} share {opts['pump_around_share']:.1f}"


def upset_label(config: dict) -> str:
    if config.get("setpoint_steps"):
        (loop, steps), = config["setpoint_steps"].items()
        return f"{loop} set-point {steps[0][1]:+g}"
    d = config["disturbance"]
    if d["name"] == "fresh_feed_temperature":
        return f"fresh feed {d['change']:+g} K"
    return f"{d['name']} {d['fraction']:+.0%}"


def limit_key(config: dict) -> str:
    """The part of a configuration that sets the range of a loop's manipulated variable.

    The structure names the loops, and the options and parameters place them: the trim
    heater's range follows the preheat duty, which follows the network, the pump-around
    share and the column pressure.  Tuning, instruments and the upset move a loop within
    its range but not the range itself, so runs that differ only in those share a key and
    a single build answers for all of them.
    """
    return json.dumps({"structure": config.get("structure"),
                       "options": config.get("options", {}),
                       "params": config.get("params", {})}, sort_keys=True)


def build_plant(key: str) -> tuple[dict[str, tuple[str, float, float]], float]:
    """Build one plant and return only what the analysis reads from it: the range of every
    feedback loop's manipulated variable, and the column pressure it runs at.

    The setup itself is not kept.  A design space sampled continuously has as many plants
    as runs, and holding a built flowsheet for each would cost more memory than the
    trajectories do.
    """
    setup = pb.build(CASE, CASE.config(**json.loads(key)))
    limits = {e.name: (e.mv, e.lo, e.hi) for e in setup.structure
              if isinstance(e, ctl.Loop) and not e.mv.startswith("sp:")}
    return limits, float(setup.design.pressure)


_PLANTS: dict[str, tuple[dict[str, tuple[str, float, float]], float]] = {}


def prime(configs, workers: int = 1) -> None:
    """Build every distinct plant among `configs` up front, optionally in parallel.

    A build takes about a second, so a dataset that samples the design space continuously
    needs as many builds as runs and is worth spreading over the machine.  Priming is an
    optimization only: `_plant` builds on demand whatever is missing.
    """
    keys = sorted({limit_key(c) for c in configs} - set(_PLANTS))
    if not keys:
        return
    if workers <= 1:
        _PLANTS.update({k: build_plant(k) for k in keys})
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        _PLANTS.update(zip(keys, pool.map(build_plant, keys)))


def _plant(key: str) -> tuple[dict[str, tuple[str, float, float]], float]:
    if key not in _PLANTS:
        _PLANTS[key] = build_plant(key)
    return _PLANTS[key]


def _limits(key: str) -> dict[str, tuple[str, float, float]]:
    return _plant(key)[0]


def _design_pressure(key: str) -> float:
    return _plant(key)[1]


@task.labeler("regimes.outcome")
def outcome_of(tr) -> str:
    """The outcome of one run, read from its trajectory by the criterion of labels.py.

    Registered under the name a specification's [task] table gives as its target, so a
    task reads the label the same way the analysis does.
    """
    signals = {**tr.y, **tr.u}
    mvs = {name: (signals[mv], lo, hi)
           for name, (mv, lo, hi) in _limits(limit_key(tr.config)).items()}
    return outcome(tr.t, tr.y["reactor T"], mvs)


def load(out_dir: str | Path | list, t_from: float = 0.0, workers: int = 1) -> list[Run]:
    """Every run that finished, with its outcome, in order of run id.

    `out_dir` may be one dataset directory or several, which are read as one collection.
    The journal records runs in the order they finished, which in a parallel generation
    is not the order they were submitted in and differs between generations of the same
    specification.  t-SNE, PaCMAP and HDBSCAN all depend on the order of their input
    rows, so the analysis is sorted by run id here and depends only on the data.  Sorting
    across the directories together, rather than concatenating them, keeps the order a
    property of the runs rather than of how they were split into specifications.
    """
    dirs = [out_dir] if isinstance(out_dir, (str, Path)) else list(out_dir)
    records = [(d, r) for d in dirs for r in datagen.load_records(d) if r["status"] == "ok"]
    seen: dict[str, str] = {}
    for d, r in records:
        if r["run_id"] in seen:
            raise ValueError(f"run {r['run_id']} is in both {seen[r['run_id']]} and {d}; "
                             "the directories overlap and would be counted twice")
        seen[r["run_id"]] = str(d)
    prime([r["config"] for _, r in records], workers)
    runs = []
    for out_dir, r in records:
        tr = datagen.load_run(out_dir, r["run_id"])
        signals = {**tr.y, **tr.u}
        if not all(np.isfinite(signals[v]).all() for v in VARIABLES):
            raise ValueError(f"run {r['run_id']} is recorded as ok but holds non-finite "
                             "values; regenerate the dataset")
        keep = tr.t >= t_from
        runs.append(Run(
            run_id=r["run_id"], config=r["config"], features=r["features"],
            wall_time=r["wall_time"], t=tr.t[keep],
            X=np.column_stack([signals[v][keep] for v in VARIABLES]),
            outcome=outcome_of(tr),
            plant=plant_label(r["config"]), upset=upset_label(r["config"]),
        ))
    runs.sort(key=lambda r: r.run_id)
    return runs
