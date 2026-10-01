"""Operating-regime discovery on `reactor_separator_recycle`'s design space.

    .venv/bin/python -m studies.regimes.analysis [DATA_DIR] [--out DIR]

DATA_DIR defaults to data/regimes, generated from spec.toml.  Writes every number the
study reports to results/results_regimes.txt, and the arrays the figures are drawn
from to results/analysis.npz (figures.py).

Two questions are asked of the trajectories.

(i) Do the outcomes of `labels.py` (settled, unsettled, limit cycle) separate in a
low-dimensional embedding of the late window of each run, and which variables separate
the clusters?  Observations are time samples, each carrying its run and time, as in the
data mining workflow of Briceno-Mena et al. (2022) and Seghers et al. (2023): sampled every 10 min, each
sample the mean of the four 2 min samples from 4 min before it to 2 min after, z-scored, embedded by PCA, t-SNE and
PaCMAP, clustered by HDBSCAN, and compared cluster against cluster by subspace greedy
search (SGS), as implemented in discovery.py.

The late window is represented in two ways.  As deviations from the design point, a
sample carries the operating point the upset has moved the plant to: the global
structure.  As deviations from the run's own late-window mean, it carries only the
behavior around that operating point, a limit cycle as a loop and a settled plant as a
point: the local structure.  A limit cycle oscillates about the same operating point a
settled run under the same upset reaches, so the two representations are needed to
separate what the upset did from what the plant does about it.

(ii) Does the early window, the first 100 min after the upset, predict the late outcome?
The question is posed as the task of spec.toml: the signals over the first 130 min of
each run, the outcome as the label, the folds and the metrics are the task's, and
`plantbench.task` hands them over.  Observations are runs, each represented by its
early-window deviations flattened into one vector.  The vectors are embedded and
clustered the same way, and a k-nearest-neighbor classifier in the embedding is scored
on the task's folds against the late outcome.  The embeddings are computed without the
labels and on all runs, so the cross-validation is transductive in the embedding;
baselines use the same folds.

Section 6 asks how much of that answer survives three stricter readings: folds that hold
out whole plant-and-upset combinations, so that no run of a test combination is seen in
training; a classifier on the design variables themselves (the plant, the upset, the
tuning and the deadtime); and the damping ratio read at the recall the early classifier
reaches, so that the two are compared on false alarms at equal detections.
"""

from __future__ import annotations

import argparse
import collections
import io
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

from plantbench import task
from plantbench.core.spec import Spec

from . import dataset, discovery
from .labels import LATE, OUTCOMES

HERE = Path(__file__).resolve().parent
SPEC = Spec.load(HERE / "spec.toml")


def _upset_time(spec: Spec) -> float:
    """The one time at which every upset of the specification acts."""
    times = set()
    for member in spec.variants["upset"]:
        if "disturbance" in member:
            times.add(float(member["disturbance"]["t"]))
        for steps in member.get("setpoint_steps", {}).values():
            times.update(float(t) for t, _ in steps)
    if len(times) != 1:
        raise ValueError(f"the upsets of spec.toml act at {sorted(times)}; the analysis "
                         "needs one time")
    return times.pop()


T_UPSET = _upset_time(SPEC)  # min
LATE_FROM = SPEC.run["t_end"] - LATE  # min, the window the outcome is read from
EARLY = (T_UPSET, T_UPSET + 100.0)  # min
# Samples per observation.  discovery.sampling_avg keeps every INTERVAL-th sample and
# averages the INTERVAL - 1 samples from INTERVAL // 2 before it to one after it, as the
# workflow it reproduces does: at the dataset's 2 min spacing, a point every 10 min, each the
# mean of 8 min, with shorter windows at the two ends.
INTERVAL = 5
SEEDS = (0, 1, 2)  # for t-SNE, PaCMAP and the reductions inside SGS
# HDBSCAN: a late-window cluster must hold the late windows of at least five runs; for
# run-level observations the workflow's default of 10 is kept.
MIN_RUNS_PER_CLUSTER = 5
MIN_CLUSTER_RUNS = 10
KNN = 10  # neighbors of the classifier in question (ii)
FOLDS = 5
TASK = task.Task.of(HERE / "spec.toml")  # question (ii), as the specification states it
METHODS = ("PCA", "t-SNE", "PaCMAP")

OUT = io.StringIO()


def say(s: str = "") -> None:
    print(s, flush=True)
    OUT.write(s + "\n")


def heading(title: str) -> None:
    say()
    say("=" * 88)
    say(title)
    say("=" * 88)


def embed(method: str, Z: np.ndarray, seed: int) -> np.ndarray:
    if method == "PCA":
        return discovery.dr_pca(Z)[0]
    if method == "t-SNE":
        return discovery.dr_tsne(Z, seed=seed)
    return discovery.dr_pacmap(Z, seed=seed)


def ari(a, b, mask=None) -> float:
    """Adjusted Rand index over the observations in `mask`; NaN if there are none."""
    a, b = np.asarray(a), np.asarray(b)
    if mask is not None:
        a, b = a[mask], b[mask]
    return float(adjusted_rand_score(a, b)) if a.size else float("nan")


def purity(clusters: np.ndarray, labels: np.ndarray) -> float:
    """Share of clustered observations whose outcome is their cluster's majority outcome."""
    keep = clusters >= 0
    total = 0
    for c in set(clusters[keep]):
        total += collections.Counter(labels[clusters == c]).most_common(1)[0][1]
    return total / max(keep.sum(), 1)


# ------------------------------------------------------------------------------------
# Observations
# ------------------------------------------------------------------------------------


REPRESENTATIONS = {
    "global": "deviations from the design point",
    "local": "deviations from each run's own late-window mean",
}


# Cleaning: a variable whose spread over the observations is below this fraction of its
# typical level carries only floating-point round-off.  z-scoring would divide the
# round-off by its own tiny standard deviation and make it a unit-variance variable that
# dominates distances; such variables are dropped before normalization.
ROUNDOFF = 1e-8


def clean(X: np.ndarray, levels: np.ndarray, names: list[str]):
    """Drop variables with negligible spread.  Returns (X, names kept, names dropped)."""
    scale = np.maximum(np.abs(levels), 1e-12)
    keep = X.std(axis=0) > ROUNDOFF * scale
    return X[:, keep], [n for n, k in zip(names, keep) if k], \
        [n for n, k in zip(names, keep) if not k]


def _levels(runs) -> np.ndarray:
    """Typical level of each variable: its largest magnitude at the design points."""
    return np.max(np.abs([r.X[0] for r in runs]), axis=0)


def late_samples(runs, representation: str):
    """Late-window samples: (Z, run index, time, variable names, dropped names)."""
    rows, owner, times = [], [], []
    for i, r in enumerate(runs):
        late = r.t >= LATE_FROM
        base = r.X[0] if representation == "global" else r.X[late].mean(axis=0)
        S, ids = discovery.sampling_avg((r.X - base)[late], INTERVAL)
        rows.append(S)
        owner.append(np.full(len(S), i))
        times.append(r.t[late][ids])
    X, names, dropped = clean(np.vstack(rows), _levels(runs), dataset.VARIABLES)
    Z = discovery.z_score(X)[0]
    return Z, np.concatenate(owner), np.concatenate(times), names, dropped


def early_matrix_of(t: np.ndarray, X: np.ndarray, names: list[str]):
    """One raw vector per run: early-window deviations, averaged and flattened.

    `X` holds the runs' signals from the start of the run, (runs, times, signals) on the
    shared grid `t`; each run is taken as a deviation from its first sample, the design
    point.  Returned before cleaning and normalization, so that a study which transfers a
    screen from one dataset to another can clean and z-score both by the training set's
    spread rather than each by its own.  Returns (vectors, typical level of each column,
    names).
    """
    w = (t >= EARLY[0]) & (t <= EARLY[1])
    V = np.array([discovery.sampling_avg((x - x[0])[w], INTERVAL)[0] for x in X])
    n_win = V.shape[1]
    levels = np.repeat(np.max(np.abs(X[:, 0, :]), axis=0)[None, :], n_win, axis=0).ravel()
    columns = [f"{v}@{k}" for k in range(n_win) for v in names]
    return V.reshape(len(V), -1), levels, columns


def early_matrix(runs):
    """`early_matrix_of` for runs read by `dataset.load`, which share one time grid."""
    head = runs[0].t <= EARLY[1]  # the design point and the early window, no more
    return early_matrix_of(runs[0].t[head], np.stack([r.X[head] for r in runs]),
                           dataset.VARIABLES)


def early_vectors(t: np.ndarray, X: np.ndarray, names: list[str]):
    """One vector per run: early-window deviations, averaged and flattened, z-scored."""
    flat, levels, columns = early_matrix_of(t, X, names)
    flat, kept, dropped = clean(flat, levels, columns)
    return discovery.z_score(flat)[0], sorted({d.split("@")[0] for d in dropped})


# ------------------------------------------------------------------------------------
# Sections
# ------------------------------------------------------------------------------------


def section_dataset(runs, records):
    heading("1.  DATASET")
    status = collections.Counter(r["status"] for r in records)
    say(f"runs {len(records)}: " + ", ".join(f"{k} {v}" for k, v in sorted(status.items())))
    outcomes = np.array([r.outcome for r in runs])
    say("outcomes: " + ", ".join(f"{o} {int((outcomes == o).sum())} "
                                  f"({(outcomes == o).mean():.1%})" for o in OUTCOMES))
    for attr, title in (("plant", "plant"), ("upset", "upset")):
        say(f"\noutcome by {title}")
        groups = sorted({getattr(r, attr) for r in runs})
        say(f"  {'':30s}" + "".join(f"{o:>13s}" for o in OUTCOMES))
        for g in groups:
            sel = [r.outcome for r in runs if getattr(r, attr) == g]
            c = collections.Counter(sel)
            say(f"  {g:30s}" + "".join(f"{c[o]:13d}" for o in OUTCOMES))
    say("\nlimit cycles by analyzer deadtime")
    edges = [0.0, 2.5, 5.0, 7.5, 10.0]
    d4 = [r for r in runs if r.plant.startswith("D4 share 0.7")]
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [r for r in d4 if lo <= r.deadtime < hi]
        n_lc = sum(r.outcome == "limit cycle" for r in sel)
        say(f"  D4 share 0.7, deadtime {lo:4.1f}-{hi:4.1f} min: {n_lc:3d} of {len(sel):3d} runs")
    say("\nlimit cycles at D4 share 0.7 by quarter of each sampled coordinate")
    lc_d4 = np.array([r.outcome == "limit cycle" for r in d4])
    for name, values, lo, hi in (("deadtime", [r.deadtime for r in d4], 0.0, 10.0),
                                 ("Kc", [r.Kc for r in d4], -2.0, -0.5),
                                 ("tau_I", [r.tau_I for r in d4], 5.0, 40.0)):
        v = np.array(values)
        edges = np.linspace(lo, hi, 5)
        inside = [(v >= a) & (v < b) for a, b in zip(edges[:-1], edges[1:])]
        shares = [lc_d4[m].mean() for m in inside]
        auc = roc_auc_score(lc_d4, v) if 0 < lc_d4.sum() < lc_d4.size else float("nan")
        say(f"  {name:8s} " + "  ".join(f"{int(lc_d4[m].sum()):2d}/{int(m.sum()):2d}" for m in inside)
            + f"   spread of the share {max(shares) - min(shares):.3f}   "
              f"ROC AUC {max(auc, 1 - auc):.3f}")
    say("\ncost of a run by outcome (wall time of the integration)")
    wall = np.array([r.wall_time for r in runs])
    total = sum(r["wall_time"] for r in records if r["wall_time"] == r["wall_time"])
    for o in OUTCOMES:
        w = wall[outcomes == o]
        if w.size:
            say(f"  {o:12s} n {w.size:5d}   median {np.median(w):7.2f} s   "
                f"90th percentile {np.percentile(w, 90):7.2f} s   max {w.max():7.1f} s")
    lc = outcomes == "limit cycle"
    if 0 < lc.sum() < lc.size:
        say(f"  wall time as a detector of limit cycles: ROC AUC {roc_auc_score(lc, wall):.3f}")
    say(f"  total {total / 3600:.2f} h of integration")


def section_linear(runs):
    heading("2.  THE LINEAR FEATURE AGAINST THE OUTCOME")
    outcomes = np.array([r.outcome for r in runs])
    zeta = np.array([r.features["damping_ratio"] for r in runs])
    for o in OUTCOMES:
        z = zeta[outcomes == o]
        if z.size:
            say(f"  {o:12s} damping ratio median {np.median(z):.3f}   range {z.min():.3f}-{z.max():.3f}")
    lc = outcomes == "limit cycle"
    if 0 < lc.sum() < lc.size:
        say(f"  damping ratio as a detector of limit cycles (lower = cycle): "
            f"ROC AUC {roc_auc_score(lc, -zeta):.3f}")
        best = max(((balanced_accuracy_score(lc, zeta < thr), thr) for thr in np.unique(zeta)))
        say(f"  best single threshold: damping < {best[1]:.4f}, balanced accuracy {best[0]:.3f}")
        above = zeta[lc].max()
        say(f"  largest damping ratio among limit cycles {above:.4f}; runs at or below it "
            f"that do not cycle: {int(((zeta <= above) & ~lc).sum())}")


def section_late(runs, store, representation: str, number: str):
    heading(f"{number}  LATE WINDOW, {representation.upper()} STRUCTURE: "
            f"{REPRESENTATIONS[representation].upper()}")
    Z, owner, times, names, dropped = late_samples(runs, representation)
    outcomes = np.array([r.outcome for r in runs])[owner]
    plants = np.array([r.plant for r in runs])[owner]
    upsets = np.array([r.upset for r in runs])[owner]
    per_run = int(np.bincount(owner).max())
    min_size = MIN_RUNS_PER_CLUSTER * per_run
    say(f"observations {Z.shape[0]} time samples x {Z.shape[1]} variables, "
        f"{per_run} per run over {LATE_FROM:.0f}-{runs[0].t[-1]:.0f} min, "
        f"each the mean of {INTERVAL} samples; z-scored")
    say(f"cleaning: dropped as round-off (spread below {ROUNDOFF:g} of level): "
        f"{', '.join(dropped) or 'none'}")
    say(f"HDBSCAN min_cluster_size {min_size} (the late windows of {MIN_RUNS_PER_CLUSTER} runs)")
    pre = f"{representation}_"
    store.update({pre + "owner": owner, pre + "time": times, pre + "outcome": outcomes,
                  pre + "plant": plants, pre + "upset": upsets})
    say(f"\n  {'method':8s} {'seed':>4s} {'dims':>4s} {'clusters':>8s} {'noise':>6s} "
        f"{'DBI':>6s} {'purity':>7s} {'ARI outcome':>11s} {'ARI plant':>9s} {'ARI upset':>9s}")
    results = {}
    for method in METHODS:
        for seed in (SEEDS if method != "PCA" else (0,)):
            E = embed(method, Z, seed)
            labels, dbi = discovery.cluster_hdbscan(E, min_size)
            keep = labels >= 0
            n_cl = len(set(labels) - {-1})
            results[(method, seed)] = labels
            say(f"  {method:8s} {seed:4d} {E.shape[1]:4d} {n_cl:8d} {1 - keep.mean():6.1%} "
                f"{dbi:6.3f} {purity(labels, outcomes):7.3f} {ari(labels, outcomes, keep):11.3f} "
                f"{ari(labels, plants, keep):9.3f} {ari(labels, upsets, keep):9.3f}")
            if seed == 0:
                store[f"{pre}{method}"] = E[:, :2]
                store[f"{pre}{method}_clusters"] = labels
    say("\nstability across seeds: ARI between the clusterings of two seeds")
    for method in ("t-SNE", "PaCMAP"):
        pairs = [ari(results[(method, a)], results[(method, b)])
                 for i, a in enumerate(SEEDS) for b in SEEDS[i + 1:]]
        say(f"  {method:8s} mean {np.mean(pairs):.3f}   min {np.min(pairs):.3f}")

    labels = results[("PaCMAP", 0)]
    say("\nPaCMAP clusters (seed 0): composition by outcome, and the plants and upsets they hold")
    comp = {}
    for c in sorted(set(labels)):
        m = labels == c
        oc = collections.Counter(outcomes[m])
        pl = collections.Counter(plants[m]).most_common(2)
        up = collections.Counter(upsets[m]).most_common(2)
        comp[c] = (m.sum(), oc)
        name = "noise" if c < 0 else f"cluster {c}"
        say(f"  {name:10s} {m.sum():6d} samples of {len(set(owner[m])):4d} runs   "
            + ", ".join(f"{o} {oc[o]}" for o in OUTCOMES if oc[o])
            + "   | " + "; ".join(f"{k} {v}" for k, v in pl)
            + "   | " + "; ".join(f"{k} {v}" for k, v in up))
    sgs_between_clusters(Z, labels, comp, upsets, store, representation, names)


def sgs_between_clusters(Z, labels, comp, upsets, store, representation, names):
    """SGS from the largest cluster to every other cluster, on the PaCMAP clusters.

    The reference is the largest cluster.  Each other cluster is compared against it and
    named by its majority outcome and its majority upset.
    """
    clusters = sorted((c for c in comp if c >= 0), key=lambda c: -comp[c][0])
    if len(clusters) < 2:
        say("\nSGS: fewer than two clusters, nothing to compare")
        return
    ref = clusters[0]
    say(f"\nSGS against cluster {ref}, the largest ({comp[ref][0]} samples, mostly "
        f"{comp[ref][1].most_common(1)[0][0]}); k = 10 as in Briceno-Mena et al. (2022); contribution, %, "
        f"mean over {len(SEEDS)} seeds of the reduction inside SGS")
    for c in clusters[1:]:
        per_seed = [discovery.sgs(Z, np.where(labels == ref)[0], np.where(labels == c)[0],
                                names, k=10, seed=s) for s in SEEDS]
        mean = {v: float(np.mean([p[v] for p in per_seed])) for v in names}
        ranked = sorted(mean.items(), key=lambda kv: -kv[1])
        m = labels == c
        up = collections.Counter(upsets[m]).most_common(1)[0][0]
        say(f"  cluster {c}: {comp[c][0]} samples, mostly {comp[c][1].most_common(1)[0][0]}, "
            f"mostly {up}")
        say("     " + "   ".join(f"{v} {pct:.1f} ({min(p[v] for p in per_seed):.1f}-"
                                  f"{max(p[v] for p in per_seed):.1f})" for v, pct in ranked[:5]))
        store[f"{representation}_sgs_{c}"] = np.array([mean.get(v, np.nan)
                                                        for v in dataset.VARIABLES])
        store[f"{representation}_sgs_{c}_label"] = np.array(
            f"cluster {c} ({comp[c][1].most_common(1)[0][0]}, {up}) against cluster {ref}")
        store[f"{representation}_sgs_{c}_outcome"] = np.array(comp[c][1].most_common(1)[0][0])


def _folds(y: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Stratified, shuffled and seeded: the construction the task's split names."""
    return list(StratifiedKFold(FOLDS, shuffle=True, random_state=0).split(np.zeros(len(y)), y))


def _cv_predict(features: np.ndarray, y: np.ndarray, k: int = KNN,
                folds: list | None = None) -> np.ndarray:
    pred = np.empty_like(y)
    for train, test in folds if folds is not None else _folds(y):
        clf = KNeighborsClassifier(n_neighbors=k).fit(features[train], y[train])
        pred[test] = clf.predict(features[test])
    return pred


def _combination_baseline(runs, y: np.ndarray, folds: list | None = None) -> np.ndarray:
    """Predict the majority outcome of the training runs with the same plant and upset."""
    key = np.array([f"{r.plant}|{r.upset}" for r in runs])
    pred = np.empty_like(y)
    for train, test in folds if folds is not None else _folds(y):
        overall = collections.Counter(y[train]).most_common(1)[0][0]
        majority = {}
        for g in set(key[train]):
            majority[g] = collections.Counter(y[train][key[train] == g]).most_common(1)[0][0]
        pred[test] = [majority.get(g, overall) for g in key[test]]
    return pred


def _scores(y, pred) -> str:
    lc = y == "limit cycle"
    tp = int((pred[lc] == "limit cycle").sum())
    fp = int(((pred == "limit cycle") & ~lc).sum())
    return (f"balanced accuracy {balanced_accuracy_score(y, pred):.3f}   "
            f"limit cycles found {tp} of {int(lc.sum())}, false alarms {fp}")


def _task_scores(y, pred) -> str:
    """The task's metrics, printed as `_scores` prints its own."""
    s = TASK.score(y, pred)
    return (f"balanced accuracy {s['balanced_accuracy']:.3f}   limit cycles found "
            f"{s['found']} of {int((y == TASK.positive).sum())}, false alarms "
            f"{s['false_alarms']}")


def section_early(runs, obs, store):
    heading("5.  EARLY WINDOW: DOES IT PREDICT THE LATE OUTCOME?")
    if obs.run_ids != tuple(r.run_id for r in runs):
        raise ValueError("the task's runs are not the runs the analysis read")
    y, folds = obs.y, obs.folds
    V, dropped = early_vectors(obs.t, obs.X, list(obs.signals))
    say(f"task: target {TASK.target}, window {TASK.window[0]:g}-{TASK.window[1]:g} min, "
        f"{TASK.n_folds} folds {'stratified' if TASK.stratified else 'unstratified'}, "
        f"seed {TASK.seed}; {task_digest()}")
    say(f"observations {V.shape[0]} runs x {V.shape[1]} features: the {len(obs.signals)} "
        f"variables over {EARLY[0]:.0f}-{EARLY[1]:.0f} min, sampled every 10 min as 8 min means, z-scored")
    say(f"cleaning: variables with windows dropped as round-off: {', '.join(dropped) or 'none'}")
    say(f"classifier: {KNN}-nearest neighbors, on the task's folds")
    say(f"HDBSCAN min_cluster_size {MIN_CLUSTER_RUNS}")
    say()
    for method in METHODS:
        E = embed(method, V, 0)
        labels, dbi = discovery.cluster_hdbscan(E, MIN_CLUSTER_RUNS)
        keep = labels >= 0
        say(f"  {method:8s} dims {E.shape[1]:3d}   clusters {len(set(labels) - {-1}):3d}   "
            f"noise {1 - keep.mean():5.1%}   DBI {dbi:6.3f}   ARI outcome "
            f"{ari(labels, y, keep):.3f}   purity {purity(labels, y):.3f}")
        say(f"           in the embedding: {_task_scores(y, _cv_predict(E, y, folds=folds))}")
        store[f"early_{method}"] = E[:, :2]
    store["early_outcome"] = y
    say("\nbaselines on the same folds")
    say(f"  all {V.shape[1]} early features, no reduction:  "
        f"{_task_scores(y, _cv_predict(V, y, folds=folds))}")
    zeta = np.array([[r.features["damping_ratio"], r.features["rightmost"]] for r in runs])
    say(f"  damping ratio and rightmost eigenvalue:   "
        f"{_task_scores(y, _cv_predict(discovery.z_score(zeta)[0], y, folds=folds))}")
    say(f"  majority outcome of the plant and upset:  "
        f"{_task_scores(y, _combination_baseline(runs, y, folds=folds))}")
    both = np.column_stack([V, discovery.z_score(zeta)[0]])
    say(f"  early features with the two eigenvalue features: "
        f"{_task_scores(y, _cv_predict(both, y, folds=folds))}")
    say(f"  always 'settled':                         "
        f"{_task_scores(y, np.full_like(y, 'settled'))}")


def _grouped_folds(runs, y: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Stratified folds that keep every run of a plant and upset in one fold."""
    groups = np.array([f"{r.plant}|{r.upset}" for r in runs])
    split = StratifiedGroupKFold(FOLDS, shuffle=True, random_state=TASK.seed)
    return list(split.split(np.zeros(len(y)), y, groups))


def design_vectors(runs) -> np.ndarray:
    """The design variables of each run: plant and upset one-hot, the three sampled ones z-scored."""
    cols = []
    for key in ("plant", "upset"):
        values = np.array([getattr(r, key) for r in runs])
        cols += [(values == v).astype(float) for v in sorted(set(values))]
    sampled = np.array([[r.Kc, r.tau_I, r.deadtime] for r in runs])
    return np.column_stack(cols + [discovery.z_score(sampled)[0]])


FOREST_TREES = 500


def _cv_forest(features: np.ndarray, y: np.ndarray, folds: list) -> np.ndarray:
    """A random forest with class weights inversely proportional to the outcome counts."""
    from sklearn.ensemble import RandomForestClassifier
    pred = np.empty_like(y)
    for train, test in folds:
        clf = RandomForestClassifier(n_estimators=FOREST_TREES, class_weight="balanced",
                                     random_state=0, n_jobs=-1).fit(features[train], y[train])
        pred[test] = clf.predict(features[test])
    return pred


def section_stricter(runs, obs):
    heading("6.  THE EARLY-WINDOW ANSWER UNDER STRICTER READINGS")
    y = obs.y
    V, _ = early_vectors(obs.t, obs.X, list(obs.signals))
    D = design_vectors(runs)
    zeta = np.array([[r.features["damping_ratio"], r.features["rightmost"]] for r in runs])
    Z = discovery.z_score(zeta)[0]
    grouped = _grouped_folds(runs, y)
    held = [len({f"{runs[i].plant}|{runs[i].upset}" for i in test}) for _, test in grouped]
    say(f"grouped folds: {FOLDS} folds stratified on the outcome, each holding out whole "
        f"plant-and-upset combinations ({', '.join(map(str, held))} of "
        f"{len({f'{r.plant}|{r.upset}' for r in runs})}), seed {TASK.seed}")
    say(f"design variables: plant and upset one-hot, Kc, tau_I and deadtime z-scored, "
        f"{D.shape[1]} features")
    say(f"classifiers: {KNN}-nearest neighbors, and a random forest of {FOREST_TREES} trees "
        "with balanced class weights, seed 0")
    say()
    for name, folds in (("task folds", obs.folds), ("grouped folds", grouped)):
        say(f"on the {name}")
        say(f"  all {V.shape[1]} early features, no reduction:  "
            f"{_task_scores(y, _cv_predict(V, y, folds=folds))}")
        say(f"  design variables:                         "
            f"{_task_scores(y, _cv_predict(D, y, folds=folds))}")
        say(f"  early features with the design variables: "
            f"{_task_scores(y, _cv_predict(np.column_stack([V, D]), y, folds=folds))}")
        say(f"  damping ratio and rightmost eigenvalue:   "
            f"{_task_scores(y, _cv_predict(Z, y, folds=folds))}")
        say(f"  forest, all early features:               "
            f"{_task_scores(y, _cv_forest(V, y, folds))}")
        say(f"  forest, design variables:                 "
            f"{_task_scores(y, _cv_forest(D, y, folds))}")
        if folds is obs.folds:  # held-out combinations leave it nothing but the overall majority
            say(f"  majority outcome of the plant and upset:  "
                f"{_task_scores(y, _combination_baseline(runs, y, folds=folds))}")
    say()
    lc = y == TASK.positive
    d = np.array([r.features["damping_ratio"] for r in runs])
    say("the damping ratio at the recall of the early classifier (threshold read on all "
        "runs, which favors the damping ratio)")
    for name, folds in (("task folds", obs.folds), ("grouped folds", grouped)):
        for clf, pred in (("10-NN", _cv_predict(V, y, folds=folds)),
                          ("forest", _cv_forest(V, y, folds))):
            found = int((pred[lc] == TASK.positive).sum())
            alarms = int(((pred == TASK.positive) & ~lc).sum())
            thr = np.sort(d[lc])[found - 1] if found else -np.inf
            flagged = d <= thr
            say(f"  {name:13s} {clf:6s}: early window finds {found} of {int(lc.sum())} with "
                f"{alarms} false alarms; damping <= {thr:.4f} finds "
                f"{int((flagged & lc).sum())} with {int((flagged & ~lc).sum())}")


def task_digest() -> str:
    """The protocol the early-window question is reported against, as `pb-protocol:<digest>`."""
    from plantbench.core.spec import PROTOCOL_ID, Spec, identifier
    return identifier(PROTOCOL_ID, Spec.load(HERE / "spec.toml").protocol_digest())


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data", nargs="?", default="data/regimes")
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args(argv)
    from plantbench import datagen, datasets
    records = datagen.load_records(args.data)
    runs = dataset.load(args.data)
    say(f"Operating-regime discovery on {args.data}: {len(runs)} runs with trajectories")
    provenance = datasets.stamp(args.data)
    for line in datasets.stamp_lines(provenance):
        say(line)
    store: dict = {"provenance": np.array(json.dumps(provenance))}
    section_dataset(runs, records)
    section_linear(runs)
    section_late(runs, store, "global", "3.")
    section_late(runs, store, "local", "4.")
    obs = TASK.data(args.data)
    section_early(runs, obs, store)
    section_stricter(runs, obs)
    store.update(
        run_plant=np.array([r.plant for r in runs]), run_upset=np.array([r.upset for r in runs]),
        run_outcome=np.array([r.outcome for r in runs]),
        run_wall=np.array([r.wall_time for r in runs]),
        run_damping=np.array([r.features["damping_ratio"] for r in runs]),
        run_deadtime=np.array([r.deadtime for r in runs]),
        variables=np.array(dataset.VARIABLES))
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "analysis.npz", **store)
    (args.out / "results_regimes.txt").write_text(OUT.getvalue())
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
