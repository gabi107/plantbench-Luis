"""Tasks: what is to be predicted from a dataset, and how an answer is scored.

    [task]
    target   = "regimes.outcome"      # a registered labeler
    signals  = ["reactor T", "Fj"]    # names, or the groups "measured" and "manipulated"
    window   = [0.0, 130.0]           # min, inclusive at both ends
    split    = {folds = 5, stratified = true, seed = 0}
    metrics  = ["balanced_accuracy", "found", "false_alarms"]
    positive = "limit cycle"          # the class the last two metrics count

A specification says which runs to generate.  A task says what is to be predicted from
them and how an answer is scored, so that two methods reported against one specification
are comparable.  `Spec.protocol_digest` names the pair in one hash.

The task fixes the observations, the labels, the folds and the scoring: which runs, which
signals over which window, how the label of a run is read, how the runs are split, and
what is reported.  What happens between the signals and the prediction -- averaging,
normalization, reduction, the classifier -- is the method, and is deliberately not part
of the protocol.

A label is read by a function registered under a name, so a study keeps its own criterion
where it was written and the library holds no domain knowledge:

    @task.labeler("regimes.outcome")
    def _outcome(tr): ...

Groups are expanded against the run, so a task that names one is only as fixed as the
case behind the dataset; an explicit list is the stronger protocol.

The folds and balanced accuracy come from scikit-learn, which the library asks for only
here: `pip install 'plantbench[study]'`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from plantbench import datagen
from plantbench.core.spec import Spec, task_table

GROUPS = ("measured", "manipulated")

_LABELERS: dict[str, Callable] = {}
_METRICS: dict[str, Callable] = {}
_NEEDS_POSITIVE: set[str] = set()


def labeler(name: str) -> Callable:
    """Register a function that reads the label of one run from its trajectory."""
    def keep(fn: Callable) -> Callable:
        _LABELERS[name] = fn
        return fn
    return keep


def labelers() -> list[str]:
    return sorted(_LABELERS)


def metric(name: str, needs_positive: bool = False) -> Callable:
    """Register a metric, called as `(y, prediction, positive class) -> a number`."""
    def keep(fn: Callable) -> Callable:
        _METRICS[name] = fn
        if needs_positive:
            _NEEDS_POSITIVE.add(name)
        return fn
    return keep


def metrics() -> list[str]:
    return sorted(_METRICS)


@metric("balanced_accuracy")
def _balanced_accuracy(y, prediction, positive=None) -> float:
    """Mean recall over the classes, so a rare class counts as much as a common one."""
    from sklearn.metrics import balanced_accuracy_score
    return float(balanced_accuracy_score(y, prediction))


@metric("accuracy")
def _accuracy(y, prediction, positive=None) -> float:
    return float((y == prediction).mean())


@metric("found", needs_positive=True)
def _found(y, prediction, positive) -> int:
    """Members of the positive class the prediction found."""
    return int((prediction[y == positive] == positive).sum())


@metric("false_alarms", needs_positive=True)
def _false_alarms(y, prediction, positive) -> int:
    """Predictions of the positive class among the runs that are not of it."""
    return int(((prediction == positive) & (y != positive)).sum())


@dataclass(frozen=True)
class Observations:
    """What a task hands a method: one entry per run, in order of run id."""

    run_ids: tuple[str, ...]
    t: np.ndarray                                  # the window's grid, shared by the runs
    X: np.ndarray                                  # (runs, times, signals)
    y: np.ndarray                                  # one label per run
    signals: tuple[str, ...]
    folds: list[tuple[np.ndarray, np.ndarray]]     # train and test indices of each fold


@dataclass(frozen=True)
class Task:
    target: str
    signals: tuple[str, ...]
    window: tuple[float, float]
    metrics: tuple[str, ...]
    n_folds: int = 5
    stratified: bool = True
    seed: int = 0
    positive: str | None = None

    @classmethod
    def from_dict(cls, table: dict) -> Task:
        """A `[task]` table, with its target and metrics resolved against the registries."""
        t = task_table(table)
        if not t:
            raise KeyError("the specification carries no [task] table")
        if t["target"] not in _LABELERS:
            raise KeyError(f"no labeler {t['target']!r}; registered: {labelers()}")
        unknown = [m for m in t["metrics"] if m not in _METRICS]
        if unknown:
            raise KeyError(f"unknown metrics {unknown}; available: {metrics()}")
        counting = [m for m in t["metrics"] if m in _NEEDS_POSITIVE]
        if counting and not t.get("positive"):
            raise KeyError(f"the metrics {counting} count one class; name it in [task] "
                           "as positive")
        return cls(target=t["target"], signals=tuple(t["signals"]),
                   window=(t["window"][0], t["window"][1]), metrics=tuple(t["metrics"]),
                   n_folds=t["split"]["folds"], stratified=t["split"]["stratified"],
                   seed=t["split"]["seed"], positive=t.get("positive"))

    @classmethod
    def of(cls, spec: Spec | str | Path) -> Task:
        """The task of a specification, given as a `Spec` or the path of its file."""
        if not isinstance(spec, Spec):
            spec = Spec.load(spec)
        return cls.from_dict(spec.task)

    def to_dict(self) -> dict:
        """The task as the plain data a `[task]` table holds."""
        out = {"target": self.target, "signals": list(self.signals),
               "window": list(self.window),
               "split": {"folds": self.n_folds, "stratified": self.stratified,
                         "seed": self.seed},
               "metrics": list(self.metrics)}
        if self.positive is not None:
            out["positive"] = self.positive
        return out

    def data(self, data_dir: str | Path) -> Observations:
        """The observations of a dataset: the named signals over the window, and the labels.

        Runs are taken in order of run id, not in the order the journal recorded them, so
        that nothing downstream depends on the order a parallel generation finished in.
        """
        out = Path(data_dir)
        records = sorted((r for r in datagen.load_records(out) if r["status"] == "ok"),
                         key=lambda r: r["run_id"])
        if not records:
            raise ValueError(f"{out} holds no completed run")
        label = _LABELERS[self.target]
        names: list[str] = []
        grid: np.ndarray | None = None
        X, y = [], []
        for r in records:
            tr = datagen.load_run(out, r["run_id"])
            signals = {**tr.y, **tr.u}
            if not names:
                names = self._resolve(tr)
                if not names:
                    raise ValueError(f"the task names no signal that {out} holds; it "
                                     f"holds {sorted(signals)}")
            missing = [s for s in names if s not in signals]
            if missing:
                raise KeyError(f"run {r['run_id']} holds no signal {missing}; it holds "
                               f"{sorted(signals)}")
            keep = (tr.t >= self.window[0]) & (tr.t <= self.window[1])
            if not keep.any():
                raise ValueError(f"run {r['run_id']} has no sample in the window "
                                 f"{self.window[0]:g} to {self.window[1]:g} min")
            if grid is None:
                grid = tr.t[keep]
            elif not np.array_equal(tr.t[keep], grid):
                raise ValueError(f"run {r['run_id']} is on a different time grid from the "
                                 "runs before it; a task needs one grid across a dataset")
            X.append(np.column_stack([signals[s][keep] for s in names]))
            y.append(label(tr))
        y = np.array(y)
        return Observations(run_ids=tuple(r["run_id"] for r in records), t=grid,
                            X=np.array(X), y=y, signals=tuple(names), folds=self.splits(y))

    def splits(self, y) -> list[tuple[np.ndarray, np.ndarray]]:
        """The train and test indices of each fold, seeded, so two reports share a split."""
        from sklearn.model_selection import KFold, StratifiedKFold
        make = StratifiedKFold if self.stratified else KFold
        splitter = make(self.n_folds, shuffle=True, random_state=self.seed)
        return [(train, test) for train, test in splitter.split(np.zeros(len(y)), y)]

    def score(self, y, prediction) -> dict:
        """The task's metrics of one set of predictions, in the order the task names them."""
        y, prediction = np.asarray(y), np.asarray(prediction)
        if y.shape != prediction.shape:
            raise ValueError(f"{prediction.shape} predictions for {y.shape} observations")
        return {name: _METRICS[name](y, prediction, self.positive) for name in self.metrics}

    def _resolve(self, tr) -> list[str]:
        """Group words expanded against a run; explicit names kept in the order given."""
        group = {"measured": list(tr.y), "manipulated": list(tr.u)}
        names: list[str] = []
        for s in self.signals:
            names += group[s] if s in GROUPS else [s]
        return list(dict.fromkeys(names))
