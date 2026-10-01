"""Tasks: the observations, labels, folds and scoring that a method reports against."""

from __future__ import annotations

import collections
import json

import numpy as np
import pytest

# Folds are scikit-learn's, imported where a task makes them; the library needs it
# only for this, so a core install skips these tests rather than failing them.
pytest.importorskip("sklearn")

from plantbench import datagen, datasets, task  # noqa: E402
from plantbench.__main__ import main  # noqa: E402
from plantbench.core.spec import Spec  # noqa: E402

SPEC = {
    "study": {"name": "t", "case": "jacketed_cstr", "seed": 1},
    "base": {"structure": "temperature PI",
             "disturbance": {"name": "feed_temperature", "change": 2.0, "t": 5.0}},
    "sweep": {"tuning.reactor T.tau_I": [10.0, 15.0, 20.0, 25.0, 30.0, 35.0]},
    "run": {"t_end": 60.0, "dt": 5.0, "features": ["spectrum"]},
}
TASK = {"target": "test.slow_loop", "signals": ["measured"], "window": [0.0, 30.0],
        "split": {"folds": 2, "stratified": True, "seed": 0},
        "metrics": ["balanced_accuracy", "found", "false_alarms"], "positive": "rare"}


@task.labeler("test.slow_loop")
def _slow_loop(tr) -> str:
    """Two of the six runs, so a fold holds one of each class."""
    return "rare" if tr.config["tuning"]["reactor T"]["tau_I"] >= 30.0 else "common"


@task.labeler("test.warmer")
def _warmer(tr) -> str:
    return "warmer" if tr.y["T"][-1] > tr.y["T"][0] else "cooler"


def _quiet(msg):
    pass


def _dataset(path, spec: dict | None = None):
    return datagen.generate(Spec.from_dict(spec or {**SPEC, "task": TASK}), path, log=_quiet)


def test_the_task_leaves_the_fingerprint_of_the_runs_alone():
    runs = Spec.from_dict(SPEC)
    with_task = Spec.from_dict({**SPEC, "task": TASK})
    assert with_task.fingerprint() == runs.fingerprint()
    assert with_task.digest() == runs.digest()
    assert with_task.protocol_digest() != runs.protocol_digest()
    assert runs.protocol_digest() == runs.digest()  # no task, so the protocol is the runs
    reordered = {k: TASK[k] for k in reversed(list(TASK))}
    assert Spec.from_dict({**SPEC, "task": reordered}).protocol_digest() == \
        with_task.protocol_digest()
    assert Spec.from_dict(with_task.to_dict()).task == with_task.task


def test_observations_are_the_named_signals_over_the_window(tmp_path):
    out = _dataset(tmp_path / "d")
    obs = task.Task.of(Spec.from_dict({**SPEC, "task": TASK})).data(out)
    assert obs.signals == ("C_A", "T", "Tj")
    assert obs.X.shape == (6, obs.t.size, 3) and obs.t[0] == 0.0 and obs.t[-1] == 30.0
    assert list(obs.run_ids) == sorted(obs.run_ids)
    assert sorted(collections.Counter(obs.y).items()) == [("common", 4), ("rare", 2)]
    tr = datagen.load_run(out, obs.run_ids[0])
    keep = tr.t <= 30.0
    assert np.array_equal(obs.X[0][:, 1], tr.y["T"][keep])


def test_explicit_signals_keep_their_order_and_may_cross_the_groups(tmp_path):
    out = _dataset(tmp_path / "d")
    t = task.Task.from_dict({**TASK, "signals": ["Fj", "T"]})
    obs = t.data(out)
    assert obs.signals == ("Fj", "T")
    tr = datagen.load_run(out, obs.run_ids[0])
    assert np.array_equal(obs.X[0][:, 0], tr.u["Fj"][tr.t <= 30.0])
    both = task.Task.from_dict({**TASK, "signals": ["measured", "manipulated"]}).data(out)
    assert both.signals == ("C_A", "T", "Tj", "F0", "CA0", "T0", "Fj")


def test_the_observations_do_not_depend_on_the_order_the_runs_finished(tmp_path):
    out = _dataset(tmp_path / "d")
    t = task.Task.from_dict(TASK)
    first = t.data(out)
    lines = (out / "runs.jsonl").read_text().splitlines()
    (out / "runs.jsonl").write_text("\n".join(reversed(lines)) + "\n")
    second = t.data(out)
    assert second.run_ids == first.run_ids
    assert np.array_equal(second.X, first.X) and np.array_equal(second.y, first.y)
    for (a_train, a_test), (b_train, b_test) in zip(first.folds, second.folds):
        assert np.array_equal(a_train, b_train) and np.array_equal(a_test, b_test)


def test_the_folds_are_the_seeded_stratified_split(tmp_path):
    from sklearn.model_selection import StratifiedKFold
    out = _dataset(tmp_path / "d")
    obs = task.Task.from_dict(TASK).data(out)
    expected = StratifiedKFold(2, shuffle=True, random_state=0).split(obs.X.reshape(6, -1),
                                                                     obs.y)
    for (train, test), (want_train, want_test) in zip(obs.folds, expected):
        assert np.array_equal(train, want_train) and np.array_equal(test, want_test)
    assert all(set(obs.y[test]) == {"common", "rare"} for _, test in obs.folds)


def test_the_metrics_count_what_they_say(tmp_path):
    t = task.Task.from_dict(TASK)
    y = np.array(["rare", "rare", "common", "common", "common", "common"])
    assert t.score(y, np.array(["rare", "common", "rare", "common", "common", "common"])) == \
        {"balanced_accuracy": 0.625, "found": 1, "false_alarms": 1}
    never = np.full(6, "common")
    assert t.score(y, never) == {"balanced_accuracy": 0.5, "found": 0, "false_alarms": 0}
    assert t.score(y, y) == {"balanced_accuracy": 1.0, "found": 2, "false_alarms": 0}
    with pytest.raises(ValueError, match="predictions for"):
        t.score(y, y[:3])


def test_a_task_that_names_what_is_not_there_is_refused(tmp_path):
    out = _dataset(tmp_path / "d")
    with pytest.raises(KeyError, match="no labeler"):
        task.Task.from_dict({**TASK, "target": "nobody.knows"})
    with pytest.raises(KeyError, match="unknown metrics"):
        task.Task.from_dict({**TASK, "metrics": ["f1"]})
    with pytest.raises(KeyError, match="name it in .task."):
        task.Task.from_dict({**TASK, "positive": None, "metrics": ["found"]})
    with pytest.raises(KeyError, match="holds no signal"):
        task.Task.from_dict({**TASK, "signals": ["reactor T"]}).data(out)
    with pytest.raises(ValueError, match="no sample in the window"):
        task.Task.from_dict({**TASK, "window": [600.0, 900.0]}).data(out)


def test_a_malformed_task_table_is_refused_by_the_specification():
    with pytest.raises(KeyError, match="unknown keys"):
        Spec.from_dict({**SPEC, "task": {**TASK, "horizon": 10}})
    with pytest.raises(KeyError, match="needs 'window'"):
        Spec.from_dict({**SPEC, "task": {k: v for k, v in TASK.items() if k != "window"}})
    with pytest.raises(ValueError, match="start < end"):
        Spec.from_dict({**SPEC, "task": {**TASK, "window": [130.0, 30.0]}})
    with pytest.raises(ValueError, match="two folds"):
        Spec.from_dict({**SPEC, "task": {**TASK, "split": {"folds": 1}}})


def test_a_dataset_of_a_task_records_the_protocol(tmp_path, capsys):
    with_task = _dataset(tmp_path / "with")
    plain = _dataset(tmp_path / "without", SPEC)
    manifest = json.loads((with_task / "manifest.json").read_text())
    assert json.loads(manifest["protocol_fingerprint"])["task"]["target"] == "test.slow_loop"
    assert "protocol_fingerprint" not in json.loads((plain / "manifest.json").read_text())
    s = datasets.summary(with_task)
    assert s["protocol_digest"] == Spec.from_dict({**SPEC, "task": TASK}).protocol_digest()
    assert s["spec_digest"] == Spec.from_dict(SPEC).digest()
    assert datasets.summary(plain)["protocol_digest"] is None
    mark = datasets.stamp(with_task)
    assert datasets.check_stamp(mark, with_task) is None
    assert "protocol" in datasets.stamp_lines(mark)[0]
    assert main(["datasets", str(tmp_path)]) == 0
    listed = capsys.readouterr().out
    assert "protocol" in listed and s["protocol_digest"] in listed
