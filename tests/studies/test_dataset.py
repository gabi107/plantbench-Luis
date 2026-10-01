"""Reading a dataset of runs: which plants a run is scored against, and in what order."""

from __future__ import annotations

import pytest

pytest.importorskip("sklearn")

from studies.regimes import dataset  # noqa: E402


def _config(**over) -> dict:
    config = {"structure": "effluent fixed + cascade",
              "options": {"network": "D4", "pump_around_share": 0.5, "V_boil": 28.0},
              "params": {"column.P_network": 0.2},
              "tuning": {"reactor T": {"Kc": -1.0, "tau_I": 20.0}},
              "instruments": {"reactor C_A": {"order": 2, "deadtime": 3.0}},
              "disturbance": {"name": "throughput", "fraction": 0.1, "t": 30.0}}
    config.update(over)
    return config


def test_two_shares_a_tenth_apart_are_different_plants():
    """The share is sampled continuously, so a key that rounded it would score a run
    against the limits of a plant it was not run on."""
    a = dataset.limit_key(_config(options={"network": "D4", "pump_around_share": 0.34}))
    b = dataset.limit_key(_config(options={"network": "D4", "pump_around_share": 0.35}))
    assert a != b


def test_the_pressure_changes_the_plant_a_run_is_scored_against():
    a = dataset.limit_key(_config(params={"column.P_network": 0.20}))
    b = dataset.limit_key(_config(params={"column.P_network": 0.25}))
    assert a != b


@pytest.mark.parametrize("field,value", [
    ("tuning", {"reactor T": {"Kc": -2.0, "tau_I": 40.0}}),
    ("instruments", {"reactor C_A": {"order": 2, "deadtime": 9.0}}),
    ("disturbance", {"name": "inert", "fraction": 0.5, "t": 30.0}),
])
def test_what_moves_a_loop_within_its_range_does_not_change_the_key(field, value):
    """Otherwise a dataset that samples tuning would need one build per run for nothing."""
    assert dataset.limit_key(_config(**{field: value})) == dataset.limit_key(_config())


def test_the_key_does_not_depend_on_how_the_configuration_was_written():
    plain = _config()
    shuffled = {k: plain[k] for k in reversed(list(plain))}
    assert dataset.limit_key(shuffled) == dataset.limit_key(plain)


def test_a_plant_carries_its_loop_limits_and_its_pressure():
    limits, pressure = dataset.build_plant(dataset.limit_key(_config()))
    assert pressure == pytest.approx(0.2)
    assert limits["reactor T"][0] == "Fj"
    for name, (mv, lo, hi) in limits.items():
        assert lo < hi, name


def test_the_trim_heater_range_follows_the_plant():
    """The one loop whose range the design moves, which is why the key carries the design."""
    low, _ = dataset.build_plant(dataset.limit_key(_config(params={"column.P_network": 0.16})))
    high, _ = dataset.build_plant(dataset.limit_key(_config(params={"column.P_network": 0.26})))
    assert low["feed T"] != high["feed T"]


def test_the_regimes_specification_states_its_early_window_question_as_a_task():
    """The task of spec.toml is the question the analysis asks, and it leaves the runs of
    the specification, and so the digest data/regimes records, as they were."""
    from pathlib import Path

    from plantbench import task
    from plantbench.core.spec import Spec

    spec = Spec.load(Path(__file__).resolve().parents[2] / "studies/regimes/spec.toml")
    t = task.Task.of(spec)
    assert t.target == "regimes.outcome" and t.positive == "limit cycle"
    assert list(t.signals) == dataset.VARIABLES and t.window == (0.0, 130.0)
    assert spec.digest() == "6a6fc673261b"
