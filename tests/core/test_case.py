"""The case interface on the toy case: configurations, overrides, build and run."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from plantbench.core import case as cs
from plantbench.core import control as ctl
from plantbench.core.case import Config, Trajectory, override


def test_config_round_trips_through_plain_data():
    c = Config("PI", options={"a": 1}, params={"x.y": 2.0}, tuning={"L": {"Kc": 1.0}},
               setpoint_steps={"L": [(1.0, 2.0)]}, disturbance=[{"name": "step"}])
    d = c.to_dict()
    assert Config.from_dict(d) == Config.from_dict(Config.from_dict(d).to_dict())
    assert Config.from_dict(d).key() == c.key()


def test_config_key_changes_with_content():
    assert Config("PI").key() != Config("PI", setpoints={"L": 1.0}).key()


def test_config_rejects_unknown_sections_and_a_missing_structure():
    with pytest.raises(KeyError, match="setpoint"):
        Config.from_dict({"structure": "PI", "setpoint": {}})
    with pytest.raises(KeyError, match="structure"):
        Config.from_dict({"options": {}})


@dataclass(frozen=True)
class Inner:
    a: float = 1.0
    b: tuple = (1.0, 2.0)


@dataclass(frozen=True)
class Outer:
    inner: Inner = Inner()
    c: float = 3.0


def test_override_reaches_nested_fields_and_keeps_tuples():
    o = override(Outer(), {"inner.a": 5.0, "inner.b": [3.0, 4.0], "c": 0.0})
    assert o == Outer(Inner(5.0, (3.0, 4.0)), 0.0)


@pytest.mark.parametrize("path", ["inner.z", "z", "c.a"])
def test_override_rejects_a_path_that_names_no_field(path):
    with pytest.raises(KeyError):
        override(Outer(), {path: 1.0})


def test_validate_fills_options_and_rejects_unknowns(case):
    assert case.config().options == {"m_design": 1.0}
    with pytest.raises(KeyError, match="structure"):
        case.config("nope")
    with pytest.raises(KeyError, match="options"):
        case.config(options={"nope": 1})


def test_build_applies_every_section(case):
    s = cs.build(case, case.config(
        params={"K": 3.0}, options={"m_design": 2.0},
        tuning={"level": {"Kc": 0.8}}, setpoints={"level": 5.0},
        instruments={"level": {"tau": 1.0}},
        disturbance={"name": "step", "field": "d", "value": 1.0, "t": 5.0}))
    assert s.design.pp.K == 3.0 and s.design.x[0] == pytest.approx(6.0)
    loop = s.structure[0]
    assert loop.Kc == 0.8 and loop.setpoint == 5.0 and loop.instrument.tau == 1.0
    assert ctl.n_augmented(s.structure) == 2
    assert s.disturbance(10.0, s.design.u).d == 1.0


def test_build_rejects_an_unknown_disturbance(case):
    with pytest.raises(KeyError, match="nope"):
        cs.build(case, case.config(disturbance={"name": "nope"}))


def test_run_returns_a_trajectory_that_survives_a_file(case, tmp_path):
    tr = cs.run(case, case.config(setpoints={"level": 3.0}), t_end=100.0, dt=1.0)
    assert tr.x.shape == (1, 101) and tr.state_names == ["x"]
    assert np.allclose(tr.y["x"], tr.state("x"))
    assert tr.wall_time > 0 and tr.nfev > 0
    tr.to_npz(tmp_path / "run.npz")
    back = Trajectory.from_npz(tmp_path / "run.npz")
    assert back.config == tr.config and back.state_names == tr.state_names
    assert np.array_equal(back.x, tr.x) and set(back.u) == set(tr.u)
    assert np.array_equal(back.u["z[0]"], tr.u["z[0]"])
    w = tr.window(10.0, 20.0)
    assert w.t[0] == 10.0 and w.t[-1] == 20.0 and w.x.shape == (1, 11)
