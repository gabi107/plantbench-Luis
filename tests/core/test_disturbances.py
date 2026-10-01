"""Disturbances act on any inputs dataclass and never change the nominal inputs."""

from __future__ import annotations

import pytest
from _toy import ToyInputs

from plantbench.core import disturbances as dist


def test_step_by_fraction_and_to_a_value():
    u0 = ToyInputs(m=2.0)
    assert dist.step("m", fraction=0.1, t=5.0)(4.9, u0) is u0
    assert dist.step("m", fraction=0.1, t=5.0)(5.0, u0).m == pytest.approx(2.2)
    assert dist.step("m", value=7.0, t=5.0)(6.0, u0).m == 7.0
    assert u0.m == 2.0


def test_step_needs_exactly_one_target():
    with pytest.raises(ValueError):
        dist.step("m", t=0.0)(1.0, ToyInputs())
    with pytest.raises(ValueError):
        dist.step("m", fraction=0.1, value=1.0, t=0.0)(1.0, ToyInputs())


def test_step_on_a_field_the_inputs_lack():
    with pytest.raises(AttributeError, match="nope"):
        dist.step("nope", value=1.0, t=0.0)(1.0, ToyInputs())


def test_ramp_is_linear_and_then_holds():
    d = dist.ramp("m", value=3.0, t=10.0, duration=20.0)
    u0 = ToyInputs(m=1.0)
    assert d(20.0, u0).m == pytest.approx(2.0)
    assert d(100.0, u0).m == pytest.approx(3.0)


def test_combine_applies_in_order():
    d = dist.combine(dist.step("m", value=4.0, t=0.0), dist.step("m", fraction=0.5, t=0.0))
    assert d(1.0, ToyInputs(m=1.0)).m == pytest.approx(6.0)
