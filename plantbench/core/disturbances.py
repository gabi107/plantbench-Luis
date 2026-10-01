"""Disturbances as functions of time acting on the inputs.

A disturbance is a function `d(t, u0) -> u` that returns the inputs in force at time t
given the nominal inputs u0.  It must return u0 itself, or a modified copy, and never
change u0 in place, because the simulator calls it with the same nominal inputs at every
evaluation.  A disturbance that changes the inputs discontinuously declares the times it
does so as an attribute `times`, and the simulator restarts the integration there.  Everything here works on any inputs dataclass, so a case needs its own
disturbance functions only where the change is not a change of one input.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable

Disturbance = Callable[[float, Any], Any]


def _target(u0, field: str, fraction: float | None, value: float | None) -> float:
    if (fraction is None) == (value is None):
        raise ValueError("give exactly one of fraction and value")
    if not hasattr(u0, field):
        raise AttributeError(f"the inputs have no field {field!r}")
    return getattr(u0, field) * (1.0 + fraction) if value is None else value


def step(field: str, fraction: float | None = None, value: float | None = None,
         t: float = 30.0) -> Disturbance:
    """Move one input at time t, by a fraction of its nominal value or to a value."""

    def d(time: float, u0):
        if time < t:
            return u0
        return replace(u0, **{field: _target(u0, field, fraction, value)})

    d.times = (float(t),)
    return d


def ramp(field: str, fraction: float | None = None, value: float | None = None,
         t: float = 30.0, duration: float = 60.0) -> Disturbance:
    """Move one input linearly from its nominal value over [t, t + duration]."""
    if duration <= 0:
        raise ValueError("a ramp needs a positive duration; use step for an instant change")

    def d(time: float, u0):
        if time < t:
            return u0
        start = getattr(u0, field)
        end = _target(u0, field, fraction, value)
        w = min((time - t) / duration, 1.0)
        return replace(u0, **{field: start + w * (end - start)})

    d.times = (float(t), float(t + duration))  # the ramp's slope changes at both ends
    return d


def combine(*disturbances: Disturbance) -> Disturbance:
    """Apply several disturbances in order; later ones see the earlier ones' changes."""

    def d(time: float, u0):
        u = u0
        for dist in disturbances:
            u = dist(time, u)
        return u

    d.times = tuple(sorted({t for dist in disturbances for t in getattr(dist, "times", ())}))
    return d
