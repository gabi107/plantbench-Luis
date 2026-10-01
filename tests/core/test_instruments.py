"""The measurement and actuator elements on their own, without a plant."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from plantbench.core.instruments import Actuator, Instrument


def test_defaults_are_ideal():
    assert Instrument().is_ideal and Instrument().n_states == 0
    assert Actuator().is_ideal and Actuator().n_states == 0


def test_unimplemented_non_idealities_are_refused():
    """Declared but not implemented must raise, not be silently ignored."""
    with pytest.raises(NotImplementedError):
        Instrument(noise_std=0.1)
    with pytest.raises(NotImplementedError):
        Actuator(stiction=0.1)
    with pytest.raises(ValueError):
        Actuator(rate_limit=1.0)  # a rate limit with no travel time constant


# ------------------------------------------------------------------------------------
# The deadtime representations
# ------------------------------------------------------------------------------------


def _step_response(inst: Instrument, theta: float, t_end: float, n: int = 400):
    """Response of an instrument alone to a unit step in the process value at t = 0."""
    x0 = inst.initial(0.0)
    sol = solve_ivp(
        lambda t, z: inst.derivatives(z, 1.0),
        (0.0, t_end),
        x0,
        t_eval=np.linspace(0.0, t_end, n),
        rtol=1e-10,
        atol=1e-12,
    )
    return sol.t, np.array([inst.output(sol.y[:, i], 1.0) for i in range(sol.t.size)])


def test_instrument_starts_at_rest():
    """Every representation reports the process value exactly, with zero derivative."""
    for inst in (
        Instrument(deadtime=5.0, order=1),
        Instrument(deadtime=5.0, order=2),
        Instrument(deadtime=5.0, method="lags", order=4),
        Instrument(tau=2.0),
        Instrument(tau=2.0, deadtime=5.0),
    ):
        x0 = inst.initial(3.7)
        assert inst.output(x0, 3.7) == pytest.approx(3.7, abs=1e-12)
        assert np.allclose(inst.derivatives(x0, 3.7), 0.0, atol=1e-12)


def _transfer(inst: Instrument, s: complex) -> complex:
    """Transfer function of an instrument, read off its own linear realization."""
    n = inst.n_states
    A = np.column_stack([inst.derivatives(np.eye(n)[:, j], 0.0) for j in range(n)])
    B = inst.derivatives(np.zeros(n), 1.0)
    C = np.array([inst.output(np.eye(n)[:, j], 0.0) for j in range(n)])
    D = inst.output(np.zeros(n), 1.0)
    return C @ np.linalg.solve(s * np.eye(n) - A, B) + D


def test_pade_matches_a_delay_in_the_frequency_domain():
    """The claim a Pade approximation makes, and the only one it makes.

    It is all-pass --- the magnitude is one at every frequency, exactly --- and its phase
    tracks that of exp(-s theta) while the frequency is low against the delay.  It is a
    phase approximation, so its step response is poor: it jumps the wrong way and shows an
    excursion no delay has.  That is the trade `method="lags"` exists for, and it is why
    this is tested here rather than on a step.
    """
    theta = 5.0
    for order, tol in ((1, 0.08), (2, 0.01)):
        inst = Instrument(deadtime=theta, order=order)
        for w_theta in (0.2, 0.5, 1.0):
            w = w_theta / theta
            G = _transfer(inst, 1j * w)
            assert abs(G) == pytest.approx(1.0, abs=1e-10)
            assert np.angle(G) == pytest.approx(np.angle(np.exp(-1j * w * theta)), abs=tol)


def test_pade_step_response_settles_but_is_not_a_delay():
    """Recorded so the poor step shape is known to be expected, not a defect."""
    t, y = _step_response(Instrument(deadtime=5.0, order=2), 5.0, t_end=40.0)
    assert y[-1] == pytest.approx(1.0, abs=1e-3)  # it does settle on the step
    assert y.min() < 0.0  # and it does swing the wrong way on the way there


def test_lag_chain_converges_to_a_delay():
    """More lags put the half-rise closer to the delay itself."""
    theta = 5.0

    def half_rise(order):
        t, y = _step_response(
            Instrument(deadtime=theta, method="lags", order=order), theta, t_end=60.0, n=4000
        )
        return t[np.argmax(y >= 0.5)]

    coarse, fine = half_rise(2), half_rise(16)
    assert abs(fine - theta) < abs(coarse - theta)
    assert fine == pytest.approx(theta, rel=0.2)


def test_lag_chain_does_not_undershoot():
    """Unlike Pade, a chain of lags is monotone, which is why it is offered."""
    t, y = _step_response(Instrument(deadtime=5.0, method="lags", order=6), 5.0, t_end=40.0)
    assert y.min() >= -1e-9


def test_bad_configuration_is_rejected():
    with pytest.raises(ValueError):
        Instrument(method="nonsense")
    with pytest.raises(ValueError):
        Instrument(deadtime=5.0, order=3)  # Pade is implemented for order 1 and 2
