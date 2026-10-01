"""The measurement and actuator layer, and the guarantee that it is inert when unused.

The first three tests are the ones that matter for the published results: an ideal
structure must carry exactly the augmented states it always carried, in the same order,
and must give back the same eigenvalues.  If they fail, the seams have leaked into the
model and every reported number is downstream of the leak.
"""

from __future__ import annotations

import numpy as np
import pytest

from plantbench.cases.reactor_separator_recycle import plant
from plantbench.cases.reactor_separator_recycle import scenarios as sc
from plantbench.cases.reactor_separator_recycle.parameters import PlantParameters
from plantbench.core import control as ctl
from plantbench.core.instruments import Actuator, Instrument


@pytest.fixture(scope="module")
def design():
    return plant.solve_design(PlantParameters(), V_boil=28.0)


# ------------------------------------------------------------------------------------
# The seams are inert when unused
# ------------------------------------------------------------------------------------


def test_ideal_structures_carry_one_state_per_element(design):
    """The layout the model has always had, which is what fixes the published numbers."""
    for name, build in sc.STRUCTURES.items():
        structure = build(design)
        assert ctl.n_augmented(structure) == len(structure), name
        assert np.array_equal(
            ctl.initial_augmented(structure, design), np.zeros(len(structure))
        ), name


def test_attaching_an_ideal_instrument_changes_nothing(design):
    """A seam that is present but unconfigured must not move the spectrum."""
    structure = sc.plantwide(design)
    seamed = ctl.with_instruments(
        structure,
        instruments={"reactor C_A": Instrument()},
        actuators={"reactor T": Actuator()},
    )
    assert ctl.n_augmented(seamed) == ctl.n_augmented(structure)
    base = np.sort_complex(ctl.closed_loop_spectrum(design, structure))
    seam = np.sort_complex(ctl.closed_loop_spectrum(design, seamed))
    assert np.allclose(base, seam, rtol=0, atol=1e-12)


def test_unknown_loop_name_is_rejected(design):
    with pytest.raises(KeyError):
        ctl.with_instruments(sc.plantwide(design), instruments={"no such loop": Instrument()})


# ------------------------------------------------------------------------------------
# Effect on the plant
# ------------------------------------------------------------------------------------


def _dominant(design, structure):
    """The least damped oscillatory mode: its damping ratio and its eigenvalue."""
    lam = ctl.closed_loop_spectrum(design, structure)
    osc = lam[np.abs(lam.imag) > 0.05]
    z = -osc.real / np.abs(osc)
    k = int(np.argmin(z))
    return float(z[k]), osc[k]


def test_analyzer_deadtime_adds_states(design):
    structure = sc.plantwide(design)
    assert ctl.n_augmented(sc.analyzer_deadtime(structure, 5.0, order=1)) == len(structure) + 1
    assert ctl.n_augmented(sc.analyzer_deadtime(structure, 5.0, order=2)) == len(structure) + 2


def _track(design, structure, previous):
    """Follow one mode as a parameter changes, by taking the eigenvalue nearest the last.

    Selecting instead by "least damped" jumps between modes as the delay grows, because
    which mode is least damped changes; that jumping is what makes the damping ratio look
    erratic rather than the modes themselves behaving erratically.
    """
    lam = ctl.closed_loop_spectrum(design, structure)
    upper = lam[lam.imag > 0.0]
    return upper[int(np.argmin(np.abs(upper - previous)))]


def test_deadtime_slows_the_dominant_mode(design):
    """The monotone effect of deadtime, once the mode is followed rather than re-selected."""
    structure = sc.plantwide(design)
    _, mode = _dominant(design, structure)
    mode = mode if mode.imag > 0 else np.conj(mode)
    for theta in (0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0):
        nxt = _track(design, sc.analyzer_deadtime(structure, theta, order=2), mode)
        assert nxt.imag < mode.imag  # the oscillation gets slower at every step
        mode = nxt


def test_damping_ratio_hides_a_slower_decay(design):
    """The metric's blind spot, on the loop where deadtime actually belongs.

    At a ten-minute analyzer cycle the dominant mode decays about three times more slowly
    than it does with an instantaneous measurement, which is unambiguously worse.  Its
    damping ratio is nonetheless higher, because the mode has slowed down and zeta is
    normalized by the modulus.  Any design screening that reads zeta alone would call this
    an improvement.
    """
    structure = sc.plantwide(design)
    z_ideal, mode = _dominant(design, structure)
    mode = mode if mode.imag > 0 else np.conj(mode)

    slow = mode
    for theta in (1.0, 2.0, 3.0, 5.0, 8.0, 10.0):
        slow = _track(design, sc.analyzer_deadtime(structure, theta, order=2), slow)
    z_slow = -slow.real / abs(slow)

    assert slow.real > 2.0 * mode.real  # decay at least twice as slow (both are negative)
    assert z_slow > z_ideal  # and yet the damping ratio reads better


def test_non_ideal_plant_still_starts_at_rest(design):
    """A non-ideal structure must not manufacture a transient of its own."""
    structure = sc.analyzer_deadtime(sc.plantwide(design), minutes=5.0)
    structure = ctl.with_instruments(
        structure, actuators={"reactor T": Actuator(tau=0.5, rate_limit=2.0)}
    )
    y0 = np.concatenate([design.x, ctl.initial_augmented(structure, design)])
    r = ctl.closed_loop_rhs(0.0, y0, structure, design.u, design.pp, None, design.rhs)
    assert np.max(np.abs(r)) < 1e-8


def test_rate_limit_bounds_the_valve_slew(design):
    """A slew-limited valve cannot move faster than its limit, whatever is asked of it."""
    limit = 0.05  # m3/min per min on the jacket coolant valve
    structure = ctl.with_instruments(
        sc.plantwide(design), actuators={"reactor T": Actuator(tau=0.2, rate_limit=limit)}
    )
    t, X, U = ctl.simulate(
        design, structure, t_end=60.0, disturbance=sc.throughput_step(0.10, "F_out"),
        n_points=601,
    )
    slew = np.abs(np.diff(U["Fj"]) / np.diff(t))
    assert slew.max() <= limit * 1.05  # the sampled difference allows a little headroom
