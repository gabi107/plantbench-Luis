"""The outcome of a run, read from the end of its trajectory.

Written before any trajectory of this study was generated, and not changed after the
embeddings were inspected: the labels validate the unsupervised analysis, so they must
not be tuned to agree with it.

Over the last 300 min of a 1500 min run:

    limit cycle   the reactor temperature swings by more than 0.5 K peak to peak, and
                  the swing over the last 150 min is at least half that over the 150 min
                  before, so the oscillation is sustained rather than decaying
    unsettled     the swing exceeds 0.05 K without being sustained: a slow decay or drift
    saturated     neither of those, but a feedback loop's manipulated variable sits on a
                  limit over the last 30 min: the plant has settled with that loop lost
    settled       none of the above

The thresholds are set against the reference results of `reactor_separator_recycle`: the designs that settle
leave a reactor temperature swing of 0.004 K over the last 300 min, and D4 under a 10%
throughput step one of 15.8 K.
"""

from __future__ import annotations

import numpy as np

LATE = 300.0  # min, the window the outcome is read from
LIMIT_CYCLE_SWING = 0.5  # K
SUSTAINED = 0.5  # swing ratio, last half of the late window to first half
UNSETTLED_SWING = 0.05  # K
SATURATION_WINDOW = 30.0  # min
AT_LIMIT = 1e-6  # relative to the loop's range

OUTCOMES = ("settled", "saturated", "unsettled", "limit cycle")


def outcome(t: np.ndarray, T: np.ndarray, mvs: dict[str, tuple[np.ndarray, float, float]]) -> str:
    """The outcome of a run.

    t, T   time and reactor temperature
    mvs    loop name -> (manipulated variable, low limit, high limit), feedback loops only
    """
    t_end = t[-1]
    late = t >= t_end - LATE
    first = late & (t < t_end - LATE / 2)
    second = t >= t_end - LATE / 2
    swing = float(np.ptp(T[late]))
    if swing > LIMIT_CYCLE_SWING and np.ptp(T[second]) >= SUSTAINED * np.ptp(T[first]):
        return "limit cycle"
    if swing > UNSETTLED_SWING:
        return "unsettled"
    end = t >= t_end - SATURATION_WINDOW
    for values, lo, hi in mvs.values():
        span = (hi - lo) if np.isfinite(hi - lo) else max(abs(lo), 1.0)
        v = values[end]
        if np.all(v <= lo + AT_LIMIT * span) or np.all(v >= hi - AT_LIMIT * span):
            return "saturated"
    return "settled"
