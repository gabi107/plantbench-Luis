"""Problem-table algorithm and utility placement, checked against a case worked by hand.

Hand case, dT_min = 10 K:
    hot  H1  200 -> 100 K, CP 2   (duty 200)
    cold C1  110 -> 210 K, CP 1   (duty 100)
Shifted, hot 195 -> 95, cold 115 -> 215.  Intervals from the top:
    215-195  cold only        surplus -20
    195-115  CP 2 - 1         surplus +80
    115-95   hot only         surplus +40
Cascade from zero: -20, +60, +100, so QH_min = 20, flows 20, 0, 80, 120, QC_min = 120,
pinch at shifted 195 (hot 200 K, cold 190 K).

Utility levels:
  steam at 205 K (shifted 200): the cascade flow there is 5, so it supplies 5 and the
  remaining 15 must come from a hotter level;
  cooling water placed at 105 K (shifted 110): the flow there is 90, so the last 30 kJ,
  released below shifted 110 by the hot stream alone (CP 2 over 15 K), needs a colder
  utility.
"""

import numpy as np
import pytest

from plantbench.heat import pinch
from plantbench.heat.streams import Stream

HAND = [Stream("H1", 200.0, 100.0, 2.0), Stream("C1", 110.0, 210.0, 1.0)]


def test_hand_case_targets():
    pt = pinch.problem_table(HAND, 10.0)
    assert pt.QH_min == pytest.approx(20.0)
    assert pt.QC_min == pytest.approx(120.0)
    assert pt.pinch_shifted == pytest.approx(195.0)
    assert pt.pinch_hot == pytest.approx(200.0)
    assert pt.pinch_cold == pytest.approx(190.0)
    assert list(pt.flows) == pytest.approx([20.0, 0.0, 80.0, 120.0])


def test_hand_case_utility_placement():
    levels = [
        pinch.UtilityLevel("low steam", 205.0, hot=True),
        pinch.UtilityLevel("high steam", 300.0, hot=True),
        pinch.UtilityLevel("cooling water", 105.0, hot=False),
        pinch.UtilityLevel("chilled water", 50.0, hot=False),
    ]
    pt = pinch.problem_table(HAND, 10.0, extra_boundaries=(200.0, 295.0, 110.0, 55.0))
    q = pinch.place_utilities(pt, levels)
    assert q["low steam"] == pytest.approx(5.0)
    assert q["high steam"] == pytest.approx(15.0)
    assert q["cooling water"] == pytest.approx(90.0)
    assert q["chilled water"] == pytest.approx(30.0)


def test_extra_boundaries_do_not_change_targets():
    a = pinch.problem_table(HAND, 10.0)
    b = pinch.problem_table(HAND, 10.0, extra_boundaries=(150.0, 170.0, 200.0))
    assert (b.QH_min, b.QC_min) == pytest.approx((a.QH_min, a.QC_min))


def test_utility_difference_equals_duty_difference():
    rng = np.random.default_rng(3)
    for _ in range(20):
        streams = []
        for k in range(6):
            Ts, Tt = rng.uniform(300, 500, 2)
            streams.append(Stream(f"s{k}", Ts, Tt, rng.uniform(0.5, 5)))
        pt = pinch.problem_table(streams, 10.0)
        hot = sum(s.duty for s in streams if s.is_hot)
        cold = sum(s.duty for s in streams if not s.is_hot)
        assert pt.QH_min - pt.QC_min == pytest.approx(cold - hot, rel=1e-9, abs=1e-6)
        assert pt.flows.min() == pytest.approx(0.0, abs=1e-6)


def test_threshold_problem_has_no_pinch():
    pt = pinch.problem_table([Stream("C", 300.0, 350.0, 1.0)], 10.0)
    assert pt.QH_min == pytest.approx(50.0)
    assert pt.QC_min == pytest.approx(0.0)
    assert pt.pinch_shifted is None


def test_too_cold_hot_utility_is_rejected():
    pt = pinch.problem_table(HAND, 10.0)
    with pytest.raises(ValueError):
        pinch.place_utilities(pt, [pinch.UtilityLevel("low steam", 205.0, hot=True),
                                   pinch.UtilityLevel("cooling water", 105.0, hot=False),
                                   pinch.UtilityLevel("chilled water", 50.0, hot=False)])


def test_composite_curves_close_on_duties():
    H, T = pinch.composite_curve(HAND, hot=True)
    assert H[-1] == pytest.approx(200.0) and T[0] == 100.0 and T[-1] == 200.0
    H, T = pinch.composite_curve(HAND, hot=False)
    assert H[-1] == pytest.approx(100.0)
