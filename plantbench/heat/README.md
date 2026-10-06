# plantbench/heat/

Process stream types and pinch analysis. This layer knows nothing about control or any specific case. It is used by `plantbench/cases/reactor_separator_recycle/` to size and rate heat-exchanger networks, but it is case-independent and can be used by any case that involves heat integration.

## Files

| File | Contents |
|---|---|
| `streams.py` | `Stream` dataclass: a process stream defined by its supply temperature, target temperature, and heat-capacity flow rate (CP). Streams are passed to the pinch analysis routines and to network sizing. |
| `pinch.py` | Pinch analysis using the problem-table algorithm. Functions: `composite_curves` (hot and cold composite curves from a list of streams), `pinch_temperature` (the pinch point), `utility_targets` (minimum hot and cold utility loads at a given minimum approach temperature ΔT_min), and `grand_composite_curve`. The utility targets provide an off-design lower bound on the reboiler duty used in `test_hen.py`. |
