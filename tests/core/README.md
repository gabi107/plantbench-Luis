# tests/core/

Tests for every module in `plantbench/core/`. All tests in this directory use the toy plant defined in `tests/_toy.py` (a scalar first-order system) rather than any real case, so they run in milliseconds.

## Files

| File | Contents |
|---|---|
| `conftest.py` | Pytest fixtures shared by all core tests: builds the toy design and a reference `Setup` once per session. |
| `test_case.py` | The case interface on the toy plant: configuration construction, `override` (dotted-path parameter replacement), `build`, `run`, `Trajectory` accessors (`state`, `window`, `to_npz`/`from_npz`), and error handling for unknown fields or non-finite states. |
| `test_control.py` | PI control loops: integral-state layout, set-point schedules (`setpoint_steps`), tuning overrides, input recording including derived signals, cascade wiring, and the closed-loop eigenvalue via `spectrum`. |
| `test_disturbances.py` | Disturbance functions: `step` and `ramp` produce the expected input at each time without mutating the nominal inputs. Named disturbances declared by the toy case raise on an unknown name. |
| `test_instruments.py` | `Instrument` and `Actuator` in isolation: deadtime (Padé), first-order lag, and first-order valve dynamics each produce their expected step response. Ideal elements are transparent (zero additional states, identity transfer). Declared-but-unimplemented fields raise. |
| `test_spec.py` | Specification parsing and expansion: base + variants + sweep + sample all expand to the expected deterministic list of `Config` objects. Unknown paths raise. The digest is stable across Python versions. |
