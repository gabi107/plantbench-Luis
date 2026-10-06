# plantbench/core/

The plant-agnostic control layer. Contains everything needed to build and simulate a closed-loop plant without knowing anything about a specific case. Cases import from here; nothing in `core` imports from `cases`, `units`, or `heat`.

## Files

| File | Contents |
|---|---|
| `case.py` | The central case interface. Defines `Config` (frozen plain-data configuration), `Setup` (configured plant at its design point), `Trajectory` (simulation output), and the functions `build` and `run` that every case flows through. Also provides `override`, which applies dotted-path parameter overrides to a frozen parameter dataclass. |
| `control.py` | PI control loops and the closed-loop simulator. `Loop` and `Ratio` are the data elements of a control structure. `bias_from_design` initialises biases and set-points from the design point. `closed_loop_rhs` assembles the augmented ODE (plant states + integral states + instrument/actuator states). `spectrum` computes closed-loop eigenvalues by finite-difference linearization. |
| `instruments.py` | Measurement and actuator non-idealities, each implemented as ODE states appended to the augmented vector. `Instrument` covers deadtime (Padé), first-order lag and (declared but not yet implemented) noise. `Actuator` covers first-order valve dynamics and (declared but not yet implemented) stiction. An ideal (default-constructed) element adds zero states and is transparent to the integrator. |
| `disturbances.py` | Generic disturbance functions that act on any inputs dataclass: `step` (instantaneous change), `ramp` (linearly ramped change), and named disturbances a case declares. A disturbance returns a modified copy of the inputs; it never mutates. |
| `spec.py` | Parses dataset specification TOML files and expands them into a list of `Config` objects. Handles `base`, `variants` (single list or named groups with Cartesian product), `sweep` (Cartesian product of value lists), and `sample` (Latin hypercube). Every expansion is deterministic in the seed. Paths address configuration sections with the depth rules in `_DEPTH`. |
| `casekit.py` | Reusable helpers for case authors: `StateLayout` (names a block of states inside the augmented vector), `state_measurement` (a measurement function that reads one state by name), `cached_design` (a decorator that caches the design solve against the parameter hash), and utilities for building measurement functions. |

## Key design decisions

- **Integral states are appended to the plant state vector.** The whole closed loop (plant + controllers + instruments) is one ODE, so `scipy.integrate.solve_ivp` handles it as a single stiff system and the Jacobian is available for eigenvalue analysis.
- **Anti-windup by back-calculation; deadtime by Padé.** Both keep the RHS smooth and differentiable, which is required for the stiff solver and for linearization.
- **Discontinuities are declared as breakpoints.** A disturbance or set-point schedule lists the times it acts at; `run` restarts the integrator there. This prevents the solver from stepping over an upset and landing on an unphysical state.
