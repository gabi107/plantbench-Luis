# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**On Windows use `.venv\Scripts\python.exe` (or `.venv/Scripts/python` from Bash).**

```bash
# Install
python -m venv .venv
.venv/Scripts/pip install -e ".[dev]"          # library + tests
.venv/Scripts/pip install -e ".[dev,study]"    # add study extras (scikit-learn, PaCMAP, HDBSCAN)

# Tests
.venv/Scripts/python -m pytest -q              # ~90 s; excludes @slow; includes golden test of reference.txt
.venv/Scripts/python -m pytest -q -m slow      # golden test of designs.txt, ~150 s
.venv/Scripts/python -m pytest -q tests/cases/test_contract.py          # contract tests for all cases
.venv/Scripts/python -m pytest -q tests/core/test_control.py::test_name  # single test

# Lint
.venv/Scripts/python -m ruff check plantbench tests studies

# CLI
python -m plantbench list
python -m plantbench describe reactor_separator_recycle
python -m plantbench run case0 --config c.toml
python -m plantbench generate examples/04_dataset.toml --workers 8
python -m plantbench new-case my_tank   # scaffold a new case package

# Studies (study extra required, run in this order)
.venv/Scripts/python -m studies.reactor_separator_recycle.reference
.venv/Scripts/python -m studies.reactor_separator_recycle.designs
.venv/Scripts/python -m plantbench generate studies/regimes/spec.toml --workers 10
.venv/Scripts/python -m studies.regimes.analysis data/regimes
.venv/Scripts/python -m studies.design.analysis --workers 10
.venv/Scripts/python -m studies.acc_figures --designs-cache cache
```

## Architecture

### Layer hierarchy (enforced by `tests/test_architecture.py`)

```
plantbench/core/       control layer, instruments, disturbances, the case interface, specs
plantbench/units/      unit-operation models and frozen parameter dataclasses
plantbench/heat/       process streams and pinch analysis
plantbench/cases/      the cases (each imports only core/units/heat, never another case)
plantbench/datagen.py  dataset generation (composes cases and core, parallel + resumable)
plantbench/datasets.py reading datasets: summaries, provenance, comparison
plantbench/task.py     what a method predicts from a dataset and how it is scored
plantbench/sensitivity.py  Sobol' first-order and total indices
studies/               scripts that use the library; the library never imports studies
```

Imports run strictly one way. `core`, `units`, and `heat` import nothing from each other or from cases. No case imports another case or any module above it (`datagen`, `datasets`, `task`, `sensitivity`). The tests assert this statically.

### Core flow

`load_case(id)` → `case.config(structure, ...)` → `build(case, config)` → `run(case, config, t_end)` → `Trajectory`

- **`Config`** is a frozen plain-data dataclass (TOML/JSON round-trips; hashed to identify a run). Sections: `structure`, `options`, `params` (dotted-path overrides), `setpoints`, `setpoint_steps`, `tuning`, `instruments`, `actuators`, `disturbance`.
- **`build`** designs the plant at a steady state and wires the control structure. Returns a `Setup` with `design`, `structure`, `disturbance`, and `spectrum()`.
- **`run`** integrates the closed loop with a stiff solver (SciPy), restarting at declared breakpoints. A non-finite state raises immediately.
- **`Trajectory`** holds `t`, `x` (states), `u` (inputs + derived signals), `y` (measurements), and `wall_time`.

### Control structures as data

A structure is a list of `Loop` and `Ratio` elements, each naming a measurement (`measure` callable), a manipulated variable (`mv` field name or `"sp:<loop>"`), tuning (`Kc`, `tau_I`), and range. Integral states are appended to the plant state vector, so the whole closed loop is one ODE. An `Instrument` on the measurement path or an `Actuator` on the valve adds its own states; an ideal (default-constructed) element adds none.

### Case discovery

Cases are found in three places, in this order: built-in (listed in `_MODULES` in `plantbench/cases/__init__.py`), installed entry point (`[project.entry-points."plantbench.cases"]`), or session-registered (`@plantbench.case` decorator or `plantbench.register(CASE)`). Built-ins and entry points are available to parallel datagen workers; session cases are re-imported per worker by module name.

### Dataset generation

`generate(spec_toml, workers=N)` writes to `data/<study name>/`: `spec.toml`, `manifest.json`, `runs.jsonl`, `index.csv`, and `runs/<id>.npz`. Generation is resumable: runs already in `runs.jsonl` (including failures) are skipped. A run ends as `ok`, `invalid` (config refused), `failed` (integration raised), or `timeout` (wall budget exceeded). Run ids are hashes of their configuration.

### Frozen cases

`reactor_separator_recycle` is frozen. Its reference config must reproduce `studies/reactor_separator_recycle/results/reference.txt` and `designs.txt` byte for byte (`tests/cases/reactor_separator_recycle/test_golden.py`). New behavior goes in an option that is **off by default**, or in a new case. Never edit a frozen case's reference config in a way that would change those files.

### Key conventions

- **Parameters are frozen dataclasses.** Every physical constant lives in a parameter set. Corrections to published values are noted in a comment. Dotted-path overrides fail on unknown fields.
- **Declared but unimplemented raises.** `Instrument.noise_std`, `Actuator.stiction`, etc. raise rather than silently no-op, to prevent false conclusions in studies.
- **RHS is continuous.** Anti-windup is back-calculation; deadtime is Padé. Discontinuities in time are declared as breakpoints in the disturbance, not embedded in the RHS.
- **Tests assert physics, not stored outputs.** Conservation laws, mole-fraction sums, limit checks. Golden tests are reserved for published results.
- **Generalize on the second use.** Code moves to a shared layer only when a second case needs it. Don't abstract from a single example.

### Adding a new case

1. `python -m plantbench new-case <name>` (or copy `plantbench/cases/_template/`).
2. Implement: frozen parameter dataclass, inputs dataclass, `Design` with `rhs`, structures via `control.bias_from_design`, named disturbances, `Case` in `definition.py`.
3. Register in `_MODULES` in `plantbench/cases/__init__.py`.
4. Pass `tests/cases/test_contract.py` (fixed point, round-trip, ideal instruments, short run at rest).
5. Add case-specific tests under `tests/cases/<name>/`.
6. Write `plantbench/cases/<name>/README.md` (case card).

Case naming: lowercase words joined by single underscores, beginning with a letter, at most 40 characters, not the word `case`. The name is permanent once a dataset is published against it (it enters the spec digest).
