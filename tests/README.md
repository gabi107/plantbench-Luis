# tests/

The verification suite. Run with:

```bash
.venv/Scripts/python -m pytest -q              # all tests except @slow (~90 s)
.venv/Scripts/python -m pytest -q -m slow      # adds the golden tests (~150 s)
.venv/Scripts/python -m pytest -q tests/core/test_control.py::test_name   # single test
```

pytest is configured in `pyproject.toml` with `--import-mode=importlib` and `pythonpath = ["tests", "."]`, so test basenames may repeat across directories and the `studies/` tree is importable from tests.

## Top-level files

| File | Contents |
|---|---|
| `_toy.py` | A minimal first-order scalar plant (`dx/dt = (Km + d - x) / tau`) used as a fast substitute for a real case in all core-layer tests. Shared via `tests/core/conftest.py`. |
| `test_architecture.py` | Enforces the import layering by static analysis (AST). Fails if any layer imports something above itself or if the library imports `studies`. |
| `test_datagen.py` | Dataset generation: each run is recorded exactly once, generation is resumable, failures (`invalid`/`failed`/`timeout`) are stored as results and skipped on resume. |
| `test_datasets.py` | Reading datasets from disk: self-description, provenance stamping, and comparing two generations or two results files. |
| `test_examples.py` | Runs every script in `examples/` as a subprocess, so examples are always tested against the current library. |
| `test_extending.py` | External-case workflow: entry-point discovery from an installed package, the `new-case` scaffold, worker-process generation, and `python -m plantbench check`. |
| `test_report.py` | The `card` command output and the `verify` command's digest checking. |
| `test_sensitivity.py` | Sobol' estimators on synthetic functions with known analytical first-order and total indices. |
| `test_task.py` | Task definitions: observations, labels, folds and scoring on constructed datasets. |

## Subdirectories

| Directory | Contents |
|---|---|
| `cases/` | Contract and case-specific tests for all built-in cases. |
| `core/` | Tests for every module in `plantbench/core/`. |
| `heat/` | Tests for `plantbench/heat/`. |
| `studies/` | Tests for the study-side utilities in `studies/regimes/`. |
| `units/` | Tests for `plantbench/units/`. |
