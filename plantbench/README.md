# plantbench/

The library package. Import it as `import plantbench as pb`.

## Top-level files

| File | Contents |
|---|---|
| `__init__.py` | Public API: re-exports `load_case`, `run`, `build`, `Case`, `Config`, `Setup`, `Trajectory`, `list_cases`, `register`, and `__version__`. The only stable surface callers should import from. |
| `__main__.py` | CLI entry point (`python -m plantbench`). Subcommands: `list`, `describe`, `run`, `generate`, `datasets`, `compare`, `card`, `verify`, `new-case`, `check`. |
| `contract.py` | The interface contract every case must satisfy (`CHECKS`): fixed point under each structure, configuration round-trip, ideal instruments adding no states, a short run from rest. Also the mechanical rules a case contributed to the library must pass (`check --contribute`). |
| `datagen.py` | Parallel, resumable dataset generation from a TOML specification. Writes `data/<study>/`: `spec.toml`, `manifest.json`, `runs.jsonl`, `index.csv`, `runs/<id>.npz`. |
| `datasets.py` | Reading datasets back from disk: summary tables, provenance stamps, comparing two generations or two results files. |
| `report.py` | Produces the dataset "card" (spec digest, library version, run counts) and implements the `verify` command that recomputes digests against a stored stamp. |
| `scaffold.py` | Implements `python -m plantbench new-case <name>`: copies `plantbench/cases/_template/` into a standalone package that declares the case as an entry point. |
| `sensitivity.py` | Variance-based Sobol' first-order and total sensitivity indices using the Saltelli/Jansen estimators. Used by `studies/design/` to attribute variance in the damping ratio. |
| `task.py` | Prediction tasks: which signals and time window a method sees, how runs are labelled and split into folds, and how an answer is scored. |

## Subdirectories

| Directory | Contents |
|---|---|
| `core/` | Plant-agnostic control layer, case interface, instruments, disturbances, and dataset specifications. No plant-specific code. |
| `units/` | Unit-operation models (CSTR, tray column) and their frozen parameter dataclasses. |
| `heat/` | Process streams and pinch analysis. |
| `cases/` | The built-in cases and the case registry. Each case is a subdirectory with its own model, parameters, structures and `definition.py`. |
