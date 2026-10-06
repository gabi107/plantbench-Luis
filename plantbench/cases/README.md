# plantbench/cases/

The case registry and all built-in cases. A case is a plant model offered as a test problem: a right-hand side, a verified steady state, a set of control structures and a design space selectable through a `Config`.

## Registry (`__init__.py`)

Cases are found in three places, checked in this order:

1. **Built-in** — listed in `_MODULES` (this file). Imported lazily on first `load_case`.
2. **Entry point** — declared in an installed package under `[project.entry-points."plantbench.cases"]`.
3. **Session** — registered in the running interpreter with `@plantbench.case` or `plantbench.register(CASE)`.

`list_cases()` returns all three. `load_case(id)` resolves the id and returns the `Case`.

## Built-in cases

| Directory | Case id | States | Description |
|---|---|---|---|
| `reactor_separator_recycle/` | `reactor_separator_recycle` | 134 | Non-isothermal CSTR on the unstable branch, in a recycle loop with a 30-stage column; four HEN designs; four regulatory structures. Frozen — reproduces Romagnoli & Palazoglu (2020). |
| `jacketed_cstr/` | `jacketed_cstr` | 3 | The same reactor in isolation; three steady states; structures from open loop to a composition cascade. |

## Template

`_template/` is a minimal working case (a gravity-drained tank) used as the starting point for `python -m plantbench new-case`. Its `README.md` is the case-card template a new case fills in. Copy it with:

```bash
python -m plantbench new-case my_tank
```

or, working inside the library:

```bash
cp -r plantbench/cases/_template plantbench/cases/my_tank
```

## Adding a case

See `docs/adding-a-case.md` for the full checklist. After implementing the model:

1. Register the case's `definition` module in `_MODULES` in this file's `__init__.py`.
2. Pass `tests/cases/test_contract.py`.
3. Add case-specific tests under `tests/cases/<name>/`.
4. Write a case card at `plantbench/cases/<name>/README.md`.

**Case naming:** lowercase words joined by underscores, beginning with a letter, at most 40 characters, not the word `case`. The name is permanent once any dataset is published against it.
