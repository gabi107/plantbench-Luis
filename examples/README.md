# examples/

Runnable scripts that demonstrate the public API. Every script here is executed as a subprocess by `tests/test_examples.py`, so they serve as living documentation: if an example breaks, the test suite catches it.

## Files

| File | Contents |
|---|---|
| `01_run_case.py` | Loads `jacketed_cstr`, runs the reference (open-loop) plant and a PI-controlled plant, and prints trajectory statistics. The minimal end-to-end usage pattern. |
| `02_setpoints_and_tuning.py` | Shows that set-points and PI tuning are configuration data: builds two configurations with different tunings and set-point steps, runs both, and compares their responses. |
| `03_analyzer_deadtime.py` | Adds an analyzer deadtime (Padé approximation) to the composition loop of `reactor_separator_recycle` as an `Instrument`, and demonstrates the spectrum shift. |
| `04_dataset.toml` | TOML specification for a small dataset (a few dozen runs of `jacketed_cstr`). Used as input by `04_generate_dataset.py`. |
| `04_generate_dataset.py` | Generates a dataset from `04_dataset.toml`, reads it back from disk, and prints its manifest. Demonstrates the full generate → read workflow. |

To run an individual example from the repository root:

```bash
.venv/Scripts/python examples/01_run_case.py
```
