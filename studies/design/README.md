# studies/design/

Sensitivity study: how the damping ratio of the `reactor_separator_recycle` plant responds to the analyzer deadtime on the composition loop, and which design parameters drive that response. Reports Table IV and Section VI of the ACC 2027 paper.

## Reproduction

```bash
# Step 1: sweep deadtime vs. damping ratio and compute first Sobol' sample (~9 h ÷ workers)
.venv/Scripts/python -m studies.design.analysis --workers 10

# Step 2: generate the Sobol' outcome sample (integration-heavy; divided by workers)
.venv/Scripts/python -m studies.design.verify --workers 10

# Step 3: compute Sobol' indices from the outcome sample
.venv/Scripts/python -m studies.design.analysis --workers 10
```

Run `analysis.py` twice: the first pass generates `sobol-outcome.json` if it does not exist; the second reads it back and computes the indices. `verify.py` runs the Sobol' sample configurations through the integrator and writes the outcomes.

## Files

| File | Contents |
|---|---|
| `analysis.py` | Sweeps analyzer deadtime (0–10 min) against the damping ratio for the D4 design. Also reads `sobol-outcome.json` and computes Sobol' first-order and total indices over the five-dimensional box defined by `SOBOL_BOUNDS` (pump-around share, column pressure, K_c, τ_I, deadtime). Writes `results/results_design.txt`. |
| `verify.py` | Generates the Sobol' sample configurations and runs them through `plantbench generate`, writing each outcome to `results/sobol-outcome.json`. This is the integration-heavy step. |

## results/

| File | Contents | Paper location |
|---|---|---|
| `results_design.txt` | Deadtime sweep (section 1) and Sobol' indices (section 2). | Table IV; Section VI |
| `sobol-outcome.json` | Raw integration outcomes for the Sobol' sample; input to the index computation in `analysis.py`. | (intermediate) |
