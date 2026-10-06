# tests/studies/

Tests for the study-side utility modules in `studies/regimes/`. These test the data-mining pipeline components in isolation, using synthetic data with known structure rather than running the full dataset generation.

| File | Contents |
|---|---|
| `test_dataset.py` | Tests reading a small generated dataset through `studies.regimes.dataset`: run signals are loaded correctly, design-space coordinates match the configuration, and outcome labels are consistent with the trajectory tail. Also tests scoring a method's predictions against multiple plant definitions. |
| `test_discovery.py` | Tests the subspace greedy search (SGS) step of `studies.regimes.discovery` on synthetic two-cluster data: verifies that SGS identifies the separating variables when the clusters are linearly separable and that the Davies–Bouldin index decreases monotonically as the correct variables are added. |
