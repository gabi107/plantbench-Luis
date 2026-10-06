# tests/cases/reactor_separator_recycle/

Case-specific tests for the `reactor_separator_recycle` built-in case. This is the most extensively tested case because it is frozen and its numbers appear in the ACC 2027 paper.

| File | Contents |
|---|---|
| `test_golden.py` | **The frozen-case guard.** Regenerates `studies/reactor_separator_recycle/results/reference.txt` and `designs.txt` and compares byte for byte against the committed versions. Marked `@slow`; run with `-m slow`. Any change that would move these files must regenerate them and document each changed value. |
| `test_plant.py` | Plantwide steady state, mass and energy conservation (component balances close to 1e-9, energy balances to 1e-6), and the two published structural results (recycle gain, open-loop unstable eigenvalue). |
| `test_hen.py` | Heat-exchanger network: energy balances around each exchanger, minimum approach temperatures, and the pinch-utility target as an off-design lower bound on the reboiler duty. |
| `test_integrated.py` | Stage-2 (heat-integrated) plant: rest states for each network, and consistency between the steady-state network rating and the design-point integration. |
| `test_instruments.py` | Ideal instruments and actuators add no states and reproduce the published eigenvalues exactly. Non-ideal elements shift the eigenvalues in the expected direction. |
| `test_streams.py` | Stream table: energy conservation around the whole plant (the sum of all stream duties, including utilities, closes to 1e-6). |
| `test_thermo.py` | Vapor-pressure model: `bubble_point` and `dew_point` reproduce the volatilities the column's constant-relative-volatility assumption already uses. |
| `test_pareto.py` | Pareto filtering, normalization, and the pinch target as an off-design bound on the operating profit. |
| `test_definition.py` | The `Case` interface: confirms that the options mechanism selects exactly the plants that the study scripts build directly, and that unknown options or structures raise. |
