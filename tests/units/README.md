# tests/units/

Tests for `plantbench/units/`.

| File | Contents |
|---|---|
| `test_reactor.py` | **Gating tests.** The CSTR model must reproduce the reference steady-state concentrations, temperatures and eigenvalues from Romagnoli & Palazoglu (2020) Table 4.2 to the published number of significant figures. These tests must pass before any change to `plantbench/units/cstr.py` or `plantbench/units/parameters.py` is committed. Also checks mass and energy conservation at the design point. |
