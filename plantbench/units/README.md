# plantbench/units/

Unit-operation models and their frozen parameter dataclasses. This layer knows nothing about control structures, cases, or datasets. It is imported by cases that need these unit models; `core` and `heat` do not import from here.

## Files

| File | Contents |
|---|---|
| `parameters.py` | Reference parameter sets for the CSTR, the distillation column, and thermodynamics. Each set is a frozen dataclass. Values corrected from their published source (Romagnoli & Palazoglu 2020) carry a comment with the correction and its evidence. Three such corrections are documented. |
| `cstr.py` | Non-isothermal CSTR model. Provides two RHS forms: `cstr_rhs` (constant-volume, single reaction A→B with inert I, energy balance on both reactor and cooling jacket) and a multicomponent form. Molar hold-up is fixed; the model is on the open-loop-unstable branch at the design point used by `reactor_separator_recycle` and `jacketed_cstr`. |
| `column.py` | Tray-by-tray dynamic distillation column model for three components. Uses Francis weir hydraulics for tray-to-tray liquid flows and constant molar overflow. The 30-tray column of `reactor_separator_recycle` is an instance of this model. States are liquid mole fractions and hold-ups on each tray plus the condenser and reboiler. |
| `thermo.py` | Vapor-pressure temperature model consistent with the constant relative volatility assumption used by the column. `bubble_point` inverts the Raoult's-law condition to recover temperature from composition and pressure; `dew_point` and `y_from_x` complete the VLE. These functions are used inside the column RHS and in the stream table. |
