# Examples

The scripts in `examples/` are short and self-contained, and the test suite runs each one as written. Run them from the repository root.

## Running a case

The reference plant of `jacketed_cstr`, and the same feed-temperature disturbance with the temperature loop open.

```{literalinclude} ../examples/01_run_case.py
:language: python
```

## Set-points and tuning

A set-point step on the reactor temperature under three tunings, with the damping ratio from the linearized plant against the overshoot of the nonlinear run.

```{literalinclude} ../examples/02_setpoints_and_tuning.py
:language: python
```

## Analyzer deadtime

Deadtime on the composition analyzer of `reactor_separator_recycle`, and its effect on the rightmost closed-loop eigenvalue.

```{literalinclude} ../examples/03_analyzer_deadtime.py
:language: python
```

## Generating a dataset

A dataset of 48 runs of `jacketed_cstr`, generated from a specification and read back. The specification:

```{literalinclude} ../examples/04_dataset.toml
:language: toml
```

The script that generates it:

```{literalinclude} ../examples/04_generate_dataset.py
:language: python
```
