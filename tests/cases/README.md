# tests/cases/

Tests that run against the built-in cases. Organized as one directory per case, plus the cross-case contract test.

## Files

| File | Contents |
|---|---|
| `test_contract.py` | Runs `plantbench.contract.CHECKS` against every case returned by `list_cases()`. Checks that each case: has a valid id, provides a fixed point under every structure (residual < 1e-9), round-trips its reference config through TOML/JSON, produces finite measurements, leaves the state count unchanged with ideal instruments, and completes a short run from rest. Any new case must pass these checks. |

## Subdirectories

| Directory | Contents |
|---|---|
| `jacketed_cstr/` | Tests specific to the `jacketed_cstr` case: steady-state values, eigenvalues, conservation. |
| `reactor_separator_recycle/` | Tests specific to the `reactor_separator_recycle` case: conservation, HEN physics, heat integration, the interface contract, and the golden byte-for-byte comparison against committed results files. |
