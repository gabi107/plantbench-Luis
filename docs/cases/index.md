# Cases

Each case is a plant model with a verified ground truth and a design space. Its card states where the model comes from, its states and inputs, its options, structures, disturbances and measurements, and how it is verified.

| Case | States | Description |
|---|---|---|
| [`reactor_separator_recycle`](reactor_separator_recycle.md) | 134 (base plant) | A non-isothermal CSTR on the open-loop unstable branch of its multiplicity, in a recycle loop closed by a thirty-stage column with a purge; four heat-exchanger networks at three column pressures; four regulatory structures. Frozen. |
| [`jacketed_cstr`](jacketed_cstr.md) | 3 | The same reactor on its own, with three steady states and structures from open loop to a composition cascade. |

New cases are named and verified as described in [Adding a case](../adding-a-case.md).

```{toctree}
:hidden:

reactor_separator_recycle
jacketed_cstr
```
