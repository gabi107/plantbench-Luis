# docs/

Source for the [plantbench documentation site](https://plantbench.readthedocs.io), built by Sphinx with the MyST Markdown parser. The site is hosted on Read the Docs; its build configuration is `.readthedocs.yaml` in the repository root.

## Files

| File | Contents |
|---|---|
| `conf.py` | Sphinx configuration: project metadata, extensions (`autodoc2`, `myst-parser`, `furo` theme), intersphinx targets. |
| `index.md` | Site front page and navigation tree. |
| `user-guide.md` | How to install the library, run cases, build configurations, use non-idealities and generate datasets. The primary reference for users. |
| `philosophy.md` | Design principles: what a case is, the layer hierarchy, the conventions every case follows, and why each convention exists. Required reading before modifying the library. |
| `adding-a-case.md` | Step-by-step checklist for adding a case to the library, including naming rules, the required contract, and how to contribute from a standalone package. |
| `developing-a-case.md` | How to develop a case in a package of its own: entry-point registration, the `@plantbench.case` decorator, and using the library without contributing back. |
| `examples.md` | Narrative walkthrough of the `examples/` scripts. |
| `api.md` | API reference index (populated by `autodoc2` from module docstrings). |
| `make_logo.py` | One-off script that generated the SVG logo files in `_static/`. Not part of the documentation build. |

## Subdirectories

| Directory | Contents |
|---|---|
| `_static/` | Logo SVGs (light and dark variants, mark-only and full wordmark) used by the site theme. |
| `cases/` | Per-case documentation pages, one per built-in case plus an index. These are authored alongside `plantbench/cases/<case>/README.md` (the case card). |
