"""Sphinx configuration for the plantbench documentation.

    .venv/bin/pip install -e ".[docs]"
    .venv/bin/sphinx-build -W --keep-going -b html docs docs/_build/html
"""

from importlib.metadata import version as _version

project = "plantbench"
author = "Kyle Territo, Luis A. Briceno-Mena and Jose A. Romagnoli"
copyright = f"2026, {author}"
release = _version("plantbench")
version = ".".join(release.split(".")[:2])

extensions = [
    "myst_parser",
    "autodoc2",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
]

# autodoc2 writes its own index; the API page in the toctree is api.md.
exclude_patterns = ["_build", "apidocs/index.rst"]

# The guides and the case cards are Markdown, read by GitHub as well as here.
myst_enable_extensions = ["colon_fence", "dollarmath"]
myst_heading_anchors = 3

# The API reference is read from the source without importing it, so the optional
# dependencies of plantbench.task need not be installed.  The docstrings are written in
# Markdown (backticks for code, indented code blocks) and are parsed as such.
autodoc2_packages = [
    {
        "path": "../plantbench",
        "exclude_dirs": ["__pycache__", "_template"],
        "exclude_files": ["__main__.py"],
    },
]
autodoc2_docstring_parser_regexes = [(r".*", "myst")]
autodoc2_hidden_objects = ["private", "dunder", "inherited"]
autodoc2_render_plugin = "myst"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}

html_theme = "furo"
html_title = f"plantbench {release}"
html_static_path = ["_static"]
html_favicon = "_static/logo-mark.svg"
html_theme_options = {
    "light_logo": "logo-mark.svg",
    "dark_logo": "logo-mark-dark.svg",
}
