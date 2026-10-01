"""`reactor_separator_recycle` reproduces its reference results byte for byte.

The committed results files are the source of every number reported for `reactor_separator_recycle`.  These tests regenerate them into a temporary directory and compare bytes,
so a change that moves any reported number fails here rather than surfacing later as a
wrong figure in the text.  If a change is meant to move a number, regenerate the file
with the script and commit it alongside the change, saying which numbers moved and why.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[3]
RESULTS = ROOT / "studies" / "reactor_separator_recycle" / "results"


def _run(module: str, *args: str) -> None:
    subprocess.run([sys.executable, "-m", f"studies.reactor_separator_recycle.{module}", *args],
                   check=True, capture_output=True, text=True, cwd=ROOT)


def _assert_identical(produced: pathlib.Path, reference: pathlib.Path) -> None:
    if produced.read_bytes() != reference.read_bytes():
        new, old = produced.read_text().splitlines(), reference.read_text().splitlines()
        first = next((i for i, (a, b) in enumerate(zip(new, old)) if a != b),
                     min(len(new), len(old)))
        pytest.fail(f"{reference.name} differs from line {first + 1}:\n"
                    f"  committed: {old[first] if first < len(old) else '<end>'}\n"
                    f"  produced:  {new[first] if first < len(new) else '<end>'}")


def test_results_reproduce(tmp_path):
    """The base plant and its structures: about 20 s."""
    _run("reference", "--out", str(tmp_path))
    _assert_identical(tmp_path / "reference.txt", RESULTS / "reference.txt")


@pytest.mark.slow
def test_heat_integration_results_reproduce(tmp_path):
    """The design comparison: about 150 s, most of it the poorly damped D4 runs."""
    _run("designs", "--out", str(tmp_path), "--cache", str(tmp_path), "--figures", str(tmp_path))
    _assert_identical(tmp_path / "designs.txt", RESULTS / "designs.txt")
