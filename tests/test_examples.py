"""The examples run as written, so the documentation that points to them stays true."""

from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

EXAMPLES = pathlib.Path(__file__).resolve().parents[1] / "examples"


@pytest.mark.parametrize("script", sorted(p.name for p in EXAMPLES.glob("0*.py")))
def test_example_runs(script, tmp_path):
    args = [str(tmp_path / "data")] if "dataset" in script else []
    out = subprocess.run([sys.executable, str(EXAMPLES / script), *args], cwd=tmp_path,
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip()
