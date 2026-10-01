"""Fixtures for the core tests; the toy plant itself is in tests/_toy.py."""

from __future__ import annotations

import pytest
from _toy import TOY_CASE, toy_design


@pytest.fixture
def design():
    return toy_design()


@pytest.fixture
def case():
    return TOY_CASE
