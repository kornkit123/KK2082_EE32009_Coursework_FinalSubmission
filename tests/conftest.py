"""Shared pytest setup: make the modules in Script/ importable and provide fixtures."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'Script'))


@pytest.fixture
def rng():
    """A seeded random generator, so every test run is identical."""
    return np.random.default_rng(0)
