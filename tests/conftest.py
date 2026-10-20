"""Shared fixtures."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.config import load_config  # noqa: E402
from sim.corridor import build_corridor  # noqa: E402
from sim.trains import load_train_classes  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    """Project configuration."""
    return load_config()


@pytest.fixture(scope="session")
def corridor(cfg):
    """Corridor geometry."""
    return build_corridor(cfg)


@pytest.fixture(scope="session")
def classes(cfg):
    """Train classes."""
    return load_train_classes(cfg)
