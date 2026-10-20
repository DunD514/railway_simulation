"""Configuration loading helpers. The whole project reads a single config.yaml."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML configuration file and return it as a nested dict."""
    with open(path or DEFAULT_CONFIG, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def with_overrides(cfg: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Return a deep copy of ``cfg`` with dotted-key overrides applied.

    Example: ``with_overrides(cfg, {"learning.alpha": 0.2})``.
    """
    out = copy.deepcopy(cfg)
    for dotted, value in overrides.items():
        node = out
        keys = dotted.split(".")
        for k in keys[:-1]:
            node = node[k]
        node[keys[-1]] = value
    return out
