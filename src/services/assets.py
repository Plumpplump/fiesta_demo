"""Resolve bundled asset paths in dev and in a packaged Flet build."""

from __future__ import annotations

import os
from pathlib import Path


def src_dir() -> Path:
    here = Path(__file__).resolve()
    return here.parents[1]


def assets_dir() -> Path:
    """``FLET_ASSETS_DIR`` when packaged, ``src/assets`` in development."""
    env = os.environ.get("FLET_ASSETS_DIR")
    if env:
        candidate = Path(env)
        if candidate.is_dir():
            return candidate
    candidate = src_dir() / "assets"
    if candidate.is_dir():
        return candidate
    raise FileNotFoundError("assets folder not found (expected src/assets)")


def models_dir() -> Path:
    return assets_dir() / "models"


def model_file(name: str) -> Path:
    return models_dir() / name


def has_model(name: str) -> bool:
    return model_file(name).is_file()