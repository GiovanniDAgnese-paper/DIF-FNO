"""Shared fixtures and helpers for the DIF-FNO test suite."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

# Make ``src/`` importable when the package is not installed (e.g. bare CI checkouts).
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

torch.set_default_dtype(torch.float32)


@pytest.fixture(scope="session", autouse=True)
def _deterministic():
    """Reproducible tests: fixed seeds and deterministic algorithms where possible."""
    from dagnese_fno import set_seed

    set_seed(0, deterministic=True, warn_only=True)
    torch.set_num_threads(1)
    yield


@pytest.fixture
def grid32():
    """Uniform 32 x 32 reference grid on [-1, 1]^2 with its spacing."""
    n = 32
    t = torch.linspace(-1, 1, n)
    coords = torch.stack(torch.meshgrid(t, t, indexing="ij"), dim=-1)  # (n, n, 2)
    return coords, 2.0 / (n - 1)


@pytest.fixture
def batched_grid():
    """Batched version of :func:`grid32`: ``(B=2, 32, 32, 2)``."""
    n, b = 32, 2
    t = torch.linspace(-1, 1, n)
    coords = torch.stack(torch.meshgrid(t, t, indexing="ij"), dim=-1)
    return coords.unsqueeze(0).expand(b, -1, -1, -1).contiguous(), 2.0 / (n - 1)


@pytest.fixture
def device():
    return torch.device(os.environ.get("DIF_FNO_TEST_DEVICE", "cpu"))
