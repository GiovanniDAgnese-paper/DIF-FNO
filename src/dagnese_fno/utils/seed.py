"""Reproducibility helpers."""

from __future__ import annotations

import os
import random

import numpy as np

__all__ = ["set_seed", "describe_environment", "torch_dtype"]


def set_seed(seed: int, deterministic: bool = True, warn_only: bool = True) -> None:
    """Seed Python, NumPy and PyTorch (and cuDNN when relevant).

    ``deterministic=True`` additionally requests deterministic algorithms; with
    ``warn_only`` an unsupported operator only warns instead of raising, which
    keeps CPU training reproducible without breaking on exotic kernels.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dependency anyway
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=warn_only)
        except TypeError:  # torch < 1.11
            torch.use_deterministic_algorithms(True)


def describe_environment() -> dict:
    """Versions and hardware, recorded with every run for provenance."""
    import platform
    import sys

    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["torch_threads"] = torch.get_num_threads()
        info["cuda_available"] = bool(torch.cuda.is_available())
    except ImportError:  # pragma: no cover
        pass
    try:
        import scipy
        info["scipy"] = scipy.__version__
    except ImportError:  # pragma: no cover
        pass
    return info


def torch_dtype(name: str):
    """Resolve a dtype name (``float32``/``float64``) to a ``torch.dtype``."""
    import torch
    return getattr(torch, str(name), torch.float32)
