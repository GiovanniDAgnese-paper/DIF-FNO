"""Activations together with their Lipschitz constants (needed for certificates)."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn

__all__ = ["get_activation", "activation_lipschitz", "ACTIVATION_LIPSCHITZ"]

# Best-known (upper) Lipschitz constants.
ACTIVATION_LIPSCHITZ: dict[str, float] = {
    "tanh": 1.0,          # |tanh'| <= 1
    "relu": 1.0,          # |ReLU'| <= 1
    "leaky_relu": 1.0,
    "sigmoid": 0.25,      # max sigma' = 1/4
    "gelu": 1.12,         # max GELU' ~ 1.0845, 1.12 is a safe upper bound
    "silu": 1.10,         # max SiLU' ~ 1.0992
    "identity": 1.0,
    "softplus": 1.0,
}


def get_activation(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return the functional form of a supported activation."""
    key = str(name).lower()
    table = {
        "tanh": torch.tanh,
        "relu": torch.relu,
        "leaky_relu": torch.nn.functional.leaky_relu,
        "sigmoid": torch.sigmoid,
        "gelu": torch.nn.functional.gelu,
        "silu": torch.nn.functional.silu,
        "identity": lambda x: x,
        "softplus": torch.nn.functional.softplus,
    }
    if key not in table:
        raise ValueError(f"unsupported activation '{name}'; available: {sorted(table)}")
    return table[key]


def activation_lipschitz(module: nn.Module) -> float:
    """Lipschitz constant contributed by an activation module (1 for anything else)."""
    if isinstance(module, nn.Tanh):
        return ACTIVATION_LIPSCHITZ["tanh"]
    if isinstance(module, nn.ReLU):
        return ACTIVATION_LIPSCHITZ["relu"]
    if isinstance(module, nn.LeakyReLU):
        return ACTIVATION_LIPSCHITZ["leaky_relu"]
    if isinstance(module, nn.Sigmoid):
        return ACTIVATION_LIPSCHITZ["sigmoid"]
    if isinstance(module, nn.GELU):
        return ACTIVATION_LIPSCHITZ["gelu"]
    if isinstance(module, nn.SiLU):
        return ACTIVATION_LIPSCHITZ["silu"]
    if isinstance(module, nn.Softplus):
        return ACTIVATION_LIPSCHITZ["softplus"]
    return 1.0
