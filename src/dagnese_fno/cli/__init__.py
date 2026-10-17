"""Command-line entry points (``dif-fno-train``, ``dif-fno-generate-data``)."""

from .generate_data import main as generate_data_main
from .train import main as train_main

__all__ = ["train_main", "generate_data_main"]
