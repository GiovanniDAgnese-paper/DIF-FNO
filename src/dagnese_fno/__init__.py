"""
D'Agnese DIF-FNO: Diffeomorphic Implicit Fourier Neural Operators
Official Implementation with D'Agnese Topological Barrier Loss.
"""

from .barrier_loss import DAgneseBarrierLoss

__version__ = "1.0.0"
__author__ = "Giovanni D'Agnese"
__email__ = "jovannidagnese2@gmail.com"

__all__ = ["DAgneseBarrierLoss"]
