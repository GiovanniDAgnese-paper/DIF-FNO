"""Backwards-compatible shim -- use :mod:`dagnese_fno.data.meshes` instead.

The original module computed the NACA 0012 thickness distribution
:math:`y_t(x)` and then returned a plain tensor grid that ignored it.  The
implementation in :mod:`dagnese_fno.data.meshes` builds a genuine two-block
body-fitted H-grid around the profile and is available here under the same
function name.
"""

from __future__ import annotations

import warnings

from .data.meshes import (
    generate_dagnese_naca0012_mesh,
    naca0012_grid_spacing,
    naca0012_surface,
    naca0012_thickness,
)

warnings.warn(
    "dagnese_fno.dagnese_mesh_generator is deprecated; use "
    "dagnese_fno.data.meshes instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "generate_dagnese_naca0012_mesh",
    "naca0012_thickness",
    "naca0012_surface",
    "naca0012_grid_spacing",
]


if __name__ == "__main__":
    mesh = generate_dagnese_naca0012_mesh()
    print(f"D'Agnese NACA mesh generated successfully: {tuple(mesh.shape)}")
