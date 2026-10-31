"""Data generation for the benchmark: masks, Darcy solver, body-fitted meshes.

The three sub-modules are independent:

* :mod:`dagnese_fno.data.masks` -- the four non-convex domains and the registry
  :data:`~dagnese_fno.data.masks.DOMAINS` that drives everything else;
* :mod:`dagnese_fno.data.darcy` -- the log-normal permeability sampler, the
  5-point finite-volume solver and the ``(a, u)`` dataset generator;
* :mod:`dagnese_fno.data.meshes` -- the two-block body-fitted H-grid around a
  NACA 0012 (the geometry the paper is motivated by).

Typical use::

    from dagnese_fno.data import generate_dataset, save_dataset, load_dataset

    data = generate_dataset("annulus", n=64, n_samples=100, seed=2)
    save_dataset(data, "data/darcy_annulus.pt")
    coeffs, targets, mask = load_dataset("data/darcy_annulus.pt").values()
"""

from __future__ import annotations

from .darcy import (
    assemble_darcy,
    generate_dataset,
    load_dataset,
    sample_coeff_field,
    save_dataset,
    solve_darcy,
)
from .masks import (
    DOMAINS,
    DomainSpec,
    annulus_mask,
    clear_frame,
    get_mask,
    grid_2d,
    lshape_mask,
    mask_summary,
    double_hole_mask,
    naca0012_mask,
    slit_mask,
    star_mask,
)
from .meshes import (
    NACA_THICKNESS_RATIO,
    generate_dagnese_naca0012_mesh,
    naca0012_grid_spacing,
    naca0012_surface,
    naca0012_thickness,
)

__all__ = [
    # masks
    "DOMAINS",
    "DomainSpec",
    "annulus_mask",
    "clear_frame",
    "get_mask",
    "grid_2d",
    "lshape_mask",
    "mask_summary",
    "naca0012_mask",
    "slit_mask",
    "double_hole_mask",
    "star_mask",
    # darcy
    "assemble_darcy",
    "generate_dataset",
    "load_dataset",
    "sample_coeff_field",
    "save_dataset",
    "solve_darcy",
    # meshes
    "NACA_THICKNESS_RATIO",
    "generate_dagnese_naca0012_mesh",
    "naca0012_grid_spacing",
    "naca0012_surface",
    "naca0012_thickness",
]
