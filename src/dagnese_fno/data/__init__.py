"""Data: computational domains, the Darcy reference solver and the datasets.

This package was missing from the 1.1.0 release (``.gitignore`` contained the
pattern ``data/``, which silently excluded ``src/dagnese_fno/data/`` from the
repository, so ``import dagnese_fno`` failed for every user).  It is restored
here, together with the tests that cover it.
"""

from .darcy import assemble_darcy, interior_mask, sample_coeff_field, solve_darcy
from .dataset import dataset_path, generate_dataset, load_dataset, reference_coords, save_dataset
from .masks import (
    DOMAINS,
    DomainSpec,
    annulus_mask,
    clear_frame,
    get_mask,
    double_hole_mask,
    lshape_mask,
    mask_summary,
    naca0012_mask,
    naca0012_thickness,
    slit_mask,
    star_mask,
    unit_coordinates,
)

__all__ = [
    # masks
    "DOMAINS", "DomainSpec", "get_mask", "clear_frame", "mask_summary",
    "star_mask", "lshape_mask", "annulus_mask", "naca0012_mask",
    "naca0012_thickness", "slit_mask", "double_hole_mask", "unit_coordinates",
    # solver
    "assemble_darcy", "solve_darcy", "sample_coeff_field", "interior_mask",
    # datasets
    "generate_dataset", "save_dataset", "load_dataset", "reference_coords",
    "dataset_path",
]
