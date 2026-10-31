"""Utilities: seeding, provenance, dtype helpers."""

from .provenance import (
    file_sha256,
    find_repo_root,
    git_commit,
    git_is_dirty,
    provenance_block,
    tensor_fingerprint,
)
from .seed import describe_environment, set_seed, torch_dtype

__all__ = [
    "set_seed", "describe_environment", "torch_dtype",
    "provenance_block", "git_commit", "git_is_dirty", "find_repo_root",
    "file_sha256", "tensor_fingerprint",
]
