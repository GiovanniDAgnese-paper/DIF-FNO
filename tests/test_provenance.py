"""Provenance: every published number must be traceable to code + data + env."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch

from dagnese_fno.utils import (
    describe_environment,
    file_sha256,
    find_repo_root,
    git_commit,
    git_is_dirty,
    provenance_block,
    set_seed,
    tensor_fingerprint,
)


def test_repo_root_is_the_checkout():
    root = find_repo_root()
    assert root is not None
    assert (root / "pyproject.toml").exists()
    assert (root / "src" / "dagnese_fno").is_dir()


def test_git_state_is_readable():
    commit = git_commit()
    assert commit is None or (len(commit) == 40 and all(c in "0123456789abcdef" for c in commit))
    assert git_is_dirty() in (True, False, None)


def test_file_sha256_matches_hashlib(tmp_path):
    payload = b"dagnese-fno provenance" * 1000
    p = tmp_path / "blob.bin"
    p.write_bytes(payload)
    assert file_sha256(p) == hashlib.sha256(payload).hexdigest()
    assert file_sha256(p, chunk_size=7) == file_sha256(p), "chunking must not matter"


def test_tensor_fingerprint_is_deterministic_and_discriminating():
    a = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    b = a.clone()
    c = a + 1e-3
    d = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    assert tensor_fingerprint(a) == tensor_fingerprint(b)
    assert tensor_fingerprint(a) != tensor_fingerprint(c)
    assert tensor_fingerprint(a) != tensor_fingerprint(d), "shape enters the digest"
    assert tensor_fingerprint(a, b) != tensor_fingerprint(a)


def test_provenance_block_records_the_dataset(tmp_path):
    p = tmp_path / "dataset.pt"
    p.write_bytes(b"x" * 128)
    block = provenance_block(p)
    assert block["dataset"].endswith("dataset.pt")
    assert block["dataset_sha256"] == file_sha256(p)
    assert "git_commit" in block and "git_dirty" in block


def test_provenance_block_without_a_dataset():
    block = provenance_block(None)
    assert "dataset" not in block and "dataset_sha256" not in block
    assert block["git_commit"] == git_commit()


def test_provenance_block_survives_a_missing_file(tmp_path):
    block = provenance_block(tmp_path / "nope.pt")
    assert "dataset_sha256" not in block


# --------------------------------------------------------------------------- #
# Reproducibility of the training stack
# --------------------------------------------------------------------------- #
def test_set_seed_makes_the_split_reproducible():
    n = 64
    set_seed(1234)
    first = torch.randperm(n, generator=torch.Generator().manual_seed(1234))
    set_seed(1234)
    second = torch.randperm(n, generator=torch.Generator().manual_seed(1234))
    assert torch.equal(first, second)
    set_seed(1234)
    a = torch.randn(8, 8)
    set_seed(1234)
    b = torch.randn(8, 8)
    assert torch.equal(a, b), "set_seed must also fix the global torch RNG"


def test_describe_environment_is_serialisable():
    import json

    env = describe_environment()
    assert json.dumps(env)                      # no numpy / torch scalars
    assert "python" in env and "torch" in env
    assert isinstance(env["torch"], str)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dataset_regeneration_is_bitwise_identical(tmp_path, dtype):
    """The strongest reproducibility statement we can make about the data."""
    from dagnese_fno.data import generate_dataset, save_dataset

    kwargs = {"domain": "star", "n": 24, "n_samples": 3, "seed": 5, "dtype": dtype}
    d1 = generate_dataset(**kwargs)
    d2 = generate_dataset(**kwargs)
    assert torch.equal(d1["coeffs"], d2["coeffs"])
    assert torch.equal(d1["targets"], d2["targets"])
    # torch's archive stores the file name, so compare equal names in
    # different folders: the bytes of the *content* must be reproducible.
    p1 = save_dataset(d1, tmp_path / "run1" / "dataset.pt")
    p2 = save_dataset(d2, tmp_path / "run2" / "dataset.pt")
    assert file_sha256(p1) == file_sha256(p2), "saving must be byte-reproducible"

    d3 = generate_dataset(**{**kwargs, "seed": 6})
    assert not torch.equal(d1["coeffs"], d3["coeffs"]), "the seed must matter"
    assert (tmp_path / "run1" / "dataset.json").exists()
    meta = np.asarray(torch.load(p1, map_location="cpu", weights_only=True)["mask"])
    assert meta.dtype == np.bool_
