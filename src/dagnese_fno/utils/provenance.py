"""Provenance helpers: what code, what data, what environment produced a run."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

__all__ = [
    "find_repo_root",
    "git_commit",
    "git_is_dirty",
    "file_sha256",
    "tensor_fingerprint",
    "provenance_block",
]


def find_repo_root(start: Path | None = None) -> Path | None:
    """Walk up from ``start`` (default: this file) until a ``.git`` directory is found."""
    p = (start or Path(__file__).resolve()).parent
    for candidate in [p, *p.parents]:
        if (candidate / ".git").exists():
            return candidate
    return None


def git_commit(root: Path | None = None) -> str | None:
    """Full hash of ``HEAD``, or ``None`` when git is unavailable."""
    root = root or find_repo_root()
    if root is None:
        return None
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(root),
                             capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            return out.stdout.strip() or None
    except Exception:  # pragma: no cover - git missing / timeout
        return None
    return None


def git_is_dirty(root: Path | None = None) -> bool | None:
    """``True`` when the working tree has uncommitted changes."""
    root = root or find_repo_root()
    if root is None:
        return None
    try:
        out = subprocess.run(["git", "status", "--porcelain"], cwd=str(root),
                             capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            return bool(out.stdout.strip())
    except Exception:  # pragma: no cover
        return None
    return None


def file_sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of a file, streamed (datasets are tens of MB)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_fingerprint(*tensors, decimals: int = 6) -> str:
    """Short, deterministic digest of tensor contents (dataset-level provenance)."""
    h = hashlib.sha256()
    for t in tensors:

        arr = t.detach().cpu().numpy() if hasattr(t, "detach") else t
        h.update(arr.shape.__str__().encode())
        h.update(arr.astype("float64").round(decimals).tobytes())
    return h.hexdigest()[:16]


def provenance_block(dataset_path: str | Path | None = None) -> dict:
    """Everything needed to answer 'what exactly produced this number?'."""
    block: dict = {
        "git_commit": git_commit(),
        "git_dirty": git_is_dirty(),
    }
    if dataset_path is not None and Path(dataset_path).exists():
        block["dataset"] = str(dataset_path)
        block["dataset_sha256"] = file_sha256(dataset_path)
    return block
