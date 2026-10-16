"""Accounting helpers for serialized Nox checkpoint artifacts."""

import hashlib
import json
from pathlib import Path


def checkpoint_file_sizes(checkpoint_dir, layer_indices=None):
    """Return serialized byte sizes for the expected checkpoint files."""
    checkpoint_dir = Path(checkpoint_dir)
    if layer_indices is None:
        paths = sorted(checkpoint_dir.glob("compressed_layer_*.pt"))
    else:
        paths = [checkpoint_dir / f"compressed_layer_{i}.pt" for i in layer_indices]

    if not paths:
        raise FileNotFoundError(f"No compressed layer checkpoints in {checkpoint_dir}")

    sizes = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing checkpoint file: {path}")
        size = path.stat().st_size
        if size <= 0:
            raise ValueError(f"Checkpoint file is empty: {path}")
        sizes[path.name] = size
    return sizes


def checkpoint_storage_bytes(checkpoint_dir, layer_indices=None):
    return sum(checkpoint_file_sizes(checkpoint_dir, layer_indices).values())


def fingerprint(payload):
    """Hash a JSON-compatible run identity with stable key ordering."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def source_files_fingerprint(paths):
    """Hash source paths and contents so uncommitted pipeline edits invalidate reuse."""
    digest = hashlib.sha256()
    for path in sorted((Path(p) for p in paths), key=lambda p: str(p)):
        digest.update(str(path.name).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()
