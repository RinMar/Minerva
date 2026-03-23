import hashlib
from pathlib import Path


def hash_string(text: str) -> str:
    """Compute full SHA256 hash of a string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fact_id(text: str) -> str:
    """Generate a deterministic short fact ID: f_ + first 8 hex chars of SHA256."""
    return "f_" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


def hash_file(filepath: Path) -> str:
    """Compute SHA256 hash of a file's contents."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
