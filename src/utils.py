"""
Utility functions for common computational and helper tasks across the project.
Used to abstract external library dependencies for pure functions like similarity checks,
and to provide standardized short-ID generation for entities.
"""
import numpy as np
import hashlib
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity


def cosine_similarity(v1, v2):
    """Computes the cosine similarity between two 1D vectors using scikit-learn."""
    v1_2d = np.array(v1).reshape(1, -1)
    v2_2d = np.array(v2).reshape(1, -1)
    return float(sklearn_cosine_similarity(v1_2d, v2_2d)[0][0])


def fact_id(text: str) -> str:
    """Generate a deterministic short fact ID: f_ + first 8 hex chars of SHA256."""
    return "f_" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
