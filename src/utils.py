import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity


def cosine_similarity(v1, v2):
    """Computes the cosine similarity between two 1D vectors using scikit-learn."""
    v1_2d = np.array(v1).reshape(1, -1)
    v2_2d = np.array(v2).reshape(1, -1)
    return float(sklearn_cosine_similarity(v1_2d, v2_2d)[0][0])
