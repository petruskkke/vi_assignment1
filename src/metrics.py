from __future__ import division


import numpy as np


def euclidean_distance(vec1, vec2):
    return np.sqrt(np.sum((vec1 - vec2) ** 2))


def cosine_similarity(vec1, vec2):
    a = np.sum(vec1 * vec2)
    b = np.sqrt(np.sum(vec1 ** 2)) * np.sqrt(np.sum(vec2 ** 2))
    return a / b
    