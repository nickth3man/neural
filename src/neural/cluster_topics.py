"""Lightweight k-means clustering for topic discovery reports."""

from __future__ import annotations

import numpy as np


def kmeans_cosine(
    points: np.ndarray,
    k: int,
    *,
    max_iter: int = 30,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Run k-means on L2-normalized rows (cosine geometry). Returns ``(labels, centroids)``."""
    if k < 1:
        msg = "k must be at least 1"
        raise ValueError(msg)
    x = np.asarray(points, dtype=np.float32)
    if x.ndim != 2:
        msg = "points must be a 2D array"
        raise ValueError(msg)
    n, d = x.shape
    if n == 0:
        msg = "points cannot be empty"
        raise ValueError(msg)
    k = min(k, n)
    rng = np.random.default_rng(seed)
    centroid_idx = rng.choice(n, size=k, replace=False)
    centroids = x[centroid_idx].copy()

    labels = np.zeros(n, dtype=np.int32)
    for _ in range(max_iter):
        sim = x @ centroids.T
        labels = np.argmax(sim, axis=1)
        new_centroids = np.zeros_like(centroids)
        for j in range(k):
            mask = labels == j
            if not np.any(mask):
                new_centroids[j] = centroids[j]
            else:
                new_centroids[j] = x[mask].mean(axis=0)
        norms = np.linalg.norm(new_centroids, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        new_centroids /= norms
        if np.allclose(new_centroids, centroids, atol=1e-5):
            break
        centroids = new_centroids
    return labels, centroids
