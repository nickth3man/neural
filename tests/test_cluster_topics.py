"""Tests for k-means topic clustering helper."""

from __future__ import annotations

import numpy as np

from neural.cluster_topics import kmeans_cosine


def test_kmeans_cosine_groups_opposite_poles() -> None:
    x = np.array(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.0, 1.0],
            [0.01, 0.99],
        ],
        dtype=np.float32,
    )
    labels, _ = kmeans_cosine(x, k=2, seed=0)
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]
