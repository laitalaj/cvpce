from math import sqrt

import torch

from thesis.models import classification

def test_nearest_neigbors():
    anchors = torch.tensor([
        [1, 0, 0],
        [1 / sqrt(3), 1 / sqrt(3), 1 / sqrt(3)],
        [-1 / sqrt(3), -1 / sqrt(3), -1 / sqrt(3)],
        [-1, 0, 0],
        [1 / sqrt(2), 0, 1 / sqrt(2)],
        [-1 / sqrt(2), 0, -1 / sqrt(2)],
    ], dtype=torch.float)
    queries = torch.tensor([
        [1 / sqrt(1.01), 0.1 / sqrt(1.01), 0], # should be 0
        [0.9 / sqrt(2.02), 0, 1.1 / sqrt(2.02)], # should be 4
        [-1, 0, 0], # should be 3
        [1, 0, 0], # should be 0
        [1 / sqrt(3), 1 / sqrt(3), 1 / sqrt(3)], # should be 1
        [-1.1 / sqrt(2.02), 0, -0.9 / sqrt(2.02)], # should be 5
        [-1, 0, 0], # should be 3
    ])
    expected = torch.tensor([0, 4, 3, 0, 1, 5, 3])

    actual = classification.nearest_neighbors(anchors, queries)
    assert expected.equal(actual)
