import random

import torch


def pairwise_sqr_dists(points, centers):
    """Square L2 distance between each point and each center (dists[i_point, i_center])"""
    size = (
        len(points),
        len(centers),
        points.size(1),
    )  # n_points, n_centers, n_features
    diffs = points.unsqueeze(1).expand(size) - centers.unsqueeze(0).expand(size)
    square_distances = diffs.pow(2).sum(-1)  # n_points, n_centers
    return square_distances


def kmeans_farthest(train_points, n_centers):
    """
    train_points: Training points (size = n_points, n_features)
    n_centers: Number of centers

    Returns: Cluster centers (size = n_centers, train_points.size(1))
    """

    assert len(train_points) >= n_centers, (train_points, n_centers)

    dists = None  # Distance of each point from its closest center
    centers = torch.empty(n_centers, train_points.size(1), device=train_points.device)
    centers[0] = train_points[random.randint(0, len(train_points) - 1)]
    for i_center in range(1, n_centers):
        dists = pairwise_sqr_dists(train_points, centers[:i_center]).min(1)[0]

        # Point that is farthest from all previous centers
        centers[i_center] = train_points[dists.argmax()[0]]

    avg_dist = torch.inf
    while True:
        dists, closest_centers = pairwise_sqr_dists(train_points, centers).min(1)
        new_avg_dist = dists.mean()
        if new_avg_dist >= avg_dist * (1 - 1e-4):
            # new_avg_dist isn't much better than avg_dist
            break

        avg_dist = new_avg_dist
        for i_center in range(n_centers):
            # OPTIM: train_points[nn.functional.one_hot(closest_centers)[:, i_center]]?
            centers[i_center] = train_points[closest_centers == i_center].mean(0)

    return centers
