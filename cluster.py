import time

import torch
from pykeops.torch import LazyTensor

import data2d

# With LazyTensor: 50 loops, best of 5: 5.53 msec per loop
# Without LazyTensor: 10 loops, best of 5: 20.6 msec per loop


def pairwise_sqr_dists(points, centers):
    """Square L2 distance between each point and each center (dists[i_point, i_center])"""
    return ((points[:, None, :] - centers[None, :, :])**2).sum(-1)


def kmeans_farthest(train_points, n_centers):
    """Initialize kmeans centers by choosing the farthest point from existing centers"""

    centers = torch.empty(n_centers, train_points.size(1), device=train_points.device)
    centers[0] = train_points[0]

    # Square distance of each point from its closest center
    dists = ((train_points - centers[0])**2).sum(-1)
    for i_center in range(1, n_centers):
        # Point that is farthest from previous centers
        centers[i_center] = train_points[dists.argmax().item()]
        if i_center == n_centers - 1:
            break  # Don't need distances

        new_dists = ((train_points - centers[i_center])**2).sum(-1)
        torch.minimum(dists, new_dists, out=dists)

    return centers


def kmeans_plusplus(train_points, n_centers):
    """Initialize kmeans centers by choosing points with probability proportional to their distance
    from existing centers"""

    centers = torch.empty(n_centers, train_points.size(1), device=train_points.device)
    centers[0] = train_points[0]

    # Square distance of each point from its closest center
    dists = ((train_points - centers[0])**2).sum(-1)
    for i_center in range(1, n_centers):
        # Sample with probability proportional to square distance from previous centers.
        # Previous centers themselves have distance 0 and thus probability 0.
        # Sampling with replacement is probably faster than without replacement, but doesn't affect
        # the result, since we only take one sample.
        idx = torch.multinomial(dists, 1, replacement=True).item()
        centers[i_center] = train_points[idx]
        if i_center == n_centers - 1:
            break  # Don't need distances

        new_dists = ((train_points - centers[i_center])**2).sum(-1)
        torch.minimum(dists, new_dists, out=dists)

    return centers


def kmeans_lloyd(train_points, centers, accuracy):
    """Implements Lloyd's algorithm for the Euclidean metric."""

    assert 0 <= accuracy <= 1, accuracy

    point_i = LazyTensor(train_points[:, None, :])
    center_j = LazyTensor(centers[None, :, :])
    
    avg_dist = torch.inf
    for iter in range(1000):
        # E step: assign points to the closest cluster
        dist_ij = point_i.sqdist(center_j)  # symbolic squared distances
        if iter % 2 == 1:
            dist_i, which_j = dist_ij.min_argmin(dim=1)
        else:
            which_j = dist_ij.argmin(dim=1)
        
        # M step: update the center to the cluster average
        centers.scatter_reduce_(
            0,
            which_j.view(-1, 1).expand_as(train_points),
            train_points,
            reduce="mean",
            include_self=False
        )

        if iter % 2 == 1:
            new_avg_dist = dist_i.mean()
            # Average square distance should decrease by at least a factor of `accuracy`
            if new_avg_dist >= accuracy * avg_dist:
                # new_avg_dist isn't much better than avg_dist
                break

            avg_dist = new_avg_dist

    return centers


def kmeans(train_points, n_centers, accuracy=0.999):
    """
    train_points: Training points (size = n_points, n_features)
    n_centers: Number of centers

    Returns: Cluster centers (size = n_centers, n_features)
    """

    centers = train_points[torch.randperm(len(train_points))[:n_centers]]
    return kmeans_lloyd(train_points, centers, accuracy)


def bench():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_points = data2d.hollow(5000, 1, device, 100)[0]
    print("train_points", len(train_points))

    def b():
        # t = time.time()
        centers = kmeans(train_points, 1+len(train_points)//100)
        torch.cuda.synchronize()
        # print(time.time() - t)
        return centers

    return b
