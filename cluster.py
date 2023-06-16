import time

import torch
from pykeops.torch import LazyTensor

import data2d

# With LazyTensor: 50 loops, best of 5: 5.53 msec per loop
# Without LazyTensor: 10 loops, best of 5: 20.6 msec per loop


def pairwise_sqr_dists(X, centers):
    """Square L2 distance between each point and each center (dists[i_x, i_center])"""
    return ((X[:, None, :] - centers[None, :, :]) ** 2).sum(-1)


def kmeans_farthest(X_train, n_centers):
    """Initialize kmeans centers by choosing the farthest point from existing centers"""

    centers = torch.empty(n_centers, X_train.size(1), device=X_train.device)
    centers[0] = X_train[0]

    # Square distance of each point from its closest center
    dists = ((X_train - centers[0]) ** 2).sum(-1)
    for i_center in range(1, n_centers):
        # Point that is farthest from previous centers
        centers[i_center] = X_train[dists.argmax().item()]
        if i_center == n_centers - 1:
            break  # Don't need distances

        new_dists = ((X_train - centers[i_center]) ** 2).sum(-1)
        torch.minimum(dists, new_dists, out=dists)

    return centers


def kmeans_plusplus(X_train, n_centers):
    """Initialize kmeans centers by choosing points with probability proportional to their distance
    from existing centers"""

    centers = torch.empty(n_centers, X_train.size(1), device=X_train.device)
    centers[0] = X_train[0]

    # Square distance of each point from its closest center
    dists = ((X_train - centers[0]) ** 2).sum(-1)
    for i_center in range(1, n_centers):
        # Sample with probability proportional to square distance from previous centers.
        # Previous centers themselves have distance 0 and thus probability 0.
        # Sampling with replacement is probably faster than without replacement, but doesn't affect
        # the result, since we only take one sample.
        idx = torch.multinomial(dists, 1, replacement=True).item()
        centers[i_center] = X_train[idx]
        if i_center == n_centers - 1:
            break  # Don't need distances

        new_dists = ((X_train - centers[i_center]) ** 2).sum(-1)
        torch.minimum(dists, new_dists, out=dists)

    return centers


def kmeans_lloyd(X_train, centers, accuracy):
    """Implements Lloyd's algorithm for the Euclidean metric."""

    assert 0 <= accuracy <= 1, accuracy

    x_i = LazyTensor(X_train[:, None, :])
    center_j = LazyTensor(centers[None, :, :])

    avg_dist = torch.inf
    for iter in range(1000):
        # E step: assign points to the closest cluster
        dist_ij = x_i.sqdist(center_j)  # symbolic squared distances
        if iter % 2 == 1:
            dist_i, j_center_from_i = dist_ij.min_argmin(dim=1)
        else:
            j_center_from_i = dist_ij.argmin(dim=1)

        # M step: update the center to the cluster average
        centers.scatter_reduce_(
            0,
            j_center_from_i.view(-1, 1).expand_as(X_train),
            X_train,
            reduce="mean",
            include_self=False,
        )

        if iter % 2 == 1:
            new_avg_dist = dist_i.mean()
            # Average square distance should decrease by at least a factor of `accuracy`
            if new_avg_dist >= accuracy * avg_dist:
                # new_avg_dist isn't much better than avg_dist
                break

            avg_dist = new_avg_dist

    return centers


def kmeans(X_train, n_centers, accuracy=0.9999):
    """
    X_train: Training points (size = n_points, n_features)
    n_centers: Number of centers

    Returns: Cluster centers (size = n_centers, n_features)
    """

    centers = X_train[torch.randperm(len(X_train))[:n_centers]]
    return kmeans_lloyd(X_train, centers, accuracy)


def cluster_var_pr_(var, pr, X_train, centers):
    """
    Overwrites `var` and `pr`

    var: Variance for each cluster (n_centers)
    pr: Weight proportional to probability of each cluster (n_centers)
    X_train: Training points (n_points, n_features)
    centers: Center of each cluster (n_centers, n_features)
    """

    # OPTIM: LazyTensor with gather?
    dist_ij = (X_train[:, None, :] - centers[None, :, :]).pow(2).sum(2)
    j_center_from_i = dist_ij.argmin(dim=1)
    for j in range(len(centers)):
        var[j] = dist_ij[j_center_from_i == j].mean()
    pr.copy_(j_center_from_i.bincount(minlength=len(centers)) / float(len(X_train)))


def bench():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X_train = data2d.hollow(5000, 1, device, 100)[0]
    print("X_train", len(X_train))

    def b():
        # t = time.time()
        centers = kmeans(X_train, 1 + len(X_train) // 100)
        torch.cuda.synchronize()
        # print(time.time() - t)
        return centers

    return b
