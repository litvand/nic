import time

import torch
from pykeops.torch import LazyTensor

import data2d


def kmeans_farthest_(centers, train_X):
    """
    Initialize kmeans centers by choosing the farthest point from existing centers.

    Overwrites centers
    """

    centers[0] = train_X[0]

    # Square distance of each point from its closest center
    dists = ((train_X - centers[0]) ** 2).sum(-1)

    for i_center in range(1, len(centers)):
        # Point that is farthest from previous centers
        centers[i_center] = train_X[dists.argmax().item()]
        if i_center == len(centers) - 1:
            break  # Don't need distances

        new_dists = ((train_X - centers[i_center]) ** 2).sum(-1)
        torch.minimum(dists, new_dists, out=dists)


def kmeans_plusplus_(centers, train_X):
    """
    Initialize kmeans centers by choosing points with probability proportional to their distance
    from existing centers.

    Overwrites centers
    """

    centers[0] = train_X[0]

    # Square distance of each point from its closest center
    dists = ((train_X - centers[0]) ** 2).sum(-1)

    for i_center in range(1, len(centers)):
        # Sample with probability proportional to square distance from previous centers.
        # Previous centers themselves have distance 0 and thus probability 0.
        # Sampling with replacement is probably faster than without replacement, but doesn't affect
        # the result, since we only take one sample.
        idx = torch.multinomial(dists, 1, replacement=True).item()
        centers[i_center] = train_X[idx]
        if i_center == len(centers) - 1:
            break  # Don't need distances

        new_dists = ((train_X - centers[i_center]) ** 2).sum(-1)
        torch.minimum(dists, new_dists, out=dists)


def kmeans_lloyd_(centers, train_X, accuracy):
    """
    Implements Lloyd's algorithm for the Euclidean metric.

    Assumes centers are already initialized and modifies them.
    """

    assert 0 <= accuracy <= 1, accuracy

    x_i, center_j = LazyTensor(train_X[:, None, :]), LazyTensor(centers[None, :, :])
    avg_dist = torch.inf
    for iteration in range(1000):
        # E step: assign points to the closest cluster
        dist_ij = x_i.sqdist(center_j)  # symbolic squared distances
        if iteration % 2 == 1:
            dist_i, j_center_from_i = dist_ij.min_argmin(dim=1)
        else:
            j_center_from_i = dist_ij.argmin(dim=1)

        # M step: update the center to the cluster average
        centers.scatter_reduce_(
            0,
            j_center_from_i.view(-1, 1).expand_as(train_X),
            train_X,
            reduce="mean",
            include_self=False,
        )

        if iteration % 2 == 1:
            new_avg_dist = dist_i.mean()
            # Average square distance should decrease by at least a factor of `accuracy`
            if new_avg_dist >= accuracy * avg_dist:
                # new_avg_dist isn't much better than avg_dist
                break

            avg_dist = new_avg_dist


def kmeans_(centers, train_X, accuracy=0.9999):
    """
    centers: Cluster centers (n_centers, n_features). Will be overwritten.
    train_X: Training points (n_points, n_features)
    """

    if len(centers) >= len(train_X):
        centers[: len(train_X)] = train_X[:]
        centers[len(train_X) :] = train_X[0]
        return

    i_centers = torch.randperm(len(train_X), device=train_X.device)[: len(centers)]
    torch.index_select(train_X, 0, i_centers, out=centers)
    kmeans_lloyd_(centers, train_X, accuracy)


def cluster_var_pr_(vars, prs, train_X, centers, min_var=1e-8):
    """
    Overwrites `vars` and `prs`

    vars: Variance for each cluster (n_centers)
    prs: Probability of each cluster (n_centers)
    train_X: Training points (n_points, n_features)
    centers: Center of each cluster (n_centers, n_features)
    min_var: Variance of a cluster with a single point
    """

    diff_ij = LazyTensor(train_X[:, None, :]) - LazyTensor(centers[None, :, :])
    dist_i, j_center_from_i = (diff_ij**2).sum(2).min_argmin(1)
    dist_i, j_center_from_i = dist_i.view(len(train_X)), j_center_from_i.view(len(train_X))

    # for j in range(len(centers)):
    #     vars[j] = dist_i[j_center_from_i == j].mean()
    vars.scatter_reduce_(0, j_center_from_i, dist_i, reduce="mean", include_self=False)
    vars *= 1.2
    vars += min_var

    torch.div(j_center_from_i.bincount(minlength=len(centers)), float(len(train_X)), out=prs)


# With LazyTensor: 50 loops, best of 5: 5.53 msec per loop
# Without LazyTensor: 10 loops, best of 5: 20.6 msec per loop


def bench():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_X = data2d.hollow(5000, 1, device, 100)[0]
    print("train_X", len(train_X))

    def b():
        t = time.time()
        centers = kmeans_(train_X, 1 + len(train_X) // 100)
        torch.cuda.synchronize()
        print(time.time() - t)
        return centers

    return b
