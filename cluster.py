import time

import torch
from pykeops.torch import LazyTensor

import data2d

# With LazyTensor: 50 loops, best of 5: 5.53 msec per loop
# Without LazyTensor: 10 loops, best of 5: 20.6 msec per loop

def kmeans_farthest_(centers, X_train):
    """
    Initialize kmeans centers by choosing the farthest point from existing centers.
    
    Overwrites centers
    """

    centers[0] = X_train[0]

    # Square distance of each point from its closest center
    dists = ((X_train - centers[0]) ** 2).sum(-1)

    for i_center in range(1, len(centers)):
        # Point that is farthest from previous centers
        centers[i_center] = X_train[dists.argmax().item()]
        if i_center == len(centers) - 1:
            break  # Don't need distances

        new_dists = ((X_train - centers[i_center]) ** 2).sum(-1)
        torch.minimum(dists, new_dists, out=dists)


def kmeans_plusplus_(centers, X_train):
    """
    Initialize kmeans centers by choosing points with probability proportional to their distance
    from existing centers.
    
    Overwrites centers
    """

    centers[0] = X_train[0]

    # Square distance of each point from its closest center
    dists = ((X_train - centers[0]) ** 2).sum(-1)

    for i_center in range(1, len(centers)):
        # Sample with probability proportional to square distance from previous centers.
        # Previous centers themselves have distance 0 and thus probability 0.
        # Sampling with replacement is probably faster than without replacement, but doesn't affect
        # the result, since we only take one sample.
        idx = torch.multinomial(dists, 1, replacement=True).item()
        centers[i_center] = X_train[idx]
        if i_center == len(centers) - 1:
            break  # Don't need distances

        new_dists = ((X_train - centers[i_center]) ** 2).sum(-1)
        torch.minimum(dists, new_dists, out=dists)


def kmeans_lloyd_(centers, X_train, accuracy):
    """
    Implements Lloyd's algorithm for the Euclidean metric.
    
    Assumes centers are already initialized and modifies them.
    """

    assert 0 <= accuracy <= 1, accuracy

    x_i = LazyTensor(X_train[:, None, :])
    center_j = LazyTensor(centers[None, :, :])

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
            j_center_from_i.view(-1, 1).expand_as(X_train),
            X_train,
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


def kmeans_(centers, X_train, accuracy=0.9999):
    """
    centers: Cluster centers (n_centers, n_features). Will be overwritten.
    X_train: Training points (n_points, n_features)
    """

    i_centers = torch.randperm(len(X_train), device=X_train.device)[:len(centers)]
    torch.index_select(X_train, 0, i_centers, out=centers)
    kmeans_lloyd_(centers, X_train, accuracy)


def cluster_var_pr_(var, pr, X_train, centers, min_var=1e-8):
    """
    Overwrites `var` and `pr`

    var: Variance for each cluster (n_centers)
    pr: Probability of each cluster (n_centers)
    X_train: Training points (n_points, n_features)
    centers: Center of each cluster (n_centers, n_features)
    min_var: Variance of a cluster with a single point
    """

    diff_ij = LazyTensor(X_train[:, None, :]) - LazyTensor(centers[None, :, :])
    dist_i, j_center_from_i = (diff_ij**2).sum(2).min_argmin(1)
    dist_i, j_center_from_i = dist_i.view(len(X_train)), j_center_from_i.view(len(X_train))

    # for j in range(len(centers)):
    #     var[j] = dist_i[j_center_from_i == j].mean()
    var.scatter_reduce_(0, j_center_from_i, dist_i, reduce="mean", include_self=False)
    var += min_var
    
    torch.div(j_center_from_i.bincount(minlength=len(centers)), float(len(X_train)), out=pr)


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
