import matplotlib.pyplot as plt
import numpy as np
import torch
from pycave.bayes import GaussianMixture

import data2d


def mixture_densities(mixture, points):
    """Density of learned distribution at each point (ignoring which component it's from)"""
    return torch.exp(-mixture.score_samples(points).flatten())


if __name__ == '__main__':
    device = 'cuda'
    train_points, val_points, val_labels = data2d.line(5000, 5000, device)

    mixture = GaussianMixture(
        num_components=min(100, 2 + len(train_points) // 25),
        covariance_type='spherical',
        init_strategy='kmeans',
        batch_size=10000,
    )
    mixture.fit(train_points)
    threshold = mixture_densities(mixture, train_points).min().item()
    print("Density threshold:", threshold)

    print(f"Number of positive training points: {len(train_points)}")
    data2d.scatter_labels_outputs(
        train_points.detach().cpu(),
        torch.ones(len(train_points), dtype=torch.bool),
        torch.ones(len(train_points), dtype=torch.bool),
        'Training data',
    )

    val_densities = mixture_densities(mixture, val_points)
    val_labels = val_labels.cpu()
    d = val_densities[val_labels]
    print('Densities in validation distribution:', d.mean() - torch.arange(3)*d.std())
    d = val_densities[~val_labels]
    print('Densities outside validation distribution:', d.mean() + torch.arange(3)*d.std())

    val_outputs = val_densities - threshold
    data2d.print_accuracy(val_labels, val_outputs, 'pycave GMM validation')
    data2d.scatter_labels_outputs(
        val_points.detach().cpu(),
        val_labels,
        val_outputs.detach().cpu(),
        'pycave GMM',
        # There doesn't seem to be a nice way to get the means
        centers=mixture.model_.means
    )
    plt.show()
