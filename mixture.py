import matplotlib.pyplot as plt
import numpy as np
import torch
from pycave.bayes import GaussianMixture

import data


def mixture_densities(mixture, points):
    return torch.exp(-mixture.score_samples(points).squeeze())


if __name__ == '__main__':
    device = 'cpu'
    (points, labels), n_train = data.circles(2000, device), 1000
    train_points = points[:n_train][labels[:n_train]]
    val_points, val_labels = points[n_train:], labels[n_train:]

    mixture = GaussianMixture(
        num_components=4,
        covariance_type='full',
        init_strategy='kmeans',
        trainer_params={}
    )
    mixture.fit(train_points)
    threshold = mixture_densities(mixture, train_points).min().item()
    print("Density threshold:", threshold)

    val_densities = mixture_densities(mixture, val_points)
    d = val_densities[val_labels]
    print('Densities in validation distribution', d.mean() - torch.arange(3)*d.std())
    d = val_densities[~val_labels]
    print('Densities outside validation distribution', d.mean() + torch.arange(3)*d.std())

    val_outputs = val_densities - threshold
    data.binary_accuracy(val_labels, val_outputs, 'pycave GMM validation')
    data.scatter_labels_outputs(
        val_points.cpu().numpy(),
        val_labels.cpu().numpy(),
        val_outputs.cpu().numpy(),
        'pycave GMM',
        # There doesn't seem to be a nice way to get the centers
        centers=mixture.model_.means
    )
    plt.show()
