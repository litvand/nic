import matplotlib.pyplot as plt
import torch
from pycave.bayes import GaussianMixture
from torch import nn

import data2d
import eval
import train


class DetectorMixture(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.mixture = GaussianMixture(*args, **kwargs)
        self.threshold = nn.Parameter(torch.nan, requires_grad=False)

    def forward(self, points):
        return self.densities(points) - self.threshold

    def densities(self, points):
        """Density of learned distribution at each point (ignoring which component it's from)"""
        return torch.exp(-self.mixture.score_samples(points).flatten())

    def fit(self, train_points):
        self.fit_predict(train_points)
        return self

    def fit_predict(self, train_points):
        self.mixture.fit(train_points)
        densities = self.densities(train_points)
        self.threshold = nn.Parameter(densities.min(), requires_grad=False)
        return densities - self.threshold


if __name__ == "__main__":
    device = "cuda"
    train_points, val_points, val_targets = data2d.hollow(5000, 5000, device, 4)
    whiten = train.Whiten(train_points[0]).fit(train_points)
    train_points, val_points = whiten(train_points), whiten(val_points)
    val_targets = val_targets.detach().cpu()

    n_runs = 5
    balanced_accs = torch.zeros(n_runs)
    for i_run in range(n_runs):
        print(f"-------------------------------- Run {i_run} --------------------------------")
        detector = DetectorMixture(
            num_components=min(300, 2 + len(train_points) // 25),
            covariance_type="spherical",
            init_strategy="kmeans",
            batch_size=10000,
        ).fit(train_points)
        val_outputs = detector(val_points)
        balanced_accs[i_run] = eval.bin_acc(val_outputs, val_targets)[1]

    print(f"Balanced validation accuracy: {100*balanced_accs.mean()} +- {300*balanced_accs.std()}")

    # More detailed results from the last run
    print(f"Number of positive training points: {len(train_points)}")
    print("Density threshold:", detector.threshold)
    # exit(0)
    data2d.scatter_outputs_targets(
        train_points,
        detector(train_points),
        torch.ones(len(train_points), dtype=torch.bool),
        "Training data",
        # There doesn't seem to be a nice way to get the means
        centers=detector.mixture.model_.means,
    )
    data2d.scatter_outputs_targets(
        val_points,
        val_outputs,
        val_targets,
        "pycave GMM validation",
        centers=detector.mixture.model_.means,
    )
    eval.plot_distr_overlap(
        val_outputs[val_targets],
        val_outputs[~val_targets],
        "Validation positive",
        "negative point thresholded densities",
    )
    plt.show()
