import matplotlib.pyplot as plt
import torch
from pycave.bayes import GaussianMixture

import data2d
import eval
import train


def mixture_densities(mixture, points):
    """Density of learned distribution at each point (ignoring which component it's from)"""
    return torch.exp(-mixture.score_samples(points).flatten())


if __name__ == "__main__":
    device = "cuda"
    train_points, val_points, val_targets = data2d.line(5000, 5000, device)
    whiten = train.Normalize(train_points[0]).fit(train_points)
    train_points, val_points = whiten(train_points), whiten(val_points)

    n_runs = 1
    balanced_accs = torch.zeros(n_runs)
    for i_run in range(n_runs):
        print(f"-------------------------------- Run {i_run} --------------------------------")
        mixture = GaussianMixture(
            num_components=min(100, 2 + len(train_points) // 25),
            covariance_type="spherical",
            init_strategy="kmeans",
            batch_size=10000,
        )
        mixture.fit(train_points)
        threshold = mixture_densities(mixture, train_points).min().item()

        val_densities = mixture_densities(mixture, val_points)
        val_outputs = val_densities - threshold
        val_targets = val_targets.detach().cpu()
        balanced_accs[i_run] = eval.bin_acc(val_outputs, val_targets)[1]
    
    print(f"Balanced validation accuracy: {100*balanced_accs.mean()} +- {300*balanced_accs.std()}")

    # More detailed results from the last run
    print("Density threshold:", threshold)
    print(f"Number of positive training points: {len(train_points)}")
    data2d.scatter_outputs_targets(
        train_points.detach().cpu(),
        torch.ones(len(train_points), dtype=torch.bool),
        torch.ones(len(train_points), dtype=torch.bool),
        "Training data",
    )
    data2d.scatter_outputs_targets(
        val_points,
        val_outputs,
        val_targets,
        "pycave GMM",
        # There doesn't seem to be a nice way to get the means
        centers=mixture.model_.means,
    )

    eval.plot_distr_overlap(
        val_densities[val_targets],
        val_densities[~val_targets],
        "Validation positive",
        "negative point densities"
    )
    # d = val_densities[val_targets]
    # print("Densities in validation distribution:", d.mean() - torch.arange(3) * d.std())
    # d = val_densities[~val_targets]
    # print(
    #     "Densities outside validation distribution:",
    #     d.mean() + torch.arange(3) * d.std(),
    # )
    plt.show()
