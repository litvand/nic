from typing import Any, Dict

import matplotlib.pyplot as plt
import pykeops.torch as ke
import torch
import torch.nn.functional as F
from pycave.bayes import GaussianMixture
from torch import nn

import data2d
import eval
import train


class DetectorKe(nn.Module):
    def __init__(self, example_input, n_centers):
        super().__init__()

        n_features, dtype = len(example_input), example_input.dtype
        self.centers = nn.Parameter(torch.empty(n_centers, n_features, dtype=dtype))
        self.covs_inv_sqrt = nn.Parameter(
            torch.empty(n_centers, n_features, n_features, dtype=dtype)
        )
        self.covs_inv = None  # Calculated based on covs_inv_sqrt
        self.weights = nn.Parameter(torch.empty(n_centers, dtype=dtype))
        self.threshold = nn.Parameter(torch.tensor(torch.nan, dtype=dtype), requires_grad=False)

    def refresh(self):
        """Updates intermediate variables for calculating likelihoods based on parameters."""
        self.covs_inv = torch.matmul(self.covs_inv_sqrt, self.covs_inv_sqrt.transpose(1, 2))
        self.coefs = F.softmax(self.weights) * self.covs_inv.det().sqrt()
    
    def get_extra_state(self):
        return 1  # Make sure `set_extra_state` is called
    
    def set_extra_state(self, _):
        assert not self.threshold.isnan().item(), self.threshold  # Other state was already loaded
        self.refresh()

    def likelihoods(self, points):
        dists = ke.Vi(points).weightedsqdist(ke.Vj(self.centers), ke.Vj(self.covs_inv))
        return self.coefs * (-dists).exp()

    def log_likelihoods(self, points):
        """Log-density, sampled on a given point cloud."""
        self.update_covariances()
        K_ij = -Vi(points).weightedsqdist(Vj(self.centers), Vj(self.params["gamma"]))
        return K_ij.logsumexp(dim=1, weight=Vj(self.weights()))

    def neglog_likelihood(self, points):
        """Returns -log(likelihood(points)) up to an additive factor."""
        ll = self.log_likelihoods(points)
        log_likelihood = torch.mean(ll)
        # N.B.: We add a custom sparsity prior, which promotes empty clusters
        #       through a soft, concave penalization on the class weights.
        return -log_likelihood + self.sparsity * F.softmax(self.w, 0).sqrt().mean()

    def plot(self, points):
        """Displays the model."""
        plt.clf()
        # Heatmap:
        heatmap = self.likelihoods(grid)
        heatmap = (
            heatmap.view(res, res).data.cpu().numpy()
        )  # reshape as a "background" image

        scale = np.amax(np.abs(heatmap[:]))
        plt.imshow(
            -heatmap,
            interpolation="bilinear",
            origin="lower",
            vmin=-scale,
            vmax=scale,
            cmap=cm.RdBu,
            extent=(0, 1, 0, 1),
        )

        # Log-contours:
        log_heatmap = self.log_likelihoods(grid)
        log_heatmap = log_heatmap.view(res, res).data.cpu().numpy()

        scale = np.amax(np.abs(log_heatmap[:]))
        levels = np.linspace(-scale, scale, 41)

        plt.contour(
            log_heatmap,
            origin="lower",
            linewidths=1.0,
            colors="#C8A1A1",
            levels=levels,
            extent=(0, 1, 0, 1),
        )

        # Scatter plot of the dataset:
        xy = points.data.cpu().numpy()
        plt.scatter(xy[:, 0], xy[:, 1], 100 / len(xy), color="k")


class DetectorMixture(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.mixture = GaussianMixture(**kwargs)
        self.threshold = nn.Parameter(torch.tensor(torch.nan), requires_grad=False)
    
    def get_extra_state(self):
        return (self.mixture.get_params(), self.mixture.model_.state_dict())

    def set_extra_state(self, extra_state: Dict):
        assert len(extra_state) == 2, extra_state
        self.mixture.set_params(extra_state[0])
        self.mixture.model_.load_state_dict(extra_state[1])

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
    train_points, val_points, val_targets = data2d.hollow(5000, 5000, device, 100)
    whiten = train.Whiten(train_points[0]).fit(train_points)
    train_points, val_points = whiten(train_points), whiten(val_points)
    val_targets = val_targets.detach().cpu()

    n_runs = 1
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
    # data2d.scatter_outputs_targets(
    #     train_points,
    #     detector(train_points),
    #     torch.ones(len(train_points), dtype=torch.bool),
    #     "Training data",
    #     # There doesn't seem to be a nice way to get the means
    #     centers=detector.mixture.model_.means,
    # )
    # data2d.scatter_outputs_targets(
    #     val_points,
    #     val_outputs,
    #     val_targets,
    #     "pycave GMM validation",
    #     centers=detector.mixture.model_.means,
    # )
    # eval.plot_distr_overlap(
    #     val_outputs[val_targets],
    #     val_outputs[~val_targets],
    #     "Validation positive",
    #     "negative point thresholded densities",
    # )
    plt.show()
