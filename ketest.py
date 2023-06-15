import math

import matplotlib.cm as cm
import numpy as np
import pykeops.torch as ke
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.nn.functional import softmax


class GaussianMixture(nn.Module):
    def __init__(self, example_input, n_centers, equal_clusters=True):
        super().__init__()

        n_features, dtype, device = len(example_input), example_input.dtype, example_input.device
        self.centers = torch.rand(n_centers, n_features, dtype=dtype, device=device)
        self.covs_inv_sqrt = (
            (15 * torch.eye(n_features, n_features, dtype=dtype, device=device))
            .expand(n_centers, n_features, n_features)
            .contiguous()
        )
        self.weights = torch.ones(n_centers, dtype=dtype, device=device)
        self.centers.requires_grad, self.covs_inv_sqrt.requires_grad, self.weights.requires_grad = (
            True,
            True,
            True,
        )

        # Whether clusters are approximately equally probable (--> don't use softmax):
        self.equal_clusters = nn.Parameter(torch.tensor(equal_clusters), requires_grad=False)
        # Keep boolean parameters and the threshold on the CPU.
        self.threshold = nn.Parameter(torch.tensor(torch.nan, dtype=dtype), requires_grad=False)

        self.covs_inv, self.center_prs, self.coefs = None, None, None
        self.refresh()

        self.grid = None
    
    def get_extra_state(self):
        return 1  # Make sure `set_extra_state` is called
    
    def set_extra_state(self, _):
        assert not self.threshold.isnan().item(), self.threshold  # Other state was already loaded
        self.refresh()

    def refresh(self):
        """Update intermediate variables when the model's parameters change."""

        self.covs_inv = torch.matmul(self.covs_inv_sqrt, self.covs_inv_sqrt.transpose(1, 2))

        if self.equal_clusters.item():
            weights = self.weights.abs()
            self.center_prs = weights / (weights.sum() + 1e-30)
        else:
            self.center_prs = softmax(self.weights, 0)

        self.coefs = self.center_prs * self.covs_inv.det().sqrt()

    def likelihoods(self, points):
        """
        Returns density at each point (up to a constant factor depending on the number of features)
        """

        d_ij = -0.5 * ke.Vi(points).weightedsqdist(
            ke.Vj(self.centers), ke.Vj(self.covs_inv.flatten(1))
        )
        return d_ij.exp().matvec(self.coefs)

    def log_likelihoods(self, points):
        """
        Returns density at each point (up to a constant term depending on the number of features)
        """

        d_ij = -0.5 * ke.Vi(points).weightedsqdist(
            ke.Vj(self.centers), ke.Vj(self.covs_inv.flatten(1))
        )
        return d_ij.logsumexp(dim=1, weight=ke.Vj(self.coefs[:, None]))

    def forward(self, points):
        return self.log_likelihoods(points) - self.threshold

    def loss(self, points, sparsity):
        return -self.log_likelihoods(points).mean() + sparsity * self.center_prs.sqrt().mean()

    def plot(self, points):
        if self.grid is None:
            # Create a uniform grid on the unit square
            res = 200
            ticks = torch.linspace(0.5/res, 1. - 0.5/res, res, dtype=points.dtype, device=device)
            grid0 = ticks.view(res, 1, 1).expand(res, res, 1)
            grid1 = ticks.view(1, res, 1).expand(res, res, 1)
            self.grid = torch.cat((grid1, grid0), dim=-1).view(-1, 2).to(device, points.dtype)

        plt.figure(figsize=(8, 8))
        plt.title("Density, iteration " + str(it), fontsize=20)
        plt.axis("equal")
        plt.axis([0, 1, 0, 1])
        plt.tight_layout()

        # Heatmap
        res = int(math.sqrt(len(self.grid)))
        with torch.no_grad():
            heatmap = self.likelihoods(self.grid)
        heatmap = heatmap.view(res, res).cpu().numpy()  # reshape as a "background" image

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

        # Log-contours
        with torch.no_grad():
            log_heatmap = self.log_likelihoods(self.grid)
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

        # Scatter plot of the dataset
        xy = points.cpu().numpy()
        plt.scatter(xy[:, 0], xy[:, 1], 100 / len(xy), color="k")


# Choose the storage place for our data : CPU (host) or GPU (device) memory.
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
n_train = 10000  # Number of samples
angle = torch.linspace(0, 2 * np.pi, n_train + 1, device=device)[:-1]
train_points = torch.stack((0.5 + 0.4 * (angle / 7) * angle.cos(), 0.5 + 0.3 * angle.sin()), 1)
train_points = train_points + 0.02 * torch.randn(train_points.shape, device=device)

# Model
model = GaussianMixture(train_points[0], 30, equal_clusters=True)
print("Expecting equal clusters:", model.equal_clusters.item())

# Train
optimizer = torch.optim.Adam([model.covs_inv_sqrt, model.weights, model.centers], lr=0.1)
losses = np.zeros(501)

for it in range(501):
    optimizer.zero_grad()  # Reset the gradients (PyTorch syntax...).
    loss = model.loss(train_points, sparsity=20)  # Cost to minimize.
    loss.backward()  # Backpropagate to compute the gradient.
    optimizer.step()
    model.refresh()
    losses[it] = loss.item()

    # if it in [0, 10, 100, 150, 250, 500]:
    if it in [500]:
        model.plot(train_points)

with torch.no_grad():
    print("Final log likelihood:", model.log_likelihoods(train_points).mean())

plt.figure()
plt.plot(losses)
plt.tight_layout()
plt.show()
