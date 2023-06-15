import math
from copy import deepcopy

import matplotlib.cm as cm
import numpy as np
import pykeops.torch as ke
import torch
from matplotlib import pyplot as plt
from torch import nn, linalg
from torch.nn.functional import softmax

import data2d
from cluster import cluster_covs_weights_, kmeans


def get_cov_inv_sqrt_(cov):
    if cov.size(1) == 1:
        return cov.rsqrt_()

    eigvals, eigvecs = linalg.eigh(cov)
    return eigvecs.mm(eigvals.clamp(min=1e-12).rsqrt_().diag().mm(eigvecs.T))



class GaussianMixture(nn.Module):
    def __init__(self, x_example, n_centers, equal_clusters=True, full_cov=True):
        super().__init__()

        n_features, dtype, device = len(x_example), x_example.dtype, x_example.device
        self.center = nn.Parameter(torch.rand(n_centers, n_features, dtype=dtype, device=device))
        
        # OPTIM: Custom code for spherical clusters without full covariance
        c = n_features if full_cov else 1
        self.cov_inv_sqrt = nn.Parameter(torch.empty(n_centers, c, c, dtype=dtype, device=device))
        self.weight = nn.Parameter(torch.ones(n_centers, dtype=dtype, device=device))

        # Whether clusters are approximately equally probable (--> don't use softmax):
        self.equal_clusters = nn.Parameter(torch.tensor(equal_clusters), requires_grad=False)
        # Keep boolean parameters and the threshold on the CPU.
        self.threshold = nn.Parameter(torch.tensor(torch.nan, dtype=dtype), requires_grad=False)

        self.cov_inv, self.pr, self.coef = None, None, None
        self.refresh()

        self.grid = None
    
    def get_extra_state(self):
        return 1  # Make sure `set_extra_state` is called
    
    def set_extra_state(self, _):
        # assert not self.threshold.isnan().item(), self.threshold  # Other state was already loaded
        self.refresh()

    def refresh(self):
        """Update intermediate variables when the model's parameters change."""

        if self.equal_clusters.item():
            weight = self.weight.abs()
            self.pr = weight / weight.sum()
        else:
            self.pr = softmax(self.weight, 0)

        self.cov_inv = torch.matmul(self.cov_inv_sqrt, self.cov_inv_sqrt.transpose(1, 2))
        self.coef = self.pr * self.cov_inv.det().sqrt()

    def likelihoods(self, X):
        """
        Returns density at each point (up to a constant factor depending on the number of features)
        """

        d_ij = -0.5 * ke.Vi(X).weightedsqdist(
            ke.Vj(self.center.data), ke.Vj(self.cov_inv.flatten(1))
        )
        return d_ij.exp().matvec(self.coef)

    def log_likelihoods(self, X):
        """
        Returns density at each point (up to a constant term depending on the number of features)
        """

        d_ij = -0.5 * ke.Vi(X).weightedsqdist(
            ke.Vj(self.center.data), ke.Vj(self.cov_inv.flatten(1))
        )
        return d_ij.logsumexp(dim=1, weight=ke.Vj(self.coef[:, None]))
    
    def net_ll(self, X_pos, X_neg):
        """Net log-likelihood, for maximizing balanced detection accuracy"""

        ll_pos = self.log_likelihoods(X_pos).mean()
        # with torch.no_grad():
        #     max = ll_pos.mean()
        # ll_pos = ll_pos.clamp(max=max).mean()
        if X_neg is None:
            return ll_pos
        
        ll_neg = self.log_likelihoods(X_neg).mean()
        # with torch.no_grad():
        #     min = ll_neg.mean()
        # ll_neg = ll_neg.clamp(min=min).mean()
        return ll_pos - ll_neg

    def forward(self, X):
        return self.log_likelihoods(X) - self.threshold

    def fit(
        self,
        X_train_pos,
        X_train_neg=None,
        X_val_pos=None,
        X_val_neg=None,
        n_epochs=500,
        sparsity=0,
        plot=False
    ):
        early_stop = 10000  # Early stop if loss hasn't decreased for this many epochs
        val_interval = 5  # Validate/update min loss every this many epochs

        with torch.no_grad():
            self.center.copy_(kmeans(X_train_pos, len(self.center)))
            cov = self.cov_inv_sqrt  # cov has same shape as cov_inv_sqrt
            cluster_covs_weights_(cov, self.weight, X_train_pos, self.center)
            # Increase covariance to account for ignoring points outside of the cluster (for
            # each cluster).
            cov *= 1.1
            if cov.size(1) == 1:
                cov.rsqrt_()
            else:
                # OPTIM: Inverse of triangular matrix after Cholesky decomposition
                self.cov_inv_sqrt.copy_(linalg.inv(linalg.cholesky(cov)))
        self.refresh()

        min_loss = float("inf")
        min_state = self.state_dict()  # Not worth deepcopying
        min_epoch = -1

        optimizer = torch.optim.Adam(self.parameters(), lr=0.1)
        if plot:
            losses_train, losses_val = [], ([] if X_val_pos is not None else None)

        for epoch in range(n_epochs):
            optimizer.zero_grad(set_to_none=True)
            loss = sparsity * self.pr.sqrt().mean() - self.net_ll(X_train_pos, X_train_neg)
            loss.backward()
            optimizer.step()
            self.refresh()
            if plot:
                losses_train.append(loss.item())

            if epoch % val_interval == 0:
                with torch.no_grad():
                    if X_val_pos is not None:
                        loss = -self.net_ll(X_val_pos, X_val_neg).item()
                        losses_val.append(loss)
                
                    if loss <= min_loss:
                        min_loss = loss
                        min_state = deepcopy(self.state_dict())
                        min_epoch = epoch
            
            if epoch - min_epoch >= early_stop:
                break
        
        self.load_state_dict(min_state)
        threshold = self.log_likelihoods(X_train_pos).min()
        if X_train_neg is not None:
            threshold = 0.5 * (threshold + self.log_likelihoods(X_train_neg).max())
        self.threshold.copy_(threshold)

        if plot:
            if X_val_pos is not None:
                self.plot(X_val_pos, X_val_neg)
            else:
                self.plot(X_train_pos, X_train_neg)
            
            if len(losses_train) == 0:
                return self

            losses_train = np.array(losses_train)
            plt.figure()
            plt.title("Epoch training loss")
            plt.tight_layout()
            plt.plot(losses_train)
            min_epoch = losses_train.argmin()
            print("Min training loss, epoch:", losses_train[min_epoch], min_epoch)

            if X_val_pos is not None:
                losses_val = np.array(losses_val)
                plt.figure()
                plt.title("Epoch validation loss")
                plt.tight_layout()
                plt.plot(np.arange(len(losses_val)) * val_interval, losses_val)
                i_min = losses_val.argmin()
                print("Min validation loss, epoch:", losses_val[i_min], i_min * val_interval)
        
        return self

    def plot(self, X, X_neg=None):
        with torch.no_grad():
            low, high = X.min().item(), X.max().item()
        margin = 0.5 * (high - low)
        low, high = low - margin, high + margin

        if self.grid is None:
            # Create a uniform grid on the unit square
            res = 200
            ticks = torch.linspace(low, high, res, dtype=X.dtype, device=X.device)
            grid0 = ticks.view(res, 1, 1).expand(res, res, 1)
            grid1 = ticks.view(1, res, 1).expand(res, res, 1)
            self.grid = torch.cat((grid1, grid0), dim=-1).view(-1, 2).to(X.device, X.dtype)

        plt.figure(figsize=(8, 8))
        plt.title("Likelihood", fontsize=20)
        plt.axis("equal")
        plt.axis([low, high, low, high])

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
            extent=(low, high, low, high),
        )

        # Log-contours
        with torch.no_grad():
            log_heatmap = self.log_likelihoods(self.grid)
        log_heatmap = log_heatmap.view(res, res).cpu().numpy()

        scale = np.amax(np.abs(log_heatmap[:]))
        levels = np.linspace(-scale, scale, 81)

        plt.contour(
            log_heatmap,
            origin="lower",
            linewidths=1.0,
            colors="#C8A1A1",
            levels=levels,
            extent=(low, high, low, high),
        )

        # Scatter plot of the dataset
        X = X.cpu().numpy()
        plt.scatter(X[:, 0], X[:, 1], 1000 / len(X), c="k")
        if X_neg is not None:
            X_neg = X_neg.cpu().numpy()
            plt.scatter(X_neg[:, 0], X_neg[:, 1], 1000 / len(X_neg), c="b")
        
        plt.tight_layout()


device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
X_train_pos, X_train_neg, X_val_pos, X_val_neg = data2d.triangle(5000, 5000, device)
n_centers = 1 + len(X_train_pos) // 100
model = GaussianMixture(X_train_pos[0], n_centers, equal_clusters=True, full_cov=True)
model.fit(X_train_pos, n_epochs=0, sparsity=10, plot=True)
print("Expecting equal clusters:", model.equal_clusters.item())
plt.show()
