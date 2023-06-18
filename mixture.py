import math
import time
from copy import deepcopy

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pykeops.torch as ke
import torch
from pycave.bayes import GaussianMixture
from torch import linalg, nn
from torch.nn.functional import softmax
from torch.distributions.multivariate_normal import MultivariateNormal

import data2d
import train
from cluster import cluster_var_pr_, kmeans_
from eval import acc, percent


"""
DetectorMixture is too slow with a large amount of data and isn't significantly more accurate than
DetectorKmeans. Fitting `data2d.hollow` with 100 features takes 8 seconds with 5k training points
and 300 seconds with 50k training points. DetectorMixture gets accuracy ~71.5% and DetectorKmeans
can get 71% with a few retries (retries which we can afford to do, because DetectorKmeans is
several orders of magnitude faster than DetectorMixture).

-------------------------------- Run 0 --------------------------------
len X_train_pos, X_train_neg, X_val_pos, X_val_neg: 4594 406 4642 358
n_centers: 185
Fitting K-means estimator...
Running initialization...
Epoch 368: 100%|
...
fit time: 8.077536821365356
Predicting: 100%|
...
Balanced validation accuracy: 71.28% +- nan%
True positive rate, true negative rate: 97.03% 45.53%

-------------------------------- Run 0 --------------------------------
len X_train_pos, X_train_neg, X_val_pos, X_val_neg: 46059 3941 46225 3775
n_centers: 1844
Fitting K-means estimator...
Running initialization...
Epoch 3686: 100%|
...
fit time: 306.76016759872437
Predicting: 100%|
...
Balanced validation accuracy: 71.67% +- nan%
True positive rate, true negative rate: 98.81% 44.53%

-------------------------------- Run 0 --------------------------------
len X_train_pos, X_train_neg, X_val_pos, X_val_neg: 46185 3815 46083 3917
n_centers: 1849
/home/mets/Code/nic/cluster.py:80: UserWarning: scatter_reduce()...
fit time: 1.6404266357421875
Balanced validation accuracy: 71.09% +- nan%
True positive rate, true negative rate: 99.99% 42.2%
"""


class DetectorKe(nn.Module):
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

    def likelihood(self, X):
        """
        Returns density at each point (up to a constant factor depending on the number of features)
        """

        d_ij = -0.5 * ke.Vi(X).weightedsqdist(
            ke.Vj(self.center.data), ke.Vj(self.cov_inv.flatten(1))
        )
        return d_ij.exp().matvec(self.coef).view(-1)

    def log_likelihood(self, X):
        """
        Returns density at each point (up to a constant term depending on the number of features)
        """

        d_ij = -0.5 * ke.Vi(X).weightedsqdist(
            ke.Vj(self.center.data), ke.Vj(self.cov_inv.flatten(1))
        )
        return d_ij.logsumexp(dim=1, weight=ke.Vj(self.coef[:, None])).view(-1)

    def net_ll(self, X_pos, X_neg):
        """Net log-likelihood, for maximizing balanced detection accuracy"""

        ll_pos = self.log_likelihood(X_pos)
        if X_neg is None:
            return ll_pos.mean()

        ll_neg = self.log_likelihood(X_neg)
        with torch.no_grad():
            # For detection, it's important that `ll_pos >= ll_neg + margin` and
            # `ll_neg <= ll_pos - margin`, so `ll_pos > ll_neg.max() + margin` is just as good as
            # `ll_pos = ll_neg.max() + margin` and `ll_neg < ll_pos.min() - margin` is just as good
            # as `ll_neg = ll_pos.min() - margin`.
            edge_pos = ll_pos.min() - 2.
            edge_neg = ll_neg.max() + 2.
            low, high = min(edge_pos, edge_neg), max(edge_pos, edge_neg)
        return ll_pos.clamp(max=high).mean() - ll_neg.clamp(min=low).mean()

    def forward(self, X):
        return self.log_likelihood(X) - self.threshold

    def fit(
        self,
        X_train_pos,
        X_train_neg=None,
        X_val_pos=None,
        X_val_neg=None,
        n_epochs=500,
        sparsity=0,
        plot=False,
    ):
        early_stop = 10000  # Early stop if loss hasn't decreased for this many epochs
        val_interval = 5  # Validate/update min loss every this many epochs

        with torch.no_grad():
            self.center.copy_(kmeans(X_train_pos, len(self.center)))
            cov = self.cov_inv_sqrt  # cov has same shape as cov_inv_sqrt
            cluster_covs_weights_(cov, self.weight, X_train_pos, self.center)
            # Increase covariance to account for ignoring points outside of the cluster (for
            # each cluster).
            cov *= 1.2
            if cov.size(1) == 1:
                cov.rsqrt_()
            else:
                # OPTIM: Inverse of triangular matrix after Cholesky decomposition
                self.cov_inv_sqrt.copy_(linalg.inv_ex(linalg.cholesky_ex(cov)[0])[0])
        self.refresh()

        min_loss = float("inf")
        min_state = self.state_dict()  # Not worth deepcopying
        min_epoch = -1

        optimizer = torch.optim.Adam(self.parameters(), lr=0.1)
        if plot:
            losses_train, losses_val = [], ([] if X_val_pos is not None else None)

        for epoch in range(n_epochs):
            optimizer.zero_grad(set_to_none=True)
            reg = sparsity * self.pr.sqrt().mean()
            loss = reg - self.net_ll(X_train_pos, X_train_neg)
            loss.backward()
            optimizer.step()
            self.refresh()
            if plot:
                losses_train.append(loss.item())

            if epoch % val_interval == 0:
                with torch.no_grad():
                    if X_val_pos is not None:
                        loss = -self.net_ll(X_val_pos, X_val_neg).item()
                        if plot:
                            losses_val.append(loss)

                    if loss <= min_loss:
                        min_loss = loss
                        min_state = deepcopy(self.state_dict())
                        min_epoch = epoch

            if epoch - min_epoch >= early_stop:
                break

        self.load_state_dict(min_state)
        threshold = self.log_likelihood(X_train_pos).min()
        if X_train_neg is not None:
            threshold = 0.5 * (threshold + self.log_likelihood(X_train_neg).max())
        print("threshold 1", threshold)
        self.threshold.copy_(threshold)
        print("threshold 2", self.threshold)

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
            heatmap = self.likelihood(self.grid)
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
            log_heatmap = self.log_likelihood(self.grid)
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
        plt.scatter(X[:, 0], X[:, 1], 1000 / len(X), c="green")
        if X_neg is not None:
            X_neg = X_neg.cpu().numpy()
            plt.scatter(X_neg[:, 0], X_neg[:, 1], 1000 / len(X_neg), c="red")

        plt.tight_layout()


class DetectorMixture(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.mixture = GaussianMixture(**kwargs)
        self.threshold = nn.Parameter(torch.tensor(torch.nan), requires_grad=False)
        self.center = None

    def get_extra_state(self):
        return (self.mixture.get_params(), self.mixture.model_.state_dict())

    def set_extra_state(self, extra_state):
        assert len(extra_state) == 2, extra_state
        self.mixture.set_params(extra_state[0])
        self.mixture.model_.load_state_dict(extra_state[1])

    def forward(self, X, batch_size):
        return self.log_likelihood(X, batch_size) - self.threshold

    def log_likelihood(self, X, batch_size):
        """Likelihood of learned distribution at each point (ignoring which component it's from)"""
        # TODO: pycave can only handle input lengths that are a multiple of batch_size
        if len(X) > batch_size:
            X = X[:batch_size * (len(X) // batch_size)]
        return -self.mixture.score_samples(X).flatten()

    def fit(self, X_train, batch_size):
        self.fit_predict(X_train, batch_size)
        return self

    def fit_predict(self, X_train, batch_size):
        if len(X_train) > batch_size:
            X_train = X_train[:batch_size * (len(X_train) // batch_size)]
        self.mixture.fit(X_train)
        log_likelihood = self.log_likelihood(X_train, batch_size)
        self.threshold.copy_(log_likelihood.min())
        # There doesn't seem to be a nice way to get the means
        self.center = self.mixture.model_.means
        return log_likelihood - self.threshold


def all_params(model):
    return [(n, p.data.detach().clone()) for (n, p) in model.named_parameters()]


def grad_params(model):
    return [(n, p.data.detach().clone()) for (n, p) in model.named_parameters() if p.requires_grad]


def set_params(model, params):
    if params is not None:
        for name, param in params:
            getattr(model, name).data = param


class DetectorKmeans(nn.Module):
    def __init__(self, x_example, n_centers):
        super().__init__()

        n_features, dtype, device = len(x_example), x_example.dtype, x_example.device
        self.center = nn.Parameter(
            torch.empty(n_centers, n_features, dtype=dtype, device=device), requires_grad=False
        )
        self.var = nn.Parameter(
            torch.empty(n_centers, dtype=dtype, device=device), requires_grad=False
        )
        self.pr = nn.Parameter(
            torch.empty(n_centers, dtype=dtype, device=device), requires_grad=False
        )
        self.threshold = nn.Parameter(
            torch.empty(1, dtype=dtype, device=device), requires_grad=False
        )
    
    def density(self, X):
        diff_ij = ke.LazyTensor(X[:, None, :]) - ke.LazyTensor(self.center[None, :, :])

        # Cauchy-like:
        # var = ke.LazyTensor(self.var[None, :, None])
        # return (1. / (var + (diff_ij**2).sum(2))) @ (self.var**2 * self.pr)

        # Gaussian:
        d_ij = -0.5 * ke.Vi(X).weightedsqdist(
            ke.Vj(self.center.data), ke.LazyTensor(1. / self.var[None, :, None])
        )
        return d_ij.exp().matvec(self.pr).view(-1)

    def forward(self, X):
        return self.density(X) - self.threshold
    
    def fit(self, *args, **kwargs):
        self.fit_predict(*args, **kwargs)
        return self
    
    def fit_predict(
        self, X_train_pos, X_train_neg=None, X_val_pos=None, X_val_neg=None, n_retries=4
    ):
        """
        It can be better to leave X_train_neg=None
        
        Retries only if validation data is available
        """

        params = None
        best_acc = 0.

        outputs_train_pos = None
        with torch.no_grad():
            for _ in range(n_retries):
                kmeans_(self.center, X_train_pos)
                cluster_var_pr_(self.var, self.pr, X_train_pos, self.center)
                
                outputs_train_pos = self.density(X_train_pos)
                if X_train_neg is not None:
                    # The arithmetic mean `(a+b)/2` and geometric mean `sqrt(a*b)` give surprisingly
                    # bad results, but the reciprocal mean `2 / (1/a + 1/b)` is sometimes better;
                    # maybe because it's the arithmetic mean of square distances.
                    self.threshold.copy_(2. / (
                        1. / outputs_train_pos.min() + 1. / self.density(X_train_neg).max()
                    ))
                else:
                    self.threshold.copy_(self.density(X_train_pos).min())

                if n_retries > 1 and X_val_pos is not None and X_val_neg is not None:
                    pos_acc = acc(self.density(X_val_pos) > self.threshold)
                    neg_acc = acc(self.density(X_val_neg) <= self.threshold)
                    val_acc = 0.5 * (pos_acc + neg_acc)
                    if val_acc > best_acc:
                        params = all_params(self)
                        best_acc = val_acc
        
        set_params(self, params)
        if params is not None:  
            outputs_train_pos = self.density(X_train_pos)
        return outputs_train_pos - self.threshold


if __name__ == "__main__":
    device = "cuda"
    fns = [data2d.spiral]  # data2d.hollow, data2d.circles, data2d.triangle, data2d.line]

    n_runs = 4
    accs_on_pos, accs_on_neg = torch.zeros(n_runs), torch.zeros(n_runs)
    for run in range(n_runs):
        print(f"-------------------------------- Run {run} --------------------------------")

        fn = fns[run % len(fns)]
        X_train_pos, X_train_neg, X_val_pos, X_val_neg = fn(5000, 5000, device)
        print(
            f"len X_train_pos, X_train_neg, X_val_pos, X_val_neg:",
            len(X_train_pos), len(X_train_neg), len(X_val_pos), len(X_val_neg)
        )
        
        n_centers = 2 + len(X_train_pos) // 25
        print(f"n_centers: {n_centers}")
        
        start = time.time()
        # batch_size = 1000
        # detector = DetectorMixture(
        #     num_components=n_centers,
        #     covariance_type="spherical",
        #     init_strategy="kmeans",
        #     batch_size=batch_size,
        #     trainer_params={"gpus": 1}
        # ).fit(X_train_pos, batch_size)
        # detector = DetectorKe(
        #     X_train_pos[0],
        #     n_centers,
        #     equal_clusters=True,
        #     full_cov=False
        # ).fit(X_train_pos, n_epochs=200, sparsity=0, plot=False)
        detector = DetectorKmeans(X_train_pos[0], n_centers).fit(
            X_train_pos, None, X_val_pos, X_val_neg
        )
        print("fit time:", time.time() - start)

        outputs_pos, outputs_neg = detector(X_val_pos), detector(X_val_neg)
        accs_on_pos[run], accs_on_neg[run] = acc(outputs_pos >= 0), acc(outputs_neg < 0)

    print("Expecting equal clusters:", getattr(detector, "equal_clusters", None))
    balanced_accs = 0.5 * (accs_on_pos + accs_on_neg)
    print(
        "Balanced validation accuracy; min, max:",
        percent(balanced_accs.mean()), percent(balanced_accs.min()), percent(balanced_accs.max())
    )
    print(
        "True positive rate, true negative rate:",
        percent(accs_on_pos.mean()), percent(accs_on_neg.mean())
    )

    # More detailed results from the last run
    # print("Likelihood threshold:", detector.threshold)
    data2d.scatter_outputs_y(
        X_train_pos,
        detector(X_train_pos),
        X_train_neg,
        detector(X_train_neg),
        f"{type(detector).__name__} training",
        centers=detector.center,
    )
    data2d.scatter_outputs_y(
        X_val_pos,
        outputs_pos,
        X_val_neg,
        outputs_neg,
        f"{type(detector).__name__} validation",
        centers=detector.center,
    )
    # eval.plot_distr_overlap(
    #     val_outputs[y_val],
    #     val_outputs[~y_val],
    #     "Validation positive",
    #     "negative point thresholded densities",
    # )
    plt.show()
