import gc
import math
import time
from copy import deepcopy

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pykeops.torch as ke
import torch
# from pycave.bayes import GaussianMixture
from torch import linalg, nn
from torch.nn.functional import softmax
from torch.distributions.multivariate_normal import MultivariateNormal

import data2d
import train
from cluster import cluster_var_pr_, kmeans_
from eval import acc, percent, round_tensor


"""
DetectorMixture is too slow with a large amount of data and isn't significantly more accurate than
DetectorKmeans. Fitting `data2d.hollow` with 100 features takes 8 seconds with 5k training points
and 300 seconds with 50k training points. DetectorMixture gets accuracy ~71.5% and DetectorKmeans
can get 71% with a few retries (retries which we can afford to do, because DetectorKmeans is
several orders of magnitude faster than DetectorMixture).

-------------------------------- Run 0 --------------------------------
len train_X_pos, train_X_neg, val_X_pos, val_X_neg: 4594 406 4642 358
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
len train_X_pos, train_X_neg, val_X_pos, val_X_neg: 46059 3941 46225 3775
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
len train_X_pos, train_X_neg, val_X_pos, val_X_neg: 46185 3815 46083 3917
n_centers: 1849
/home/mets/Code/nic/cluster.py:80: UserWarning: scatter_reduce()...
fit time: 1.6404266357421875
Balanced validation accuracy: 71.09% +- nan%
True positive rate, true negative rate: 99.99% 42.2%
"""

"""
-------------------------------- Run 4 --------------------------------
len train_X_pos, train_X_neg, val_X_pos, val_X_neg: 2488 2488 2512 2488
n_centers: 101
Fitting K-means estimator...
Running initialization...
...
fit time: 3.4230003356933594
...
<function overlap at 0x7ff57c24a8e0> 99.6% 4.25%
-------------------------------- Summary --------------------------------
Expecting equal clusters: None
Balanced validation accuracy; min, max: 51.83% 51.25% 52.58%
True positive rate, true negative rate: 99.45% 4.21%
"""

"""
Only positive examples:

-------------------------------- Run 4 --------------------------------
len train_X_pos, train_X_neg, val_X_pos, val_X_neg: 2498 2498 2502 2498
n_centers: 101
fit time: 0.07199597358703613
<function overlap at 0x7fe9fed0e840> 99.64% 4.12%
-------------------------------- Summary --------------------------------
Expecting equal clusters: None
Balanced validation accuracy; min, max: 51.8% 51.11% 53.18%
True positive rate, true negative rate: 99.71% 3.88%
"""

"""
Threshold equal to geometric mean of positive and negative examples:

-------------------------------- Run 4 --------------------------------
len train_X_pos, train_X_neg, val_X_pos, val_X_neg: 25097 24903 24903 24903
n_centers: 1005
fit time: 0.10455560684204102
<function overlap at 0x7f1697542840> 99.73% 4.58%
-------------------------------- Summary --------------------------------
Expecting equal clusters: None
Balanced validation accuracy [min-max]: 51.88% [51.05%-52.64%]
True positive rate, true negative rate: 99.59% 4.17%
"""


class DetectorKe(nn.Module):
    def __init__(self, example_x, n_centers, equal_clusters=True, full_cov=True):
        super().__init__()

        n_features, dtype, device = len(example_x), example_x.dtype, example_x.device
        self.centers = nn.Parameter(torch.rand(n_centers, n_features, dtype=dtype, device=device))

        # OPTIM: Custom code for spherical clusters without full covariance
        c = n_features if full_cov else 1
        self.cov_inv_sqrt = nn.Parameter(torch.empty(n_centers, c, c, dtype=dtype, device=device))
        self.weight = nn.Parameter(torch.ones(n_centers, dtype=dtype, device=device))

        # Whether clusters are approximately equally probable (--> don't use softmax):
        self.equal_clusters = nn.Parameter(torch.tensor(equal_clusters), requires_grad=False)
        # Keep boolean parameters and the threshold on the CPU.
        self.threshold = nn.Parameter(torch.tensor(torch.nan, dtype=dtype), requires_grad=False)

        self.cov_inv, self.prs, self.coef = None, None, None
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
            self.prs = weight / weight.sum()
        else:
            self.prs = softmax(self.weight, 0)

        self.cov_inv = torch.matmul(self.cov_inv_sqrt, self.cov_inv_sqrt.transpose(1, 2))
        self.coef = self.prs * self.cov_inv.det().sqrt()

    def likelihood(self, X):
        """
        Returns density at each point (up to a constant factor depending on the number of features)
        """

        d_ij = -0.5 * ke.Vi(X).weightedsqdist(
            ke.Vj(self.centers.data), ke.Vj(self.cov_inv.flatten(1))
        )
        return d_ij.exp().matvec(self.coef).view(-1)

    def log_likelihood(self, X):
        """
        Returns density at each point (up to a constant term depending on the number of features)
        """

        d_ij = -0.5 * ke.Vi(X).weightedsqdist(
            ke.Vj(self.centers.data), ke.Vj(self.cov_inv.flatten(1))
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
            edge_pos = ll_pos.min() - 2.0
            edge_neg = ll_neg.max() + 2.0
            low, high = min(edge_pos, edge_neg), max(edge_pos, edge_neg)
        return ll_pos.clamp(max=high).mean() - ll_neg.clamp(min=low).mean()

    def forward(self, X):
        return self.log_likelihood(X) - self.threshold

    def fit(
        self,
        train_X_pos,
        train_X_neg=None,
        val_X_pos=None,
        val_X_neg=None,
        n_epochs=500,
        sparsity=0,
        plot=False,
    ):
        early_stop = 10000  # Early stop if loss hasn't decreased for this many epochs
        val_interval = 5  # Validate/update min loss every this many epochs

        with torch.no_grad():
            self.centers.copy_(kmeans(train_X_pos, len(self.centers)))
            cov = self.cov_inv_sqrt  # cov has same shape as cov_inv_sqrt
            cluster_covs_weights_(cov, self.weight, train_X_pos, self.centers)
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
            losses_train, losses_val = [], ([] if val_X_pos is not None else None)

        for epoch in range(n_epochs):
            optimizer.zero_grad(set_to_none=True)
            reg = sparsity * self.prs.sqrt().mean()
            loss = reg - self.net_ll(train_X_pos, train_X_neg)
            loss.backward()
            optimizer.step()
            self.refresh()
            if plot:
                losses_train.append(loss.item())

            if epoch % val_interval == 0:
                with torch.no_grad():
                    if val_X_pos is not None:
                        loss = -self.net_ll(val_X_pos, val_X_neg).item()
                        if plot:
                            losses_val.append(loss)

                    if loss <= min_loss:
                        min_loss = loss
                        min_state = deepcopy(self.state_dict())
                        min_epoch = epoch

            if epoch - min_epoch >= early_stop:
                break

        self.load_state_dict(min_state)
        threshold = self.log_likelihood(train_X_pos).min()
        if train_X_neg is not None:
            threshold = 0.5 * (threshold + self.log_likelihood(train_X_neg).max())
        print("threshold 1", threshold)
        self.threshold.copy_(threshold)
        print("threshold 2", self.threshold)

        if plot:
            if val_X_pos is not None:
                self.plot(val_X_pos, val_X_neg)
            else:
                self.plot(train_X_pos, train_X_neg)

            if len(losses_train) == 0:
                return self

            losses_train = np.array(losses_train)
            plt.figure()
            plt.title("Epoch training loss")
            plt.tight_layout()
            plt.plot(losses_train)
            min_epoch = losses_train.argmin()
            print("Min training loss, epoch:", losses_train[min_epoch], min_epoch)

            if val_X_pos is not None:
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
        self.mixture = None #GaussianMixture(**kwargs)
        self.threshold = nn.Parameter(torch.tensor(torch.nan), requires_grad=False)
        self.centers = None

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
            X = X[: batch_size * (len(X) // batch_size)]
        return -self.mixture.score_samples(X).flatten()

    def fit(self, train_X, batch_size):
        self.fit_predict(train_X, batch_size)
        return self

    def fit_predict(self, train_X, batch_size):
        if len(train_X) > batch_size:
            train_X = train_X[: batch_size * (len(train_X) // batch_size)]
        self.mixture.fit(train_X)
        log_likelihood = self.log_likelihood(train_X, batch_size)
        self.threshold.copy_(log_likelihood.min())
        # There doesn't seem to be a nice way to get the means
        self.centers = self.mixture.model_.means
        return log_likelihood - self.threshold


def all_params(model):
    return [(n, p.data.detach().clone()) for (n, p) in model.named_parameters()]


def grad_params(model):
    return [(n, p.data.detach().clone()) for (n, p) in model.named_parameters() if p.requires_grad]


def set_params(model, params):
    if params is not None:
        for name, param in params:
            getattr(model, name).data = param


def arithmetic_threshold(densities_pos, densities_neg):
    return 0.5 * (densities_pos.min() + densities_neg.max())


def geometric_threshold(densities_pos, densities_neg):
    return torch.sqrt(densities_pos.min() * densities_neg.max())


def reciprocal_threshold(densities_pos, densities_neg):
    return 2.0 / (1.0 / densities_pos.min() + 1.0 / densities_neg.max())


def true_negatives_threshold(densities_pos, densities_neg, min_acc_on_neg):
    large_neg = torch.quantile(densities_neg, min_acc_on_neg)
    # Classified as positive if density >= threshold, so need threshold > large_neg
    above_neg = densities_pos[densities_pos > large_neg]
    return above_neg.min() if len(above_neg) > 0 else large_neg + 1e-5


class DetectorKmeans(nn.Module):
    def __init__(self, example_x, n_centers):
        super().__init__()

        n_features, dtype, device = len(example_x), example_x.dtype, example_x.device
        self.centers = nn.Parameter(
            torch.empty(n_centers, n_features, dtype=dtype, device=device), requires_grad=False
        )
        self.vars = nn.Parameter(
            torch.empty(n_centers, dtype=dtype, device=device), requires_grad=False
        )
        self.prs = nn.Parameter(
            torch.empty(n_centers, dtype=dtype, device=device), requires_grad=False
        )
        self.threshold = nn.Parameter(
            torch.empty(1, dtype=dtype, device=device), requires_grad=False
        )

    def density(self, X):
        # Cauchy-like:
        # d_ij = ke.Vi(X).sqdist(ke.Vj(self.centers.data)
        # return (1. / (ke.Vj(self.vars[:, None]) + d_ij))).matvec(self.vars**2 * self.prs).view(-1)

        # Gaussian:
        d_ij = ke.Vi(X).weightedsqdist(ke.Vj(self.centers.data), 1.0 / ke.Vj(self.vars[:, None]))
        return (-0.5 * d_ij).logsumexp(dim=1, weight=ke.Vj(self.prs[:, None])).view(-1)

    def forward(self, X):
        return self.density(X) - self.threshold

    def fit(self, *args, **kwargs):
        self.fit_predict(*args, **kwargs)
        return self

    def fit_predict(
        self,
        train_X_pos,
        train_X_neg=None,
        val_X_pos=None,
        val_X_neg=None,
        n_retries=4,
        expected_acc=0.6,
    ):
        """
        It can be better to leave train_X_neg=None. train_X_neg is only used to choose the
        threshold.

        Retries only if validation data is available.
        """

        train_X_pos = train_X_pos.contiguous()
        train_X_neg = train_X_neg.contiguous() if train_X_neg is not None else train_X_neg
        val_X_pos = val_X_pos.contiguous() if val_X_pos is not None else val_X_pos
        val_X_neg = val_X_neg.contiguous() if val_X_neg is not None else val_X_neg

        params = None
        best_acc = 0.0
        best_densities = None
        with torch.no_grad():
            for _ in range(n_retries):
                gc.collect()
                kmeans_(self.centers.data, train_X_pos)
                cluster_var_pr_(self.vars.data, self.prs.data, train_X_pos, self.centers.data)

                train_densities = self.density(train_X_pos)
                nans = train_X_pos[train_densities.isnan()]
                if len(nans) > 0:
                    print(len(train_X_pos), "ERROR: nans", len(nans), nans)
                    print(
                        "[self.vars], [self.prs]",
                        f"[{round_tensor(self.vars.min())}-{round_tensor(self.vars.max())}]",
                        f"[{round_tensor(self.prs.min())}-{round_tensor(self.prs.max())}]",
                    )
                    continue

                if train_X_neg is not None:
                    # The arithmetic mean `(a+b)/2` gives surprisingly bad results, but sometimes
                    # the geometric mean `sqrt(a*b)`or reciprocal mean `2 / (1/a + 1/b)` is better;
                    # maybe when it's the geometric mean of densities or the reciprocal mean of
                    # reciprocal distances.
                    train_densities_neg = self.density(train_X_neg)
                    self.threshold.copy_(
                        true_negatives_threshold(train_densities, train_densities_neg, expected_acc)
                    )
                    train_densities = (train_densities, train_densities_neg)
                else:
                    self.threshold.copy_(train_densities.min())

                if n_retries > 1 and val_X_pos is not None and val_X_neg is not None:
                    pos_acc = acc(self.density(val_X_pos) > self.threshold)
                    neg_acc = acc(self.density(val_X_neg) <= self.threshold)
                    val_acc = 0.5 * (pos_acc + neg_acc)
                    if val_acc > best_acc:
                        params = all_params(self)
                        best_acc = val_acc
                        best_densities = train_densities
                else:
                    best_densities = train_densities

        set_params(self, params)
        if train_X_neg is not None:
            return best_densities[0] - self.threshold, best_densities[1] - self.threshold
        return best_densities - self.threshold


if __name__ == "__main__":
    device = "cpu"
    fns = [data2d.line]
    # fns = [data2d.hollow, data2d.circles, data2d.triangle, data2d.line]

    n_runs = 2
    accs_on_pos, accs_on_neg = torch.zeros(n_runs), torch.zeros(n_runs)
    for run in range(n_runs):
        print(f"-------------------------------- Run {run} --------------------------------")

        fn = fns[run % len(fns)]
        train_X_pos, train_X_neg, val_X_pos, val_X_neg = fn(50000, 50000, device)
        print(
            f"len train_X_pos, train_X_neg, val_X_pos, val_X_neg:",
            len(train_X_pos),
            len(train_X_neg),
            len(val_X_pos),
            len(val_X_neg),
        )

        n_centers = 2 + len(train_X_pos) // 100
        print(f"n_centers: {n_centers}")

        start = time.time()
        # batch_size = 1000
        # detector = DetectorMixture(
        #     num_components=n_centers,
        #     covariance_type="spherical",
        #     init_strategy="kmeans",
        #     batch_size=batch_size,
        #     trainer_params={"gpus": 1}
        # ).fit(train_X_pos, batch_size)
        # detector = DetectorKe(
        #     train_X_pos[0],
        #     n_centers,
        #     equal_clusters=True,
        #     full_cov=False
        # ).fit(train_X_pos, n_epochs=200, sparsity=0, plot=False)
        detector = DetectorKmeans(train_X_pos[0], n_centers).fit(train_X_pos)
        print("fit time:", time.time() - start)

        outputs_pos, outputs_neg = detector(val_X_pos), detector(val_X_neg)
        accs_on_pos[run], accs_on_neg[run] = acc(outputs_pos >= 0), acc(outputs_neg < 0)
        print(str(fn), percent(accs_on_pos[run]), percent(accs_on_neg[run]))

    print("-------------------------------- Summary --------------------------------")
    print("Expecting equal clusters:", getattr(detector, "equal_clusters", None))
    balanced_accs = 0.5 * (accs_on_pos + accs_on_neg)
    print(
        "Balanced validation accuracy [min-max]:",
        percent(balanced_accs.mean()),
        f"[{percent(balanced_accs.min())}-{percent(balanced_accs.max())}]",
    )
    print(
        "True positive rate, true negative rate:",
        percent(accs_on_pos.mean()),
        percent(accs_on_neg.mean()),
    )

    # More detailed results from the last run
    # print("Density threshold:", detector.threshold)
    data2d.scatter_outputs_y(
        train_X_pos,
        detector(train_X_pos),
        train_X_neg,
        detector(train_X_neg),
        f"{type(detector).__name__} training",
        centers=detector.centers,
    )
    data2d.scatter_outputs_y(
        val_X_pos,
        outputs_pos,
        val_X_neg,
        outputs_neg,
        f"{type(detector).__name__} validation",
        centers=detector.centers,
    )
    # eval.plot_distr_overlap(
    #     outputs_pos,
    #     outputs_neg,
    #     "Validation outputs on positive",
    #     "negative points",
    # )
    plt.show()
