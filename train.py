import gc
import math
from copy import deepcopy

import git
import torch
import torch.nn.functional as F
from torch import nn, linalg

from eval import acc, percent, print_multi_acc
from lsuv import LSUV_

# FullyConnected, 50k training images, 0.984 max validation accuracy
# PoolNet BatchNorm, 20k training images, 0.99 max validation accuracy


def git_commit():
    """Can look at exact parameters and data that a model was trained on using the commit."""

    repo = git.Repo()
    if repo.is_dirty():
        if repo.active_branch.name == "main":
            # Don't clutter the main branch.
            print("NOTE: No automated commit, because on main branch")
        else:
            repo.git.add(".")
            repo.git.commit("-m", "_Automated commit")

    return repo.head.object.hexsha


def save(model, model_name):
    last_commit = git_commit()
    torch.save(model.state_dict(), f"models/{model_name}-{last_commit}.pt")


def load(model, filename):
    state_dict = torch.load(f"models/{filename}.pt")
    model.load_state_dict(state_dict)


class Normalize(nn.Module):
    def __init__(self, example_x):
        super().__init__()
        n_channels = len(example_x)
        d = example_x.device
        self.shift = nn.Parameter(torch.zeros(n_channels, device=d), requires_grad=False)
        self.scale = nn.Parameter(torch.ones(n_channels, device=d), requires_grad=False)
        self.n_warm = 0

    def mean_var(self, train_X, channelwise):
        n_new = len(train_X)
        if n_new < 2:
            return

        if channelwise:
            # Average over each channel -- channel dimension first
            train_X = train_X.transpose(0, 1).view(train_X.size(1), -1)
        else:
            # Average over all data -- single fake channel
            train_X = train_X.view(1, -1)

        if self.n_warm == 0:
            torch.mean(train_X, 1, out=self.shift)
        else:
            self.shift.mul_(self.n_warm / (self.n_warm + n_new))
            self.shift.add_(train_X.sum(1), alpha=1.0 / (self.n_warm + n_new))

        var_sum = (train_X - self.shift.view(-1, 1)).pow(2).sum(-1)
        assert (var_sum >= 0.0).all(), var_sum
        if self.n_warm == 0:
            torch.mul(var_sum, 1.0 / (n_new - 1), out=self.scale)
        else:
            self.scale.mul_((self.n_warm - 1) / (self.n_warm + n_new - 1))
            self.scale.add_(var_sum, alpha=1.0 / (self.n_warm + n_new - 1))

        assert (self.scale >= 0.0).all(), self.scale
        self.n_warm += n_new

    def fit(self, train_X, channelwise=True, finish=True):
        self.mean_var(train_X, channelwise)
        if finish:
            self.scale += 1e-10
            self.scale.rsqrt_()
            self.n_warm = 0
        return self

    def forward(self, X):
        size = [1] * X.ndim
        size[1] = len(self.shift)
        return (X - self.shift.view(size)) * self.scale.view(size)


class Whiten(nn.Module):
    def __init__(self, example_x):
        super().__init__()
        n_features = example_x.numel()
        print("whiten n_features:", n_features)
        d = example_x.device
        self.mean = nn.Parameter(torch.zeros(n_features, device=d), requires_grad=False)
        self.w = nn.Parameter(torch.eye(n_features, device=d), requires_grad=False)
        self.n_warm = 0

    def mean_cov(self, train_X):
        """
        Update mean and covariance without actually calculating whitening matrix.

        NOTE: If train_X contains images, this calculates separate means for each pixel location.

        "Formulas for Robust, One-Pass Parallel Computation of Covariances and Arbitrary-Order
        Statistical Moments", Philippe Pebay, 2008
        """

        if len(train_X) < 2:
            return

        train_X = train_X.flatten(1)
        n_new = len(train_X)
        n_total = self.n_warm + n_new

        if self.n_warm == 0:
            self.mean.copy_(train_X.mean(0))  # Mean of each feature
        else:
            self.mean.mul_(self.n_warm / n_total)
            self.mean.add_(train_X.mean(0), alpha=n_new / n_total)

        # TODO: `self.mean` or just `train_X.mean` in batch covariance calculation?
        train_X = train_X - self.mean
        # Reuse `self.w` memory since it will be overwritten anyway after changing cov.
        # cov = torch.cov(train_X, correction=1)
        cov = torch.mm(train_X.T, train_X)

        if self.n_warm == 0:
            cov.mul_(1.0 / (n_total - 1))
            self.w.data = cov
        else:
            self.w.mul_((self.n_warm - 1) / (n_total - 1))
            self.w.add_(cov, alpha=1.0 / (n_total - 1))

        self.n_warm = n_total

    def fit(self, train_X, zca=True, finish=True):
        """Calculate the whitening matrix based on data given so far."""

        self.mean_cov(train_X)
        if not finish:
            return self

        self.n_warm = 0
        # `self.w` contains the covariance matrix
        eig_vals, eig_vecs = linalg.eigh(self.w)
        eig_vals.clamp_(min=1e-10).rsqrt_()
        torch.mul(eig_vals.view(-1, 1).expand_as(eig_vecs), eig_vecs.T, out=self.w)
        if zca:
            torch.mm(eig_vecs, self.w, out=self.w)

        return self

    def forward(self, X):
        return torch.mm(X.flatten(1) - self.mean, self.w.T)


_get_optimizer_warned = set()


def get_optimizer(Optimizer, model, decay=[], no_decay=[], weight_decay=0.0, **kwargs):
    """
    Split parameters of `model` into those that will experience weight decay (weights) and those
    that won't (biases, normalization, embedding) and those that won't be updated at all. Inspired
    by
    https://github.com/karpathy/minGPT/blob/3ed14b2cec0dfdad3f4b2831f2b4a86d11aef150/mingpt/model.py#L136
    """

    decay = ["Conv", "Linear", "Bilinear"] + decay
    no_decay = ["LayerNorm", "BatchNorm", "Embedding"] + no_decay

    decay_params, no_decay_params = [], []
    for module in model.modules():
        module_name = type(module).__name__

        for param_name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue  # Won't be updated.

            elif any(n in module_name for n in no_decay):
                no_decay_params.append(param)  # Don't decay

            elif param_name.endswith("bias"):
                no_decay_params.append(param)  # Don't decay biases

            elif param_name.endswith("weight") and any(n in module_name for n in decay):
                decay_params.append(param)  # Decay weights

            else:
                if (module_name, param_name) not in _get_optimizer_warned:
                    print(
                        "Warning: assuming weight_decay=0 "
                        + f"for module `{module_name}` parameter `{param_name}`"
                    )
                    _get_optimizer_warned.add((module_name, param_name))

                no_decay_params.append(param)

    groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0},
    ]
    optimizer = Optimizer(groups, **kwargs)
    return optimizer


def gradient_noise(model, i_x, initial_variance=0.01):
    with torch.no_grad():
        std = math.sqrt(initial_variance / (1.0 + i_x / 100.0) ** 0.55)
        for param in model.parameters():
            param.grad.add_(torch.randn_like(param.grad), alpha=std)


def logistic_regression(
    net, data, init=False, verbose=False, lr=1e-3, batch_size=128, n_epochs=100, grad_noise=0.0
):
    """
    net: Should output logits for each class (can be single logit for binary classification)
    data: Training and validation inputs and labels, where labels are class indices.
    init: Whether to initialize the net with random weights
    """

    (train_X, train_y), (val_X, val_y) = data
    net.train()
    with torch.no_grad():
        binary = net(train_X[:2]).size(1) == 1
        if binary:
            train_y = train_y.to(train_X.dtype)
            val_y = val_y.to(val_X.dtype) if val_y is not None else val_y

            net_fn = lambda X: net(X).view(-1)
            loss_fn = F.binary_cross_entropy_with_logits
            print_acc = lambda o, y, name: print(
                f"{name} accuracy:", percent(acc((o >= 0.0) == (y >= 0.0)))
            )
        else:
            net_fn = net
            loss_fn = F.cross_entropy
            print_acc = print_multi_acc

    # Restarts seem to increase accuracy on the original validation images, but
    # decrease accuracy on adversarial validation images.
    restarts = True
    min_lr = lr * 5e-2  # Early stop if LR becomes too low
    optimizer = get_optimizer(torch.optim.NAdam, net, weight_decay=1e-7, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, eps=0.0, min_lr=0.5 * min_lr, verbose=True
    )
    min_loss = float("inf")
    min_state = net.state_dict()  # Not worth deepcopying
    min_optim_state = optimizer.state_dict() if restarts else None

    for epoch in range(n_epochs):
        perm = torch.randperm(len(train_X))

        if init and epoch == 0:
            # GPU memory must be large enough to evaluate all validation X without gradients,
            # so we can reuse that as an upper bound on the number of X to give to LSUV.
            n = len(val_X) if val_X is not None else batch_size
            LSUV_(net, train_X[perm[:n]])

        loss, epoch_loss = None, 0.0
        for i_x in range(0, len(train_X), batch_size):
            batch_X = train_X[perm[i_x : i_x + batch_size]]
            batch_outputs = net_fn(batch_X)
            batch_y = train_y[perm[i_x : i_x + batch_size]]

            loss = loss_fn(batch_outputs, batch_y)
            epoch_loss += loss.item()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_noise > 0.0:
                gradient_noise(net, epoch * len(train_X) + i_x, grad_noise)
            optimizer.step()

        n_batches = len(train_X) // batch_size + (len(train_X) % batch_size > 0)
        epoch_loss /= n_batches

        with torch.no_grad():
            net.eval()
            if val_X is not None:
                val_outputs = net_fn(val_X)
                loss = loss_fn(val_outputs, val_y).item()
            else:
                loss = epoch_loss

            if loss <= min_loss:
                min_loss = loss
                min_state = deepcopy(net.state_dict())
                min_optim_state = deepcopy(optimizer.state_dict()) if restarts else None

            prev_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(loss)
            was_plateau = optimizer.param_groups[0]["lr"] < prev_lr

            if verbose or was_plateau or epoch == n_epochs - 1:
                print(f"--- Epoch {epoch} ({len(train_X)//1000}k samples per epoch)")
                print(f"Epoch average training loss: {epoch_loss}")
                print_acc(batch_outputs, batch_y, "Last batch")
                if val_X is not None:
                    print("Validation loss:", loss)
                    print_acc(val_outputs, val_y, "Validation")

            if all(g["lr"] < min_lr for g in optimizer.param_groups):
                break

            if restarts and was_plateau:
                net.load_state_dict(min_state)
                # Reset optimizer state, but not learning rate
                min_optim_state["param_groups"] = optimizer.param_groups
                optimizer.load_state_dict(min_optim_state)

            net.train()
    net.load_state_dict(min_state)
    net.eval()


import gc
import time


if __name__ == "__main__":
    print(torch.cuda.memory_allocated() / 1e6, "X")
    X = torch.rand((100, 1000), dtype=torch.float32, device="cuda")
    print("X size", X.size())

    print(torch.cuda.memory_allocated() / 1e6, "mean")
    X.sub_(X.mean(0))

    print(torch.cuda.memory_allocated() / 1e6, "cov")
    cov = torch.mm(X.T, X)
    cov.mul_(1.0 / (len(X) - 1))
    print("cov size", cov.size())

    print(torch.cuda.memory_allocated() / 1e6, "eig")
    X = None
    t0 = time.time()
    for _ in range(50):
        eigvals, eigvecs = torch.linalg.eigh(cov)
    print("time", time.time() - t0)

    print(torch.cuda.memory_allocated() / 1e6, "exit")
