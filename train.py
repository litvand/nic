import math
from copy import deepcopy

import git
import torch
import torch.nn.functional as F
from torch import nn, linalg

import eval
from lsuv import LSUV_

# FullyConnected, 50k training images, 0.984 max validation accuracy
# PoolNet BatchNorm, 20k training images, 0.99 max validation accuracy


def git_commit():
    """Can look at exact parameters and data that a model was trained on using the commit."""

    repo = git.Repo()
    if repo.active_branch.name == "main":
        # Don't clutter the main branch.
        print("NOTE: No automated commit, because on main branch")
    else:
        repo.git.add(".")
        repo.git.commit("-m", "_Automated commit")

    return repo.head.object.hexsha


def save(model, model_name):
    last_commit = git_commit()
    torch.save(model.cpu().state_dict(), f"models/{model_name}-{last_commit}.pt")


def load(model, filename):
    state_dict = torch.load(f"models/{filename}.pt")
    model.load_state_dict(state_dict)

import gc

def activations_at(sequential, X, module_indices):
    """Get activations from modules inside an `nn.Sequential` at indices in `module_indices`."""

    sequential = list(sequential.children())
    activations = []
    for i_module, module in enumerate(sequential):
        gc.collect()
        torch.cuda.empty_cache()

        X = module(X)
        # Support negative indices
        if i_module in module_indices or i_module - len(sequential) in module_indices:
            activations.append(X)

    assert len(activations) == len(module_indices), (
        activations,
        sequential,
        module_indices,
    )
    return activations


class Normalize(nn.Module):
    def __init__(self, example_x):
        super().__init__()
        n_channels = len(example_x)
        d = example_x.device
        self.shift = nn.Parameter(torch.zeros(n_channels, device=d), requires_grad=False)
        self.inv_scale = nn.Parameter(torch.ones(n_channels, device=d), requires_grad=False)

    def fit(self, train_X, unit_range=False):
        train_X = train_X.transpose(0, 1).flatten(1)  # Channel dimension first

        if unit_range:  # Each channel in the range [0, 1] (in training data)
            self.shift.copy_(train_X.min(1)[0])  # Min of each channel (in training data)
            self.inv_scale.copy_(1.0 / (train_X.max(1)[0] - self.shift))
        else:
            self.shift.copy_(train_X.mean(1))  # Mean of each channel
            self.inv_scale.copy_(1.0 / train_X.std(1))

        return self

    def forward(self, X):
        size = [1] * X.ndim
        size[1] = len(self.shift)
        return (X - self.shift.expand(size)) * self.inv_scale.expand(size)


class Whiten(nn.Module):
    def __init__(self, example_x):
        super().__init__()
        n_features = example_x.numel()
        d = example_x.device
        self.mean = nn.Parameter(torch.zeros(n_features, device=d), requires_grad=False)
        self.w = nn.Parameter(torch.zeros(n_features, n_features, device=d), requires_grad=False)

    def fit(self, train_X, zca=True):
        """
        NOTE: If train_X contains images, this calculates separate means for each pixel location.

        train_X: Training inputs
        zca: True --> ZCA, False --> PCA
        """

        train_X = train_X.flatten(1)
        self.mean.copy_(train_X.mean(0))

        cov = torch.cov((train_X - self.mean).T)
        eig_vals, eig_vecs = linalg.eigh(cov)
        torch.mm(eig_vals.clamp(min=1e-6).rsqrt_().diag(), eig_vecs.T, out=self.w)
        if zca:
            self.w.copy_(eig_vecs.mm(self.w))

        return self

    def forward(self, X):
        return torch.mm(X.flatten(1) - self.mean, self.w.T)


_get_optimizer_warned = set()


def get_optimizer(Optimizer, model, decay=[], no_decay=[], weight_decay=0., **kwargs):
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
        std = math.sqrt(initial_variance / (1 + i_x / 100) ** 0.55)
        for param in model.parameters():
            param.grad.add_(torch.randn_like(param.grad), alpha=std)


def logistic_regression(net, data, init=False, batch_size=150, n_epochs=1000):
    """
    net: Should output logits for each class (can be single logit for binary classification)
    data: Training and validation inputs and targets, where targets are class indices.
    init: Whether to initialize the net with random weights
    """

    (train_X, train_y), (val_X, val_y) = data
    net.train()
    with torch.no_grad():
        output_len = net(train_X[:2]).size(1)
        loss_fn = F.cross_entropy if output_len > 1 else F.binary_cross_entropy_with_logits

    # Restarts seem to increase accuracy on the original validation images, but
    # decrease accuracy on adversarial validation images.
    restarts = True
    min_lr = 5e-6  # Early stop if LR becomes too low
    optimizer = get_optimizer(torch.optim.NAdam, net, weight_decay=1e-7, lr=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, eps=0, min_lr=min_lr / 2, verbose=True
    )
    min_loss = float("inf")
    min_state = net.state_dict()  # Not worth deepcopying
    min_optim_state = optimizer.state_dict() if restarts else None

    for epoch in range(n_epochs):
        perm = torch.randperm(len(train_X))
        train_X, train_y = train_X[perm], train_y[perm]

        if init and epoch == 0:
            # GPU memory must be large enough to evaluate all validation X without gradients,
            # so we can reuse that as an upper bound on the number of X to give to LSUV.
            n = len(val_X) if val_X is not None else batch_size
            LSUV_(net, train_X[:n])

        loss_avg, coef_avg = 0., batch_size / len(train_X)
        loss = None
        for i_x in range(0, len(train_X), batch_size):
            batch_X = train_X[i_x : i_x + batch_size]
            batch_outputs = net(batch_X)
            batch_y = train_y[i_x : i_x + batch_size]

            loss = loss_fn(batch_outputs, batch_y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_noise(net, i_x)
            optimizer.step()

            loss_avg = loss_avg * (1. - coef_avg) + loss.item() * coef_avg

        with torch.no_grad():
            net.eval()
            print(f"Epoch {epoch} ({len(train_X)//1000}k samples per epoch)")
            print(f"Training loss running average {loss_avg}")
            eval.print_multi_acc(batch_outputs, batch_y, "Last batch")

            if val_X is not None:
                val_outputs = net(val_X)
                loss = loss_fn(val_outputs, val_y).item()
                print("Validation loss:", loss)
                eval.print_multi_acc(val_outputs, val_y, "Validation")
            else:
                loss = loss_avg

            if loss <= min_loss:
                min_loss = loss
                min_state = deepcopy(net.state_dict())
                min_optim_state = deepcopy(optimizer.state_dict()) if restarts else None

            prev_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(loss)
            if all(g["lr"] < min_lr for g in optimizer.param_groups):
                break
            elif restarts and optimizer.param_groups[0]["lr"] < prev_lr:
                net.load_state_dict(min_state)
                # Reset optimizer state, but not learning rate
                min_optim_state["param_groups"] = optimizer.param_groups
                optimizer.load_state_dict(min_optim_state)

            net.train()
    net.load_state_dict(min_state)
    net.eval()
