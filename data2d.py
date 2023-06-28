# 2d data for one-class classification

import math

import matplotlib.pyplot as plt
import numpy as np
import torch

from train import Whiten


def preprocess(n_train, X, y):
    perm = torch.randperm(len(X), device=X.device)
    X, y = X[perm], y[perm]
    train_X, val_X, train_y, val_y = X[:n_train], X[n_train:], y[:n_train], y[n_train:]

    train_X_pos, train_X_neg = train_X[train_y], train_X[~train_y]
    val_X_pos, val_X_neg = val_X[val_y], val_X[~val_y]

    train_X_neg, val_X_neg = train_X_neg[: len(train_X_pos)], val_X_neg[: len(val_X_pos)]

    whiten = Whiten(train_X_pos[0]).fit(train_X_pos, zca=True)
    return tuple(
        None if len(X) == 0 else whiten(X) for X in (train_X_pos, train_X_neg, val_X_pos, val_X_neg)
    )


def overlap(n_train, n_val, device):
    n = n_train + n_val

    X = torch.randn((n, 2), device=device)
    X[: n // 2] += 1.0

    y = torch.zeros(n, dtype=torch.bool, device=device)
    y[: n // 2] = True

    return preprocess(n_train, X, y)


def point(n_train, n_val, device):
    n = n_train + n_val

    X = torch.zeros(n, device=device)  # Half zero
    X[: n // 2] = torch.linspace(0.01, 1, n // 2, device=device)
    X = X.view(-1, 1).expand(-1, 2).clone()

    y = torch.zeros(n, dtype=torch.bool, device=device)
    y[: n // 2 + n // 100 + 1].fill_(True)  # All zeros false except 1%
    return preprocess(n_train, X, y)


def spiral(n_train, n_val, device):
    angle = torch.linspace(0, 2 * np.pi, n_train + n_val + 1, device=device)[:-1]
    X = torch.stack((0.5 + 0.4 * (angle / 7) * angle.cos(), 0.5 + 0.3 * angle.sin()), 1)

    X.add_(torch.randn(X.shape, device=device), alpha=0.02)
    X = 3 * X[torch.randperm(len(X))]

    y = torch.ones(len(X), dtype=torch.bool, device=X.device)
    y[0], y[-1] = False, False
    return preprocess(n_train, X, y)


def circles(n_train, n_val, device):
    X = torch.randn((n_train + n_val, 2), device=device) * 4
    # Torch optimizes `pow(b)` for integer b in (-32, 32)
    y = (X[:, 0] - 1).pow(2) + (X[:, 1] - 1).pow(2) < 0.5
    y = y | ((X[:, 0] + 1).pow(2) + (X[:, 1] + 1).pow(2) < 0.5)
    return preprocess(n_train, X, y)


def triangle(n_train, n_val, device):
    X = torch.rand(n_train + n_val, 2, device=device) * 4
    # Torch optimizes `pow(b)` for integer b in (-32, 32)
    y = 2 * X[:, 0] < X[:, 1]
    return preprocess(n_train, X, y)


def line(n_train, n_val, device):
    X = torch.randn((n_train + n_val, 2), device=device) * 4 + 1
    y = (X[:, 0] < 0) & (torch.abs(X[:, 0] - X[:, 1]) < 0.8)
    return preprocess(n_train, X, y)


def hollow(n_train, n_val, device, n_dim=2):
    X = torch.randn((n_train + n_val, n_dim), device=device) * 4
    dists = torch.norm(X, dim=1)
    y = (dists > 3.5 * math.sqrt(n_dim)) & (dists < 4.5 * math.sqrt(n_dim))
    return preprocess(n_train, X, y)


def save(train_X, val_X, val_y, name):
    torch.save((train_X, val_X, val_y), f"data/{name}.pt")


def load(name):
    train_X, val_X, val_y = torch.load(f"data/{name}.pt")
    return train_X, val_X, val_y


# def _get_colors(outputs, min_output, max_output, channel):
#     colors = np.zeros((len(outputs), 3))
#     colors[:, channel] = (1e-9 + outputs - min_output) / (1e-9 + max_output - min_output)

#     if min_output < 0 and max_output <= 0:
#         colors[:, channel] = 1 - colors[:, channel]

#     colors[:, channel] = 0.4 + 0.6 * colors[:, channel]
#     return colors


def scatter_outputs_y(X_pos, outputs_pos, X_neg, outputs_neg, model_name, centers=None):
    X_pos = X_pos.detach().cpu().numpy()
    X_neg = X_neg.detach().cpu().numpy()
    outputs_pos = outputs_pos.detach().cpu().numpy()
    outputs_neg = outputs_neg.detach().cpu().numpy()

    X_pos_pos = X_pos[outputs_pos >= 0]  # true positive
    X_pos_neg = X_pos[outputs_pos < 0]  # false negative
    X_neg_pos = X_neg[outputs_neg >= 0]  # false positive
    X_neg_neg = X_neg[outputs_neg < 0]  # true negative

    # pos_pos_outputs = outputs[pos & pos_output]  # true positive
    # neg_pos_outputs = outputs[neg & pos_output]  # false positive
    # pos_neg_outputs = outputs[pos & neg_output]  # false negative
    # neg_neg_outputs = outputs[neg & neg_output]  # true negative

    # min_pos_output = min(np.min(pos_pos_outputs, initial=0), np.min(neg_pos_outputs, initial=0))
    # max_pos_output = max(np.max(pos_pos_outputs, initial=0), np.max(neg_pos_outputs, initial=0))
    # min_neg_output = min(np.min(pos_neg_outputs, initial=0), np.min(neg_neg_outputs, initial=0))
    # max_neg_output = max(np.max(pos_neg_outputs, initial=0), np.max(neg_neg_outputs, initial=0))

    # red, green = 0, 1
    # pos_pos_colors = _get_colors(pos_pos_outputs, min_pos_output, max_pos_output, green)
    # neg_pos_colors = _get_colors(neg_pos_outputs, min_pos_output, max_pos_output, green)
    # pos_neg_colors = _get_colors(pos_neg_outputs, min_neg_output, max_neg_output, red)
    # neg_neg_colors = _get_colors(neg_neg_outputs, min_neg_output, max_neg_output, red)

    _, ax = plt.subplots()
    ax.set_title(model_name + " (marker=label, color=output)")
    ax.set_aspect("equal", adjustable="box")

    plt.scatter(X_pos[:, 0], X_pos[:, 1], marker=",", c="0.8")
    plt.scatter(X_neg[:, 0], X_neg[:, 1], marker=",", c="0.8")
    plt.scatter(X_pos_pos[:, 0], X_pos_pos[:, 1], marker="+", c="green")
    plt.scatter(X_pos_neg[:, 0], X_pos_neg[:, 1], marker="+", c="red")
    plt.scatter(X_neg_pos[:, 0], X_neg_pos[:, 1], marker="v", c="green")
    plt.scatter(X_neg_neg[:, 0], X_neg_neg[:, 1], marker="v", c="red")

    if centers is not None:
        centers = centers.detach().cpu().numpy()
        plt.scatter(centers[:, 0], centers[:, 1], marker=".", c="blue")
