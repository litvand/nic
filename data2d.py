# 2d data for one-class classification

import math

import matplotlib.pyplot as plt
import numpy as np
import torch

from train import Normalize


def preprocess(n_train, X, y):
    X_train, X_val, y_train, y_val = X[:n_train], X[n_train:], y[:n_train], y[n_train:]
    
    X_train_pos, X_train_neg = X_train[y_train], X_train[~y_train]
    X_val_pos, X_val_neg = X_val[y_val], X_val[~y_val]
    
    max_len = len(X_train_pos)
    norm = Normalize(X_train_pos[0]).fit(X_train_pos, unit_range=True)
    return tuple(
        None if len(X) == 0 else norm(X[torch.randperm(len(X))[:max_len]]) for
        X in (X_train_pos, X_train_neg, X_val_pos, X_val_neg)
    )


def spiral(n_train, n_val, device):
    angle = torch.linspace(0, 2 * np.pi, n_train + n_val + 1, device=device)[:-1]
    X = torch.stack((0.5 + 0.4 * (angle / 7) * angle.cos(), 0.5 + 0.3 * angle.sin()), 1)
    X.add_(torch.randn(X.shape, device=device), alpha=0.02)
    X = 3 * X[torch.randperm(len(X))]
    y = torch.ones(len(X), dtype=torch.bool, device=X.device)
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
    y = (dists > 3 * math.sqrt(n_dim)) & (dists < 5 * math.sqrt(n_dim))
    return preprocess(n_train, X, y)


def save(X_train, X_val, y_val, name):
    torch.save((X_train, X_val, y_val), f"data/{name}.pt")


def load(name):
    X_train, X_val, y_val = torch.load(f"data/{name}.pt")
    return X_train, X_val, y_val


def _get_colors(outputs, min_output, max_output, channel):
    colors = np.zeros((len(outputs), 3))
    colors[:, channel] = (1e-9 + outputs - min_output) / (1e-9 + max_output - min_output)

    if min_output < 0 and max_output <= 0:
        colors[:, channel] = 1 - colors[:, channel]

    colors[:, channel] = 0.4 + 0.6 * colors[:, channel]
    return colors


def scatter_outputs_y(X, outputs, y, model_name, centers=None):
    X = X.detach().cpu().numpy()
    outputs = outputs.detach().cpu().numpy()

    pos = y.detach().cpu().numpy()
    neg = np.logical_not(pos)
    pos_output = outputs > 0
    neg_output = np.logical_not(pos_output)

    pos_pos_output = X[pos & pos_output]  # true positive
    neg_pos_output = X[neg & pos_output]  # false positive
    pos_neg_output = X[pos & neg_output]  # false negative
    neg_neg_output = X[neg & neg_output]  # true negative

    pos_pos_outputs = outputs[pos & pos_output]  # true positive
    neg_pos_outputs = outputs[neg & pos_output]  # false positive
    pos_neg_outputs = outputs[pos & neg_output]  # false negative
    neg_neg_outputs = outputs[neg & neg_output]  # true negative

    min_pos_output = min(np.min(pos_pos_outputs, initial=0), np.min(neg_pos_outputs, initial=0))
    max_pos_output = max(np.max(pos_pos_outputs, initial=0), np.max(neg_pos_outputs, initial=0))
    min_neg_output = min(np.min(pos_neg_outputs, initial=0), np.min(neg_neg_outputs, initial=0))
    max_neg_output = max(np.max(pos_neg_outputs, initial=0), np.max(neg_neg_outputs, initial=0))

    red, green = 0, 1
    pos_pos_colors = _get_colors(pos_pos_outputs, min_pos_output, max_pos_output, green)
    neg_pos_colors = _get_colors(neg_pos_outputs, min_pos_output, max_pos_output, green)
    pos_neg_colors = _get_colors(pos_neg_outputs, min_neg_output, max_neg_output, red)
    neg_neg_colors = _get_colors(neg_neg_outputs, min_neg_output, max_neg_output, red)

    _, ax = plt.subplots()
    ax.set_title(model_name + " (marker=target, color=output)")
    ax.set_aspect("equal", adjustable="box")

    plt.scatter(X[:, 0], X[:, 1], marker=",", c="0.8")
    plt.scatter(pos_pos_output[:, 0], pos_pos_output[:, 1], marker="+", c=pos_pos_colors)
    plt.scatter(neg_pos_output[:, 0], neg_pos_output[:, 1], marker="v", c=neg_pos_colors)
    plt.scatter(pos_neg_output[:, 0], pos_neg_output[:, 1], marker="+", c=pos_neg_colors)
    plt.scatter(neg_neg_output[:, 0], neg_neg_output[:, 1], marker="v", c=neg_neg_colors)
    if centers is not None:
        plt.scatter(centers[:, 0], centers[:, 1], marker=".", c="blue")
