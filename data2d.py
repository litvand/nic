# 2d data for one-class classification


import matplotlib.pyplot as plt
import numpy as np
import torch

import train


def preprocess(n_train, points, targets):
    """NOTE: Number of returned training points is less than `n_train`, because negative training
    points are discarded."""

    train_points = points[:n_train][targets[:n_train]]
    val_points, val_targets = points[n_train:], targets[n_train:]
    normalize = train.Normalize(train_points[0]).fit(train_points)
    return normalize(train_points), normalize(val_points), val_targets


def circles(n_train, n_val, device):
    points = torch.randn((n_train + n_val, 2), device=device) * 4
    # Torch optimizes `pow(b)` for integer b in (-32, 32)
    targets = (points[:, 0] - 1).pow(2) + (points[:, 1] - 1).pow(2) < 0.5
    targets = targets | ((points[:, 0] + 1).pow(2) + (points[:, 1] + 1).pow(2) < 0.5)
    return preprocess(n_train, points, targets)


def triangle(n_train, n_val, device):
    points = torch.rand(n_train + n_val, 2, device=device) * 4
    # Torch optimizes `pow(b)` for integer b in (-32, 32)
    targets = 2 * points[:, 0] < points[:, 1]
    return preprocess(n_train, points, targets)


def line(n_train, n_val, device):
    points = torch.randn((n_train + n_val, 2), device=device) * 4 + 1
    targets = (points[:, 0] < 0) & (torch.abs(points[:, 0] - points[:, 1]) < 0.8)
    return preprocess(n_train, points, targets)


def hollow(n_train, n_val, device):
    points = torch.randn((n_train + n_val, 2), device=device) * 4
    dists = torch.norm(points, dim=1)
    targets = (dists > 3) & (dists < 5)
    return preprocess(n_train, points, targets)


def save(train_inputs, val_inputs, val_targets, name):
    torch.save((train_inputs, val_inputs, val_targets), f"data/{name}.pt")


def load(name):
    train_inputs, val_inputs, val_targets = torch.load(f"data/{name}.pt")
    return train_inputs, val_inputs, val_targets


def _get_colors(outputs, min_output, max_output, channel):
    colors = np.zeros((len(outputs), 3))
    colors[:, channel] = (1e-9 + outputs - min_output) / (1e-9 + max_output - min_output)

    if min_output < 0 and max_output <= 0:
        colors[:, channel] = 1 - colors[:, channel]

    colors[:, channel] = 0.4 + 0.6 * colors[:, channel]
    return colors


def scatter_outputs_targets(points, outputs, targets, model_name, centers=None):
    points = points.detach().cpu().numpy()
    outputs = outputs.detach().cpu().numpy()

    pos = targets.detach().cpu().numpy()
    neg = np.logical_not(pos)
    pos_output = outputs > 0
    neg_output = np.logical_not(pos_output)

    pos_pos_output = points[pos & pos_output]  # true positive
    neg_pos_output = points[neg & pos_output]  # false positive
    pos_neg_output = points[pos & neg_output]  # false negative
    neg_neg_output = points[neg & neg_output]  # true negative

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

    plt.scatter(points[:, 0], points[:, 1], marker=",", c="0.8")
    plt.scatter(pos_pos_output[:, 0], pos_pos_output[:, 1], marker="+", c=pos_pos_colors)
    plt.scatter(neg_pos_output[:, 0], neg_pos_output[:, 1], marker="v", c=neg_pos_colors)
    plt.scatter(pos_neg_output[:, 0], pos_neg_output[:, 1], marker="+", c=pos_neg_colors)
    plt.scatter(neg_neg_output[:, 0], neg_neg_output[:, 1], marker="v", c=neg_neg_colors)
    if centers is not None:
        plt.scatter(centers[:, 0], centers[:, 1], marker=".", c="blue")
