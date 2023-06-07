import matplotlib.pyplot as plt
import numpy as np
import torch


def circles(n_points, device):
    inputs = torch.randn((n_points, 2), device=device) * 1.5
    # Torch optimizes `pow(b)` for integer b in (-32, 32)
    labels = (inputs[:, 0] - 1).pow(2) + (inputs[:, 1] - 1).pow(2) < 0.5
    labels = labels | ((inputs[:, 0] + 1).pow(2) + (inputs[:, 1] + 1).pow(2) < 0.5)
    return inputs, labels


def triangle(n_points, device):
    inputs = torch.rand((n_points, 2), device=device)
    # Torch optimizes `pow(b)` for integer b in (-32, 32)
    labels = 2 * inputs[:, 0] < inputs[:, 1]
    return inputs, labels


def line(n_points, device):
    inputs = torch.empty((n_points, 2), device=device)
    inputs[:, 0] = torch.randn(n_points, device=device) + 1
    inputs[:, 1] = inputs[:, 0]
    labels = inputs[:, 0] < 0
    return inputs, labels


def save(train_inputs, val_inputs, val_labels, name):
    torch.save((train_inputs, val_inputs, val_labels), f"data/{name}.pt")


def load(name):
    train_inputs, val_inputs, val_labels = torch.load(f"data/{name}.pt")
    return train_inputs, val_inputs, val_labels


def scatter_labels_outputs(points, labels, outputs, model_name, centers=None):
    pos = labels
    neg = np.logical_not(labels)
    pos_output = outputs > 0
    neg_output = np.logical_not(pos_output)

    pos_pos_output = points[pos & pos_output]  # true positive
    neg_pos_output = points[neg & pos_output]  # false positive
    pos_neg_output = points[pos & neg_output]  # false negative
    neg_neg_output = points[neg & neg_output]  # true negative

    _, ax = plt.subplots()
    ax.set_title(model_name + " (marker=label, color=output)")
    ax.set_aspect('equal', adjustable='box')

    plt.scatter(points[:, 0], points[:, 1], marker=',', c='0.8')
    plt.scatter(pos_pos_output[:, 0], pos_pos_output[:, 1], marker="+", c="green")
    plt.scatter(neg_pos_output[:, 0], neg_pos_output[:, 1], marker="v", c="green")
    plt.scatter(pos_neg_output[:, 0], pos_neg_output[:, 1], marker="+", c="red")
    plt.scatter(neg_neg_output[:, 0], neg_neg_output[:, 1], marker="v", c="red")
    if centers is not None:
        plt.scatter(centers[:, 0], centers[:, 1], marker=".", c="blue")


def div_zero(a, b):
    # 0 if a == b == 0
    return a if a == 0 else a/b


def binary_accuracy(labels, outputs, model_name):
    outputs = outputs > 0
    print(
        f"{model_name} accuracy",
        np.count_nonzero(outputs == labels) / len(labels),
    )
    neg = np.logical_not(labels)
    print(
        f"{model_name} false positives (as fraction of negative examples)",
        div_zero(np.count_nonzero(outputs[neg]), np.count_nonzero(neg)),
    )
    print(
        f"{model_name} false negatives (as fraction of positive examples)",
        div_zero(np.count_nonzero(np.logical_not(outputs[labels])), np.count_nonzero(labels)),
    )

