import matplotlib.pyplot as plt
import numpy as np
import torch


def preprocess(n_train, points, labels):
    """NOTE: Number of returned training points is less than `n_train`, because negative training
    points are discarded."""

    train_points = points[:n_train][labels[:n_train]]
    val_points, val_labels = points[n_train:], labels[n_train:]

    mean, inv_std = train_points.mean(), 1.0/train_points.std()
    train_points.sub_(mean).mul_(inv_std)
    val_points.sub_(mean).mul_(inv_std)

    return train_points, val_points, val_labels


def circles(n_train, n_val, device):
    points = torch.randn((n_train + n_val, 2), device=device) * 4
    # Torch optimizes `pow(b)` for integer b in (-32, 32)
    labels = (points[:, 0] - 1).pow(2) + (points[:, 1] - 1).pow(2) < 0.5
    labels = labels | ((points[:, 0] + 1).pow(2) + (points[:, 1] + 1).pow(2) < 0.5)
    return preprocess(n_train, points, labels)


def triangle(n_train, n_val, device):
    points = torch.rand(n_train + n_val, 2, device=device) * 4
    # Torch optimizes `pow(b)` for integer b in (-32, 32)
    labels = 2 * points[:, 0] < points[:, 1]
    return preprocess(n_train, points, labels)


def line(n_train, n_val, device):
    points = torch.randn((n_train + n_val, 2), device=device) * 4 + 1
    labels = (points[:, 0] < 0) & (torch.abs(points[:, 0] - points[:, 1]) < 0.8)
    return preprocess(n_train, points, labels)


def hollow(n_train, n_val, device):
    points = torch.randn((n_train + n_val, 2), device=device) * 4
    dists = torch.norm(points, dim=1)
    labels = (dists > 3) & (dists < 5)
    return preprocess(n_train, points, labels)


def save(train_inputs, val_inputs, val_labels, name):
    torch.save((train_inputs, val_inputs, val_labels), f"data/{name}.pt")


def load(name):
    train_inputs, val_inputs, val_labels = torch.load(f"data/{name}.pt")
    return train_inputs, val_inputs, val_labels


def scatter_labels_outputs(points, labels, outputs, model_name, centers=None):
    pos = labels.detach().cpu().numpy()
    neg = np.logical_not(pos)
    pos_output = outputs.detach().cpu().numpy() > 0
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


def percent(x):
    return f"{round(100 * x.item(), 2)}%"


def print_accuracy(labels, outputs, model_name):
    outputs = outputs > 0
    print(
        f"{model_name} accuracy:",
        percent(div_zero(torch.count_nonzero(outputs == labels), len(labels)))
    )

    on_pos, on_neg = outputs[labels], outputs[~labels]
    acc_on_pos = div_zero(on_pos.count_nonzero(), len(on_pos))
    acc_on_neg = div_zero(len(on_neg) - on_neg.count_nonzero(), len(on_neg))
    print(
        f"{model_name} balanced accuracy:",
        percent((acc_on_pos + acc_on_neg)/2)
    )
    print(
        f"{model_name} true positives (as fraction of positive examples):",
        percent(acc_on_pos)
    )
    print(
        f"{model_name} true negatives (as fraction of negative examples):",
        percent(acc_on_neg)
    )
