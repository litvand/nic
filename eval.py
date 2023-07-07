import gc

import matplotlib.pyplot as plt
import torch


def batched_forward(f, X, batch_size):
    return torch.cat(
        tuple(f(X[i_x : i_x + batch_size]).cpu() for i_x in range(0, len(X), batch_size))
    )


def batched_activations(net, X, batch_size):
    # a[i_batch][i_layer][i_x]
    a = []
    for i_x in range(0, len(X), batch_size):
        batch = [layer for layer in net.activations(X[i_x : i_x + batch_size])]
        a.append(batch)

    # a[i_layer][i_x]
    a = [
        torch.cat(tuple(a[i_batch][i_layer] for i_batch in range(len(a))))
        for i_layer in range(len(a[0]))
    ]
    return a


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


def plot_distr_overlap(a, b, a_name="", b_name=""):
    """
    Plot cumulative distribution functions of two sets of numbers

    `a` and `b` are 1d tensors, with possibly different lengths
    """

    a, _ = a.sort()
    b, _ = b.sort()

    a_reverse_cumulative = torch.arange(len(a), 0, -1, device="cpu") / float(len(a))
    b_cumulative = torch.arange(len(b), device="cpu") / float(len(b))

    _, ax = plt.subplots()
    ax.set_title(f"{a_name} (reverse CDF) vs {b_name} (CDF)")
    plt.scatter(a.cpu(), a_reverse_cumulative)
    plt.scatter(b.cpu(), b_cumulative)


def div_zero(a, b):
    """0 if a == b == 0"""
    return a if a == 0 else a / b


def round_tensor(x):
    return round(x.item(), 5)


def percent(x):
    return f"{round(100 * x.item(), 2)}%"


def acc(is_correct):
    return div_zero(is_correct.count_nonzero(), len(is_correct))


def print_balanced_acc(outputs_on_pos, outputs_on_neg, name):
    acc_on_pos, acc_on_neg = acc(outputs_on_pos >= 0.0), acc(outputs_on_neg < 0.0)
    print(
        f"{name} balanced accuracy; true positives and negatives:",
        percent(0.5 * (acc_on_pos + acc_on_neg)),
        "     ",
        percent(acc_on_pos),
        percent(acc_on_neg),
    )


def bin_acc(outputs, labels):
    """
    Get binary classification accuracy, balanced accuracy, true positive and true negative rate

    outputs: Float tensor (n_outputs); positive output <--> positive class.
    labels: Boolean tensor (n_outputs); label True <--> positive class.
    """

    outputs = outputs >= 0
    assert labels.dtype == torch.bool, labels.dtype
    acc_total = acc(outputs == labels)
    acc_on_pos, acc_on_neg = acc(outputs[labels]), acc(~outputs[~labels])
    balanced = 0.5 * (acc_on_pos + acc_on_neg)
    return acc_total, balanced, acc_on_pos, acc_on_neg


def print_bin_acc(outputs, labels, model_name):
    """
    Print binary classification accuracy, balanced accuracy, true positive and true negative rate

    outputs: Float tensor (n_outputs); positive output <--> positive class.
    labels: Boolean tensor (n_outputs); label True <--> positive class.
    model_name: String
    """

    acc, balanced, acc_on_pos, acc_on_neg = bin_acc(outputs, labels)
    print(f"{model_name} accuracy:", percent(acc))
    print(f"{model_name} balanced accuracy:", percent(balanced))
    print(f"{model_name} true positives (as fraction of positive labels):", percent(acc_on_pos))
    print(f"{model_name} true negatives (as fraction of negative labels):", percent(acc_on_neg))


def print_multi_acc(outputs, labels, model_name):
    """
    Multi-class classification accuracy

    outputs: Float tensor (n_outputs, n_classes) containing logits
    labels: Int tensor (n_outputs) containing correct classes
    model_name: String
    """

    print(f"{model_name} accuracy:", percent(acc(outputs.argmax(-1) == labels)))
