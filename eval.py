from tkinter import N
import matplotlib.pyplot as plt
import torch


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


def percent(x):
    return f"{round(100 * x.item(), 2)}%"


def bin_acc(outputs, targets):
    """
    Get binary classification accuracy, balanced accuracy, true positive and true negative rate

    outputs: Float tensor (n_outputs); positive output <--> positive class.
    targets: Boolean tensor (n_outputs); target True <--> positive class.
    """

    outputs = outputs > 0
    acc = div_zero(torch.count_nonzero(outputs == targets), len(targets))

    on_pos, on_neg = outputs[targets], outputs[~targets]
    acc_on_pos = div_zero(on_pos.count_nonzero(), len(on_pos))
    acc_on_neg = div_zero(len(on_neg) - on_neg.count_nonzero(), len(on_neg))

    print(
        "Binary counts",
        torch.count_nonzero(outputs == targets),
        on_pos.count_nonzero(),
        len(on_neg) - on_neg.count_nonzero(),
    )
    print("len outputs, targets", len(outputs), len(targets), targets)
    print("outputs[0]", outputs[0], outputs[0] == 0, outputs[0] == 1, outputs[:1].count_nonzero())

    balanced = (acc_on_pos + acc_on_neg) / 2
    return acc, balanced, acc_on_pos, acc_on_neg


"""
acc = (correct_on_pos + correct_on_neg) / (n_on_pos + n_on_neg)
balanced_acc = (correct_on_pos/n_on_pos + correct_on_neg/n_on_neg)/2
if n_on_pos == n_on_neg:
    acc = (correct_on_pos + correct_on_neg) / (2 * n_on_pos)
    balanced_acc = ((correct_on_pos + correct_on_neg) / n_on_pos)/2
"""


def print_bin_acc(outputs, targets, model_name):
    """
    Print binary classification accuracy, balanced accuracy, true positive and true negative rate

    outputs: Float tensor (n_outputs); positive output <--> positive class.
    targets: Boolean tensor (n_outputs); target True <--> positive class.
    model_name: String
    """

    acc, balanced, acc_on_pos, acc_on_neg = bin_acc(outputs, targets)
    print(f"{model_name} accuracy:", percent(acc))
    print(f"{model_name} balanced accuracy:", percent(balanced))
    print(f"{model_name} true positives (as fraction of positive targets):", percent(acc_on_pos))
    print(f"{model_name} true negatives (as fraction of negative targets):", percent(acc_on_neg))


def print_multi_acc(outputs, targets, model_name):
    """
    Multi-class classification accuracy

    outputs: Float tensor (n_outputs, n_classes) containing logits
    targets: Int tensor (n_outputs) containing correct classes
    model_name: String
    """

    output_classes = outputs.argmax(-1)
    print(
        f"{model_name} accuracy:",
        percent(div_zero(torch.count_nonzero(output_classes == targets), len(targets))),
    )
