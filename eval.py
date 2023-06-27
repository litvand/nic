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


def round_tensor(x):
    return round(x.item(), 5)


def percent(x):
    return f"{round(100 * x.item(), 2)}%"


def acc(is_correct):
    return div_zero(is_correct.count_nonzero(), len(is_correct))


def print_balanced_acc(outputs_on_pos, outputs_on_neg, name):
    acc_on_pos, acc_on_neg = acc(outputs_on_pos >= 0.), acc(outputs_on_neg < 0.)
    print(
        f"{name} balanced accuracy; true positives and negatives:",
        percent(0.5 * (acc_on_pos + acc_on_neg)),
        "     ",
        percent(acc_on_pos),
        percent(acc_on_neg)
    )


def bin_acc(outputs, targets):
    """
    Get binary classification accuracy, balanced accuracy, true positive and true negative rate

    outputs: Float tensor (n_outputs); positive output <--> positive class.
    targets: Boolean tensor (n_outputs); target True <--> positive class.
    """

    outputs = outputs >= 0
    assert targets.dtype == torch.bool, targets.dtype
    acc_total = acc(outputs == targets)
    acc_on_pos, acc_on_neg = acc(outputs[targets]), acc(~outputs[~targets])
    balanced = 0.5 * (acc_on_pos + acc_on_neg)
    return acc_total, balanced, acc_on_pos, acc_on_neg


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
