import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import SGDOneClassSVM

import model


def hinge_loss(outputs, margin):
    return torch.clamp(margin - outputs, min=0).mean()


def train_one_class(svm, train_inputs, valid_inputs):
    """
    Trains SVM to output 1 for all training inputs, while minimizing regularization loss.

    Gradient descent version of "Support Vector Method for Novelty Detection", Platt et al, 1999

    svm: SVM to train
    train_inputs: Training inputs; no labels since one-class
    valid_inputs: Validation inputs; no labels since one-class
    """

    # Higher margin/lower nu --> greater importance of including positive examples
    nu = 0.5
    margin = 1
    batch_size = 150
    n_epochs = 1000
    lr = 1e-1

    svm.train()
    optimizer = torch.optim.SGD(svm.parameters(), lr=lr)
    alpha = nu / 2
    min_valid_loss = float("inf")

    for i_input in range(0, n_epochs * len(train_inputs), batch_size):
        indices = torch.randint(high=len(train_inputs), size=(batch_size,))
        batch_outputs = svm(train_inputs[indices])
        loss = hinge_loss(batch_outputs, margin)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            # Based on sklearn implementation
            svm.bias -= lr * 2 * alpha
            svm.coefs *= max(0.0, 1 - lr * alpha)

        # `< batch_size` instead of `== 0`, because might not be exactly 0
        if i_input % (len(train_inputs) * 100) < batch_size:
            with torch.no_grad():
                print(f"{i_input//1000}k inputs processed, batch loss {loss.item()}")

                svm.eval()
                valid_outputs = svm(valid_inputs)
                print(
                    "Validation accuracy",
                    torch.sum(valid_outputs > 0.0) / len(valid_outputs),
                )

                valid_loss = hinge_loss(valid_outputs, margin)
                print("Validation loss", valid_loss)
                if valid_loss <= min_valid_loss:
                    if min_valid_loss < float("inf"):
                        # Don't overwrite saved model if loss was just < infinity.
                        model.save(svm, "svm")
                    min_valid_loss = valid_loss

                svm.train()

    # Adjust bias by choosing minimum output on positive training examples.
    with torch.no_grad():
        # min_output = svm(train_inputs).min()
        # should_be_min = 0.01
        # svm.bias += should_be_min - min_output
        svm.bias -= 1  # Replicates sklearn

    svm.eval()


def circles_data(n_points, device):
    inputs = torch.randn((n_points, 2), device=device) * 1.5
    # Torch optimizes `pow(b)` for integer b in (-32, 32)
    labels = (inputs[:, 0] - 1.0).pow(2) + (inputs[:, 1] - 1.0).pow(2) < 0.5
    labels = labels | ((inputs[:, 0] + 1.0).pow(2) + (inputs[:, 1] + 1.0).pow(2) < 0.5)
    return inputs, labels


def triangle_data(n_points, device):
    inputs = torch.rand((n_points, 2), device=device)
    # Torch optimizes `pow(b)` for integer b in (-32, 32)
    labels = 2 * inputs[:, 0] < inputs[:, 1]
    return inputs, labels


def plot_results(inputs, labels, outputs, title):
    pos = labels
    neg = np.logical_not(labels)
    pos_output = outputs > 0.0
    neg_output = np.logical_not(pos_output)

    pos_pos_output = inputs[pos & pos_output]
    neg_pos_output = inputs[neg & pos_output]
    pos_neg_output = inputs[pos & neg_output]
    neg_neg_output = inputs[neg & neg_output]

    _, ax = plt.subplots()
    ax.set_title(title + " (marker=label, color=output)")

    # plt.scatter(inputs[:, 0], inputs[:, 1], marker=',', c='black')
    plt.scatter(pos_pos_output[:, 0], pos_pos_output[:, 1], marker="+", c="green")
    plt.scatter(neg_pos_output[:, 0], neg_pos_output[:, 1], marker="v", c="green")
    plt.scatter(pos_neg_output[:, 0], pos_neg_output[:, 1], marker="+", c="red")
    plt.scatter(neg_neg_output[:, 0], neg_neg_output[:, 1], marker="v", c="red")


if __name__ == "__main__":
    device = "cuda"

    (inputs, labels), n_train = triangle_data(1000, device), 500
    inputs -= inputs.mean(dim=0, keepdim=True)

    train_inputs, train_labels = inputs[:n_train], labels[:n_train]
    valid_inputs, valid_labels = inputs[n_train:], labels[n_train:]
    pos_train_inputs, pos_valid_inputs = (
        train_inputs[train_labels],
        valid_inputs[valid_labels],
    )

    torch_svm = model.SVM(train_inputs[0], 2).to(device)
    train_one_class(torch_svm, pos_train_inputs, pos_valid_inputs)
    # model.load(torch_svm, 'svm-84c77a1383308746702e10b3b5c57b9cd4587fc7.pt')
    print(*torch_svm.named_parameters())

    sk_svm = SGDOneClassSVM()
    sk_svm.fit(pos_train_inputs.detach().cpu().numpy())
    print("sk coefs", sk_svm.coef_)
    print("sk bias (== -offset)", -sk_svm.offset_)

    with torch.no_grad():
        torch_valid_outputs = (torch_svm(valid_inputs) > 0.0).detach().cpu().numpy()
        valid_inputs = valid_inputs.detach().cpu().numpy()
        valid_labels = valid_labels.detach().cpu().numpy()
        sk_valid_outputs = sk_svm.predict(valid_inputs) > 0
        plot_results(valid_inputs, valid_labels, torch_valid_outputs, "torch")
        plot_results(valid_inputs, valid_labels, sk_valid_outputs, "sk")
        plt.show()
