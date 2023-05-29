from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import torch
from kmeans_pytorch import kmeans
# from sklearn.svm import OneClassSVM
from sklearn.linear_model import SGDOneClassSVM

import model


def train_nystroem(nystroem, train_inputs):
    assert nystroem.n_centers <= len(train_inputs), (
        len(train_inputs),
        nystroem.n_centers,
    )

    with torch.no_grad():
        if nystroem.kmeans:
            # TODO: kmeans++
            # TODO: sparse kmeans
            _, centers = kmeans(
                train_inputs, nystroem.n_centers, device=train_inputs.device
            )
        else:
            # Choose random centers without replacement
            center_indices = torch.randperm(len(train_inputs))[: nystroem.n_centers]
            centers = train_inputs[center_indices].flatten(1)

        nystroem.centers.copy_(centers)

        n_features = train_inputs[0].numel()
        var = train_inputs.var().item()
        nystroem.gamma[0] = 1 / (n_features * var) if var > 0 else 1 / n_features

        # TODO: Could we use a faster matrix decomposition instead of SVD, since `center_densities`
        #       is Hermitian?
        center_densities = model.gaussian_kernel(
            nystroem.centers, nystroem.centers, nystroem.gamma
        )
        u, s, vh = torch.linalg.svd(center_densities, driver="gesvd")
        s = torch.clamp(s, min=1e-12)
        nystroem.normalization.copy_(torch.mm(u / s.sqrt(), vh).t())


# TODO: Leaky hinge loss or some other way to increase accuracy on basic triangle example?
def hinge_loss(outputs, margin):
    return torch.clamp(margin - outputs, min=0).mean()


def train_one_class(
    svm,
    train_inputs,
    valid_inputs,
    no_false_negatives=False,
    lr=0.1,
    batch_size=100,
    n_epochs=100,
    nu=0.5,
    margin=1
):
    """
    Trains SVM to give a positive output for all training inputs, while minimizing the total set
    of inputs for which its output is positive.

    Gradient descent version of "Support Vector Method for Novelty Detection", Platt et al, 1999.
    Replicates sklearn OneClassSVM given same parameters.

    svm: SVM to train
    train_inputs: Positive training inputs; no labels since one class
    valid_inputs: Positive validation inputs; no labels since one class
    no_false_negatives: Whether to postprocess bias so that the output is positive for all training
                        inputs, i.e. so that there are no false negatives.
    lr: Learning rate
    batch_size: Number of training inputs per batch
    n_epochs: Number of passes through training data
    nu: In range [0; 1]; lower nu --> higher importance of including positive examples
    """

    assert lr * nu/2 <= 1, (lr, nu)  # `lr * nu/2 > 1` breaks regularization.
    verbose = False

    print("Training SVM")
    svm.train()
    if svm.nystroem is not None:
        train_nystroem(svm.nystroem, train_inputs)

    weight_decay = nu/2
    optimizer = torch.optim.SGD(svm.parameters(), lr=lr, weight_decay=0)
    min_valid_loss = float("inf")
    min_valid_state = svm.state_dict()

    for epoch in range(n_epochs):
        train_inputs = train_inputs[torch.randperm(len(train_inputs))]

        loss = None
        for i_input in range(0, len(train_inputs), batch_size):
            batch_inputs = train_inputs[i_input:i_input + batch_size]
            batch_outputs = svm(batch_inputs)
            weight_decay_loss = (weight_decay * 0.5) * svm.coefs.dot(svm.coefs)
            loss = hinge_loss(batch_outputs, margin) + nu * svm.bias + weight_decay_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            svm.eval()
            if verbose or epoch == n_epochs - 1:
                print(f"Epoch {epoch}/{n_epochs} ({len(train_inputs)//1000}k samples per epoch)")
                print(f"Last batch loss {loss.item()}")
                print("Batch accuracy", torch.sum(batch_outputs > 0) / len(batch_outputs))

            valid_outputs = svm(valid_inputs)
            weight_decay_loss = (weight_decay * 0.5) * svm.coefs.dot(svm.coefs)
            valid_loss = hinge_loss(valid_outputs, margin) + nu * svm.bias + weight_decay_loss
            if verbose or epoch == n_epochs - 1:
                print("Validation loss", valid_loss)
                print("Validation accuracy", torch.sum(valid_outputs > 0) / len(valid_outputs))

            if valid_loss <= min_valid_loss:
                if min_valid_loss < float("inf"):
                    # Don't overwrite saved model if loss was just < infinity.
                    min_valid_state = deepcopy(svm.state_dict())
                min_valid_loss = valid_loss

            svm.train()
    svm.eval()
    svm.load_state_dict(min_valid_state)

    with torch.no_grad():
        if no_false_negatives:
            # Adjust bias by choosing minimum output that avoids false negatives.
            min_output_should_be = 0.01
            svm.bias += min_output_should_be - svm(train_inputs).min()
        else:
            svm.bias -= margin  # Replicates sklearn OneClassSVM

    print("Trained SVM")


def circles_data(n_points, device):
    inputs = torch.randn((n_points, 2), device=device) * 1.5
    # Torch optimizes `pow(b)` for integer b in (-32, 32)
    labels = (inputs[:, 0] - 1).pow(2) + (inputs[:, 1] - 1).pow(2) < 0.5
    labels = labels | ((inputs[:, 0] + 1).pow(2) + (inputs[:, 1] + 1).pow(2) < 0.5)
    return inputs, labels


def triangle_data(n_points, device):
    inputs = torch.rand((n_points, 2), device=device)
    # Torch optimizes `pow(b)` for integer b in (-32, 32)
    labels = 2 * inputs[:, 0] < inputs[:, 1]
    return inputs, labels


def div_zero(a, b):
    # 0 if a == b == 0
    return a if a == 0 else a/b


def print_results(labels, outputs, model_name):
    print(
        f"{model_name} validation accuracy",
        np.count_nonzero(outputs == labels) / len(labels),
    )
    neg = np.logical_not(labels)
    print(
        f"{model_name} validation false positives (as fraction of negative examples)",
        div_zero(np.count_nonzero(outputs[neg]), np.count_nonzero(neg)),
    )
    print(
        f"{model_name} validation false negatives (as fraction of positive examples)",
        div_zero(np.count_nonzero(np.logical_not(outputs[labels])), np.count_nonzero(labels)),
    )


def plot_results(inputs, labels, outputs, model_name):
    pos = labels
    neg = np.logical_not(labels)
    pos_output = outputs > 0
    neg_output = np.logical_not(pos_output)

    pos_pos_output = inputs[pos & pos_output]
    neg_pos_output = inputs[neg & pos_output]
    pos_neg_output = inputs[pos & neg_output]
    neg_neg_output = inputs[neg & neg_output]

    _, ax = plt.subplots()
    ax.set_title(model_name + " (marker=label, color=output)")

    # plt.scatter(inputs[:, 0], inputs[:, 1], marker=',', c='black')
    plt.scatter(pos_pos_output[:, 0], pos_pos_output[:, 1], marker="+", c="green")
    plt.scatter(neg_pos_output[:, 0], neg_pos_output[:, 1], marker="v", c="green")
    plt.scatter(pos_neg_output[:, 0], pos_neg_output[:, 1], marker="+", c="red")
    plt.scatter(neg_neg_output[:, 0], neg_neg_output[:, 1], marker="v", c="red")


if __name__ == "__main__":
    device = "cuda"
    (inputs, labels), n_train = triangle_data(6000, device), 5000
    train_inputs, train_labels = inputs[:n_train], labels[:n_train]
    valid_inputs, valid_labels = inputs[n_train:], labels[n_train:]
    # No negative training inputs
    train_inputs, train_labels = train_inputs[train_labels], None
    inputs -= train_inputs.mean(0, keepdim=True)
    inputs /= train_inputs.std(0, keepdim=True)

    # nystroem = model.Nystroem(train_inputs[0], 2).to(device)
    # train_nystroem(nystroem, train_inputs)
    # train_inputs, valid_inputs = nystroem(train_inputs), nystroem(valid_inputs)

    torch_svm = model.SVM(train_inputs[0], 2, rbf=False).to(device)
    train_one_class(torch_svm, train_inputs, valid_inputs[valid_labels])
    model.save(torch_svm, "svm")
    print(*torch_svm.named_parameters())

    sk_svm = SGDOneClassSVM()
    if sk_svm is not None:
        sk_svm.fit(train_inputs.detach().cpu().numpy())
        print("sk coefs", sk_svm.coef_)
        print("sk bias (== -offset)", -sk_svm.offset_)

    with torch.no_grad():
        torch_valid_outputs = (torch_svm(valid_inputs) > 0).detach().cpu().numpy()
        valid_inputs = valid_inputs.detach().cpu().numpy()
        valid_labels = valid_labels.detach().cpu().numpy()
        # valid_labels = np.full(len(valid_inputs), True)
        print_results(valid_labels, torch_valid_outputs, "torch")
        plot_results(valid_inputs, valid_labels, torch_valid_outputs, "torch")

        if sk_svm is not None:
            sk_valid_outputs = sk_svm.predict(valid_inputs) > 0
            print_results(valid_labels, sk_valid_outputs, "sk")
            plot_results(valid_inputs, valid_labels, sk_valid_outputs, "sk")
        plt.show()
