from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch

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
        u, s, vh = torch.linalg.svd(center_densities)
        s = torch.clamp(s, min=1e-12)
        nystroem.normalization.copy_(torch.mm(u / s.sqrt(), vh).t())


# TODO: Leaky hinge loss or some other way to increase accuracy on basic triangle example?
def hinge_loss(outputs, margin):
    return torch.clamp(margin - outputs, min=0).mean()


def train_one_class(svm, train_inputs, val_inputs, batch_size=100, n_epochs=100):
    """
    Trains SVM to give a positive output for all training inputs, while minimizing the total set
    of inputs for which its output is positive.

    Gradient descent version of "Support Vector Method for Novelty Detection", Platt et al, 1999.
    Replicates sklearn OneClassSVM given same parameters.

    svm: SVM to train
    train_inputs: Positive training inputs; no labels since one class
    val_inputs: Positive validation inputs; no labels since one class
    no_false_negatives: Whether to postprocess bias so that the output is positive for all training
                        inputs, i.e. so that there are no false negatives.
    lr: Learning rate
    batch_size: Number of training inputs per batch
    n_epochs: Number of passes through training data
    nu: In range [0; 1]; lower nu --> higher importance of including positive examples
    """

    verbose = False
    no_false_negatives = True
    lr = 0.1
    nu = 0.01 if no_false_negatives else 0.5
    margin = 1
    assert lr * nu / 2 <= 1, (lr, nu)  # `lr * nu/2 > 1` breaks regularization.

    print("Training SVM")
    svm.train()
    if svm.nystroem is not None:
        train_nystroem(svm.nystroem, train_inputs)

    weight_decay = nu / 2
    optimizer = model.get_optimizer(
        torch.optim.SGD, svm, weight_decay=weight_decay, lr=lr
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=verbose)
    min_val_loss = float("inf")
    min_val_state = svm.state_dict()

    for epoch in range(n_epochs):
        train_inputs = train_inputs[torch.randperm(len(train_inputs))]

        loss = None
        for i_input in range(0, len(train_inputs), batch_size):
            batch_inputs = train_inputs[i_input : i_input + batch_size]
            batch_outputs = svm(batch_inputs)
            loss = hinge_loss(batch_outputs, margin) + nu * svm.bias
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            svm.eval()
            if verbose or epoch == n_epochs - 1:
                print(
                    f"Epoch {epoch}/{n_epochs} ({len(train_inputs)//1000}k samples per epoch)"
                )
                print(f"Last batch loss {loss.item()}")
                print(
                    "Batch accuracy", torch.sum(batch_outputs > 0) / len(batch_outputs)
                )

            val_outputs = svm(val_inputs)
            weight_decay_loss = (weight_decay * 0.5) * svm.coefs.dot(svm.coefs)
            val_loss = (
                hinge_loss(val_outputs, margin) + nu * svm.bias + weight_decay_loss
            )
            scheduler.step(val_loss)
            if verbose or epoch == n_epochs - 1:
                print("Validation loss", val_loss)
                print(
                    "Validation accuracy", torch.sum(val_outputs > 0) / len(val_outputs)
                )

            if val_loss <= min_val_loss:
                if min_val_loss < float("inf"):
                    # Don't overwrite saved model if loss was just < infinity.
                    min_val_state = deepcopy(svm.state_dict())
                min_val_loss = val_loss

            svm.train()
    svm.eval()
    svm.load_state_dict(min_val_state)

    with torch.no_grad():
        if no_false_negatives:
            # Adjust bias by choosing minimum output that avoids false negatives.
            min_output_should_be = 0.01
            svm.bias += min_output_should_be - svm(train_inputs).min()
        else:
            svm.bias -= margin  # Replicates sklearn OneClassSVM

    print("Trained SVM")


if __name__ == "__main__":
    device = "cpu"
    (inputs, labels), n_train = line_data(2000, device), 1000
    # Only training inputs with positive labels
    train_inputs = inputs[:n_train][labels[:n_train]]
    val_inputs, val_labels = inputs[n_train:], labels[n_train:]

    normalize = model.Normalize(train_inputs.min(0)[0], train_inputs.std(0))
    train_inputs, val_inputs = normalize(train_inputs), normalize(val_inputs)

    # nystroem = model.Nystroem(train_inputs[0], 2).to(device)
    # train_nystroem(nystroem, train_inputs)
    # train_inputs, val_inputs = nystroem(train_inputs), nystroem(val_inputs)

    torch_svm = model.SVM(train_inputs[0], 2, rbf=False).to(device)
    train_one_class(torch_svm, train_inputs, val_inputs[val_labels])
    model.save(torch_svm, "svm")
    print(*torch_svm.named_parameters())

    sk_svm = SGDOneClassSVM(nu=0.01)
    if sk_svm is not None:
        sk_svm.fit(train_inputs.detach().cpu().numpy())
        print("sk coefs", sk_svm.coef_)
        print("sk bias (== -offset)", -sk_svm.offset_)

    with torch.no_grad():
        torch_val_outputs = torch_svm(val_inputs).detach().cpu().numpy()
        val_inputs = val_inputs.detach().cpu().numpy()
        val_labels = val_labels.detach().cpu().numpy()
        # val_labels = np.full(len(val_inputs), True)
        print_results(val_labels, torch_val_outputs, "torch")
        plot_results(
            val_inputs,
            val_labels,
            torch_val_outputs,
            "torch",
            torch_svm.nystroem.centers.detach().cpu().numpy(),
        )

        if sk_svm is not None:
            sk_val_outputs = sk_svm.predict(val_inputs)
            print_results(val_labels, sk_val_outputs, "sk")
            plot_results(val_inputs, val_labels, sk_val_outputs, "sk")
        plt.show()
