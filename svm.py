import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.svm import OneClassSVM

import model


def hinge_loss(outputs):
    return torch.clamp(0.1 - outputs, min=0.0).mean()


def train_one_class(svm, train_inputs, valid_inputs):
    '''
    Trains SVM to output 1 for all training inputs, while minimizing regularization loss.

    Gradient descent version of "Support Vector Method for Novelty Detection", Platt et al, 1999

    svm: SVM to train
    train_inputs: Training inputs; no labels since one-class
    valid_inputs: Validation inputs; no labels since one-class
    '''

    svm.train()
    min_valid_loss = float("inf")

    # todo Relative importance of regularization, i.e. importance of excluding inputs.
    # Higher exclusivity makes the area inside the learned distribution smaller.


    alpha = 1.0
    batch_size = 150
    n_epochs = 5000
    optimizer = torch.optim.Adam(svm.parameters(), lr=1e-3)

    for i_input in range(0, n_epochs * len(train_inputs), batch_size):
        indices = torch.randint(high=len(train_inputs), size=(batch_size,))
        batch_outputs = svm(train_inputs[indices])
        loss = hinge_loss(batch_outputs)
        optimizer.zero_grad()
        loss.backward()

        with torch.no_grad(): 

        optimizer.step()

        with torch.no_grad():
            # Only relative magnitudes of coefficients matter, not absolute magnitudes
            svm.coefs.abs_()
            sum = svm.coefs.sum()
            if sum.item() != 0.0:
                svm.coefs /= sum

        # `< batch_size` instead of `== 0`, because might not be exactly 0
        if i_input % 20000 < batch_size:
            with torch.no_grad():
                print(f"{i_input//1000}k inputs processed, batch loss {loss.item()}")

                svm.eval()
                valid_outputs = svm(valid_inputs)
                print("Validation accuracy", torch.sum(valid_outputs > 0.0) / len(valid_outputs))

                valid_loss = hinge_loss(valid_outputs) + exclusivity * svm.regularization_loss()
                print("Validation loss", valid_loss)
                if valid_loss < min_valid_loss:
                    if min_valid_loss < float("inf"):
                        # Don't overwrite saved model if loss was just < infinity.
                        model.save(svm, 'svm')
                    min_valid_loss = valid_loss

                svm.train()
    svm.eval()


def gen_data(n_train, n_valid, device):
    inputs = torch.randn(n_train + n_valid, 2, device=device) * 1.5
    # Torch optimizes `pow(b)` for integer b in (-32, 32)
    labels = (inputs[:, 0] - 1.0).pow(2) + (inputs[:, 1] - 1.0).pow(2) < 0.5
    labels = labels | ((inputs[:, 0] + 1.0).pow(2) + (inputs[:, 1] + 1.0).pow(2) < 0.5)
    return (inputs[:n_train], labels[:n_train]), (inputs[n_train:], labels[n_train:])


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
    ax.set_title(title + ' (marker=label, color=output)')

    plt.scatter(pos_pos_output[:, 0], pos_pos_output[:, 1], marker='+', c='green')
    plt.scatter(neg_pos_output[:, 0], neg_pos_output[:, 1], marker='v', c='green')
    plt.scatter(pos_neg_output[:, 0], pos_neg_output[:, 1], marker='+', c='red')
    plt.scatter(neg_neg_output[:, 0], neg_neg_output[:, 1], marker='v', c='red')


if __name__ == '__main__':
    device = 'cuda'
    (train_inputs, train_labels), (valid_inputs, valid_labels) = gen_data(500, 500, device)
    pos_train_inputs, pos_valid_inputs = train_inputs[train_labels], valid_inputs[valid_labels]
    torch_svm = model.DensitySVM(train_inputs[0], 20).to(device)
    # train_one_class(torch_svm, pos_train_inputs, pos_valid_inputs)
    model.load(torch_svm, 'svm-e42ed1343cde0d884e6b31129ca3b5f4af7203bc.pt')
    print(*torch_svm.named_parameters())
    print('coef sum', torch_svm.coefs.sum())

    sk_svm = OneClassSVM(gamma='auto', kernel='linear', nu=0.5)
    sk_svm.fit(pos_train_inputs.detach().cpu().numpy())

    with torch.no_grad():
        torch_valid_outputs = (torch_svm(valid_inputs) > 0.0).detach().cpu().numpy()
        valid_inputs = valid_inputs.detach().cpu().numpy()
        valid_labels = valid_labels.detach().cpu().numpy()
        sk_valid_outputs = sk_svm.predict(valid_inputs) > 0
        plot_results(valid_inputs, valid_labels, torch_valid_outputs, 'torch')
        plot_results(valid_inputs, valid_labels, sk_valid_outputs, 'sk')
        plt.show()
