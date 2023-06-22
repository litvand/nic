from copy import deepcopy

import torch
# from sklearn.svm import OneClassSVM
from sklearn.linear_model import SGDOneClassSVM
from torch import nn

import data2d
from eval import acc, percent
import train


def hinge_loss(outputs, margin):
    return torch.clamp(margin - outputs, min=0).mean()


class SVM(nn.Module):
    def __init__(self, x_example):
        super().__init__()
        self.linear = nn.Linear(len(x_example), 1)

    def forward(self, X):
        return self.linear(X)
    
    def fit(
        self, X_train_pos, verbose=False, no_false_negatives=True, batch_size=100, n_epochs=100
    ):
        """
        Trains SVM to give a positive output for all training inputs, while minimizing the total set
        of inputs for which its output is positive.

        Gradient descent version of "Support Vector Method for Novelty Detection", Platt et al,
        1999. Replicates sklearn OneClassSVM given same parameters.

        X_train_pos: Positive training inputs; no targets since one class
        no_false_negatives: Whether to postprocess bias so that the output is positive for all
                            training inputs, i.e. so that there are no false negatives.
        batch_size: Number of training inputs per batch
        n_epochs: Number of passes through training data
        """

        # In range [0; 1]; lower nu --> higher importance of including positive examples
        nu = 0.01 if no_false_negatives else 0.5
        lr = 0.01
        assert lr * nu / 2 <= 1, (lr, nu)  # `lr * nu/2 > 1` breaks regularization.
        margin = 1

        print("Training SVM")
        optimizer = train.get_optimizer(torch.optim.NAdam, self.linear, weight_decay=nu/2, lr=lr)

        for epoch in range(n_epochs):
            X_train_pos = X_train_pos[torch.randperm(len(X_train_pos))]

            loss = None
            for i_first in range(0, len(X_train_pos), batch_size):
                X = X_train_pos[i_first : i_first + batch_size]
                outputs = self.linear(X)
                loss = hinge_loss(outputs, margin) + nu * self.linear.bias
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                if verbose or epoch == n_epochs - 1:
                    print(f"Epoch {epoch}/{n_epochs} ({len(X_train_pos)//1000}k samples per epoch)")
                    print(f"Last batch loss {loss.item()}")

        with torch.no_grad():
            if no_false_negatives:
                # Adjust bias by choosing minimum output that avoids false negatives.
                min_output_should_be = 0.01
                self.linear.bias += min_output_should_be - self.linear(X_train_pos).min()
            else:
                self.linear.bias -= margin  # Replicates sklearn OneClassSVM

        print("Trained SVM")


def show_results(X_pos, outputs_pos, X_neg, outputs_neg, *args):
    data2d.scatter_outputs_y(X_pos, outputs_pos, X_neg, outputs_neg, *args)
    
    acc_on_pos, acc_on_neg = acc(outputs_pos >= 0), acc(outputs_neg < 0)
    print("Balanced accuracy:", percent(0.5 * (acc_on_pos + acc_on_neg)))
    print("True positive, true negative rate:", percent(acc_on_pos), percent(acc_on_neg))


if __name__ == "__main__":
    device = "cpu"
    X_train_pos, _, X_val_pos, X_val_neg = data2d.line(5000, 5000, device)

    torch_svm = SVM(X_train_pos[0]).fit(X_train_pos)
    train.save(torch_svm, "svm")
    print(*torch_svm.named_parameters())

    sk_svm = SGDOneClassSVM(nu=0.01)
    if sk_svm is not None:
        sk_svm.fit(X_train_pos.detach().cpu().numpy())
        print("sk weights", sk_svm.coef_)
        print("sk bias (== -offset)", -sk_svm.offset_)

    with torch.no_grad():
        show_results(
            X_val_pos,
            torch_svm(X_val_pos),
            X_val_neg,
            torch_svm(X_val_neg),
            "torch"
        )
        show_results(
            X_val_pos,
            sk_svm.predict(X_val_pos.detach().cpu().numpy()),
            X_val_neg,
            sk_svm.predict(X_val_neg.detach().cpu().numpy()),
            "sk"
        )
        plt.show()
