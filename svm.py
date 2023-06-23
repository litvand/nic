from copy import deepcopy

import matplotlib.pyplot as plt
import torch

from sklearn.svm import SVC
# from sklearn.linear_model import SGDOneClassSVM
from torch import nn

import data2d
from eval import acc, percent
import train


def hinge_loss(outputs, margin):
    return torch.clamp(margin - outputs, min=0).mean()


class SVM(nn.Module):
    def __init__(self, example_x):
        super().__init__()
        self.linear = nn.Linear(len(example_x), 1).to(example_x.device)
        with torch.no_grad():
            self.linear.weight.fill_(1. / len(example_x))
            self.linear.bias.fill_(0.)

    def forward(self, X):
        return self.linear(X).view(-1)
    
    def fit(self, train_X_pos, train_X_neg, verbose=False, n_epochs=1000, margin=0.5, lr=0.1):
        optimizer = train.get_optimizer(torch.optim.NAdam, self.linear, weight_decay=0., lr=lr)
        min_loss = torch.inf
        min_state = None

        for epoch in range(n_epochs):
            outputs_pos, outputs_neg = self.forward(train_X_pos), self.forward(train_X_neg)
            loss = hinge_loss(outputs_pos, margin) + hinge_loss(-outputs_neg, margin)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                if verbose or epoch == n_epochs - 1:
                    print(f"Epoch {epoch}/{n_epochs}; loss {loss.item()}")
                
                if loss <= min_loss:
                    min_loss = loss
                    min_state = deepcopy(self.linear.state_dict())
        
        if min_state is not None:
            self.linear.load_state_dict(min_state)        
        return self

    def fit_one_class(
        self,
        train_X_pos,
        verbose=False,
        perfect_train=False,
        n_epochs=1000,
        margin=1.
    ):
        """
        Trains SVM to give a positive output for all training inputs, while minimizing the total set
        of inputs for which its output is positive.

        Gradient descent version of "Support Vector Method for Novelty Detection", Platt et al,
        1999. Replicates sklearn OneClassSVM given same parameters.

        train_X_pos: Positive training inputs; no targets since one class
        perfect_train: Whether to postprocess bias so that the output is positive for all
                       positive training inputs, i.e. prioritize true positives.
        batch_size: Number of training inputs per batch
        n_epochs: Number of passes through training data
        """

        # In range [0; 1]; lower nu --> higher importance of including positive examples
        nu = 0.01 if perfect_train else 0.5
        lr = 0.01
        assert lr * nu / 2 <= 1, (lr, nu)  # `lr * nu/2 > 1` breaks regularization.

        # Momentum doesn't work well with one class
        optimizer = train.get_optimizer(torch.optim.SGD, self.linear, weight_decay=nu / 2, lr=lr)
        min_loss = torch.inf
        min_state = None

        for epoch in range(n_epochs):
            loss = hinge_loss(self.forward(train_X_pos), margin) + nu * self.linear.bias
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                if verbose or epoch == n_epochs - 1:
                    print(f"Epoch {epoch}/{n_epochs}; loss {loss.item()}")
                
                if loss <= min_loss:
                    min_loss = loss
                    min_state = deepcopy(self.linear.state_dict())

        with torch.no_grad():
            if min_state is not None:
                self.linear.load_state_dict(min_state)

            if perfect_train:
                # Adjust bias by choosing minimum output that avoids false negatives.
                min_output_should_be = 0.01
                self.linear.bias += min_output_should_be - self.linear(train_X_pos).min()
            else:
                self.linear.bias -= margin  # Replicates sklearn OneClassSVM

        return self


def show_results(X_pos, outputs_pos, X_neg, outputs_neg, *args):
    # data2d.scatter_outputs_y(X_pos, outputs_pos, X_neg, outputs_neg, *args)

    acc_on_pos, acc_on_neg = acc(outputs_pos >= 0), acc(outputs_neg < 0)
    print("Balanced accuracy:", percent(0.5 * (acc_on_pos + acc_on_neg)))
    print("True positive, true negative rate:", percent(acc_on_pos), percent(acc_on_neg))


if __name__ == "__main__":
    for run in range(5):
        print(f"-------------------------- Run {run} --------------------------")
        device = "cuda"
        train_X_pos, train_X_neg, val_X_pos, val_X_neg = data2d.point(5000, 5000, device)

        torch_svm = SVM(train_X_pos[0])
        print("Training SVM")
        torch_svm.fit(train_X_pos, train_X_neg, margin=0.1, lr=0.1, n_epochs=600)
        print("Trained SVM")
        print(*torch_svm.named_parameters())

        # sk_svm = SGDOneClassSVM(nu=0.5)
        sk_svm = SVC(kernel="linear")
        if sk_svm is not None:
            train_X = torch.cat((train_X_pos, train_X_neg), dim=0)
            train_y = torch.zeros(len(train_X), dtype=torch.int32)
            train_y[:len(train_X_pos)].fill_(1)
            sk_svm.fit(train_X.cpu().numpy(), train_y.numpy())
            # print("sk weights", sk_svm.coef_)
            # print("sk bias (== -offset)", -sk_svm.offset_)

        with torch.no_grad():
            show_results(val_X_pos, torch_svm(val_X_pos), val_X_neg, torch_svm(val_X_neg), "torch")
            show_results(
                val_X_pos,
                torch.tensor(sk_svm.predict(val_X_pos.detach().cpu().numpy())),
                val_X_neg,
                torch.tensor(sk_svm.predict(val_X_neg.detach().cpu().numpy())),
                "sk",
            )
            plt.show()
