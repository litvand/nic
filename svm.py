from copy import deepcopy

import matplotlib.pyplot as plt
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
        self.linear = nn.Linear(len(x_example), 1).to(x_example.device)

    def forward(self, X):
        return self.linear(X).view(-1)
    
    def fit(self, X_train_pos, X_train_neg, verbose=False, n_epochs=300, margin=0.5):
        optimizer = train.get_optimizer(torch.optim.NAdam, self.linear, weight_decay=0., lr=0.1)
        min_loss = torch.inf
        min_state = None

        for epoch in range(n_epochs):
            outputs_pos, outputs_neg = self.forward(X_train_pos), self.forward(X_train_neg)
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
        X_train_pos,
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

        X_train_pos: Positive training inputs; no targets since one class
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
            loss = hinge_loss(self.forward(X_train_pos), margin) + nu * self.linear.bias
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
                self.linear.bias += min_output_should_be - self.linear(X_train_pos).min()
            else:
                self.linear.bias -= margin  # Replicates sklearn OneClassSVM

        return self


def show_results(X_pos, outputs_pos, X_neg, outputs_neg, *args):
    # data2d.scatter_outputs_y(X_pos, outputs_pos, X_neg, outputs_neg, *args)

    acc_on_pos, acc_on_neg = acc(outputs_pos >= 0), acc(outputs_neg < 0)
    print("Balanced accuracy:", percent(0.5 * (acc_on_pos + acc_on_neg)))
    print("True positive, true negative rate:", percent(acc_on_pos), percent(acc_on_neg))


if __name__ == "__main__":
    for run in range(10):
        print(f"-------------------------- Run {run} --------------------------")
        device = "cuda"
        X_train_pos, X_train_neg, X_val_pos, X_val_neg = data2d.overlap(5000, 5000, device)

        torch_svm = SVM(X_train_pos[0])
        print("Training SVM")
        torch_svm.fit(X_train_pos, X_train_neg)
        print("Trained SVM")
        print(*torch_svm.named_parameters())

        # sk_svm = SGDOneClassSVM(nu=0.5)
        # if sk_svm is not None:
        #     sk_svm.fit(X_train_pos.detach().cpu().numpy())
        #     print("sk weights", sk_svm.coef_)
        #     print("sk bias (== -offset)", -sk_svm.offset_)

        with torch.no_grad():
            show_results(X_val_pos, torch_svm(X_val_pos), X_val_neg, torch_svm(X_val_neg), "torch")
            # show_results(
            #     X_val_pos,
            #     torch.tensor(sk_svm.predict(X_val_pos.detach().cpu().numpy())),
            #     X_val_neg,
            #     torch.tensor(sk_svm.predict(X_val_neg.detach().cpu().numpy())),
            #     "sk",
            # )
            plt.show()
