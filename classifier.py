import gc

import torch
import torch.nn.functional as F
from torch import nn

import eval
import mnist
import train

N_CLASSES = 10


class CleverHans1(nn.Module):
    """Basic CNN architecture."""

    def __init__(self, in_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, 64, 8, 1
        )  # (batch_size, 3, 28, 28) --> (batch_size, 64, 21, 21)
        self.conv2 = nn.Conv2d(
            64, 128, 6, 2
        )  # (batch_size, 64, 21, 21) --> (batch_size, 128, 8, 8)
        self.conv3 = nn.Conv2d(
            128, 128, 5, 1
        )  # (batch_size, 128, 8, 8) --> (batch_size, 128, 4, 4)
        self.fc1 = nn.Linear(128 * 4 * 4, 128)  # (batch_size, 128, 4, 4) --> (batch_size, 2048)
        self.fc2 = nn.Linear(128, 10)  # (batch_size, 128) --> (batch_size, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 128 * 4 * 4)
        x = self.fc1(x)
        x = self.fc2(x)  # sic
        return x
    
    def activations(self, x):
        a = []
        
        gc.collect()
        a.append(F.relu(self.conv1(x)))
        a.append(F.relu(self.conv2(a[-1])))
        a.append(F.relu(self.conv3(a[-1])))
        
        gc.collect()
        a.append(a[-1].view(-1, 128 * 4 * 4))
        a[-1] = self.fc1(a[-1])
        a[-1] = self.fc2(a[-1])
        return a


class FullyConnected(nn.Module):
    def __init__(self, example_img):
        super().__init__()
        n_hidden = 800
        self.seq = nn.Sequential(
            nn.Flatten(),
            nn.Linear(example_img.numel(), n_hidden),
            nn.GELU(),
            nn.BatchNorm1d(n_hidden),
            nn.Linear(n_hidden, N_CLASSES),
        )

    def forward(self, img_batch):
        return self.seq(img_batch)

    def activations(self, img_batch):
        return eval.activations_at(self.seq, img_batch, [3, -1])


class PoolNet(nn.Module):
    # http://yann.lecun.com/exdb/publis/pdf/ranzato-cvpr-07.pdf
    def __init__(self, example_img):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(
                in_channels=example_img.size()[0],
                out_channels=50,
                kernel_size=7,
                padding=2,
            ),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(50),
            nn.Conv2d(in_channels=50, out_channels=128, kernel_size=7),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(128),
            nn.Flatten(),
        )

        with torch.no_grad():
            n_fc_inputs = self.convs(example_img.cpu().unsqueeze(0)).numel()

        print("Number of input features to fully connected layers:", n_fc_inputs)
        self.fully_connected = nn.Sequential(
            nn.Linear(n_fc_inputs, 200),
            nn.GELU(),
            nn.BatchNorm1d(200),
            nn.Linear(200, N_CLASSES),
        )

    def forward(self, img_batch):
        return self.fully_connected(self.convs(img_batch))

    def activations(self, img_batch):
        """Returns activations of hidden layers before the output"""
        activations = eval.activations_at(self.convs, img_batch, [3, -1])
        activations.extend(eval.activations_at(self.fully_connected, activations[-1], [2, -1]))
        return activations


if __name__ == "__main__":
    torch.manual_seed(98765)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = mnist.load_data(n_train=20000, n_val=2000, device=device)
    net = CleverHans1().to(device)
    train.logistic_regression(net, data, init=True, verbose=True, lr=1e-3)
    train.save(net, "ch20k")

    import eval
    with torch.no_grad():
        eval.print_multi_acc(net(data[1][0]), data[1][1], "val")

"""
The initial learning rate still has a surprisingly large influence, even when using NAdam
and ReduceLROnPlateau.


PoolNet 20k/2k:
--- Epoch 28 (20k samples per epoch)
Epoch average training loss: 6.420200572543404e-05
Last batch accuracy: 100.0%
Validation loss: 0.0311002004891634
Validation accuracy: 99.25%
Epoch 00030: reducing learning rate of group 0 to 2.5000e-05.
Epoch 00030: reducing learning rate of group 1 to 2.5000e-05.
--- Epoch 29 (20k samples per epoch)
Epoch average training loss: 5.712083758069662e-05
Last batch accuracy: 100.0%
Validation loss: 0.03310703858733177
Validation accuracy: 99.3%
NOTE: No automated commit, because on main branch
val accuracy: 99.15%


CleverHans1:
--- Epoch 27 (20k samples per epoch)
Epoch average training loss: 3.359099546750762e-05
Last batch accuracy: 100.0%
Validation loss: 0.06126902624964714
Validation accuracy: 98.75%
Epoch 00029: reducing learning rate of group 0 to 2.5000e-05.
Epoch 00029: reducing learning rate of group 1 to 2.5000e-05.
--- Epoch 28 (20k samples per epoch)
Epoch average training loss: 2.6891979191792984e-05
Last batch accuracy: 100.0%
Validation loss: 0.06206578388810158
Validation accuracy: 98.75%
NOTE: No automated commit, because on main branch
val accuracy: 98.55%
"""
