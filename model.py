import git
import torch
from torch import nn

N_CLASSES = 10


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
            nn.BatchNorm1d(200),
            nn.GELU(),
            nn.Linear(200, N_CLASSES),
        )

    def forward(self, img_batch):
        return self.fully_connected(self.convs(img_batch))

    def layers(self, img_batch):
        """Returns activations of hidden layers along with the output"""
        layers = [self.convs(img_batch)]

        fc = list(self.fully_connected.modules())
        layers.append(fc[0](layers[-1]))

        x = layers[-1]
        for mod in fc[1:]:
            x = mod(x)
        layers.append(x)
        return layers


class Detector(nn.Module):
    def __init__(self, example_img):
        super().__init__()
        n_hidden = 800
        self.seq = nn.Sequential(
            nn.Flatten(),
            nn.Linear(example_img.numel(), n_hidden),
            nn.GELU(),
            nn.BatchNorm1d(n_hidden),
            nn.Linear(n_hidden, 2),
        )

    def forward(self, img_batch):
        return self.seq(img_batch)


def gaussian_kernel(inputs, centers, gamma):
    """Square L2 distance between each input and each center"""
    inputs = torch.flatten(inputs, 1)  # n_inputs, input_numel
    size = (
        len(inputs),
        len(centers),
        inputs[0].numel(),
    )  # n_inputs, n_centers, input_numel
    diffs = inputs.unsqueeze(1).expand(size) - centers.unsqueeze(0).expand(size)
    square_distances = diffs.pow(2).sum(-1)  # n_inputs, n_centers
    densities = torch.exp(-gamma * square_distances)  # n_inputs, n_centers
    return densities


class Nystroem(nn.Module):
    """Approximate kernel by choosing (e.g. random) centers and then normalizing"""

    def __init__(self, example_input, n_centers):
        super().__init__()
        self.n_centers = n_centers
        self.gamma = nn.Parameter(torch.empty(1), requires_grad=False)
        self.centers = nn.Parameter(
            torch.empty(n_centers, example_input.numel()), requires_grad=False
        )
        self.normalization = nn.Parameter(
            torch.empty(n_centers, n_centers), requires_grad=False
        )

    def forward(self, inputs):
        densities = gaussian_kernel(inputs, self.centers, self.gamma)
        normalized = torch.mm(densities, self.normalization)
        return normalized


class SVM(nn.Module):
    def __init__(self, example_input, n_centers, rbf=False):
        super().__init__()
        self.nystroem = Nystroem(example_input, n_centers) if rbf else None
        self.coefs = nn.Parameter(torch.rand(n_centers) / n_centers)
        self.bias = nn.Parameter(torch.Tensor([0.0]))

    def forward(self, inputs):
        if self.nystroem is not None:
            inputs = self.nystroem(inputs)
        return torch.mv(inputs, self.coefs) + self.bias


def save(model, model_name):
    # Can look at exact parameters and data that the model was trained on using the commit.
    last_commit = git.Repo().head.object.hexsha
    torch.save(model.state_dict(), f"models/{model_name}-{last_commit}.pt")


def load(model, filename):
    model.load_state_dict(torch.load(f"models/{filename}"))
