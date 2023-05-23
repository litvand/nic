import git
import torch
from torch import nn

N_CLASSES = 10


def get_activations(input, sequential, module_indices):
    """Get activations from modules inside an `nn.Sequential` at indices in `module_indices`."""

    sequential = list(sequential.modules())
    activations = []
    for i_module, module in enumerate(sequential):
        input = module(input)
        # Support negative indices
        if i_module in module_indices or i_module - len(sequential) in module_indices:
            activations.append(input)

    assert len(activations) == len(module_indices), (
        activations,
        sequential,
        module_indices,
    )
    return activations


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
        return get_activations(img_batch, self.seq, [3, 4])


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
        activations = get_activations(img_batch, self.convs, [3, -1])
        activations.extend(
            get_activations(activations[-1], self.fully_connected, [2, 3])
        )
        return activations


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
    """
    Approximate a kernel by choosing (random or kmeans) centers and then normalizing.

    Currently the kernel is Gaussian, but other kernels are also possible.
    """

    def __init__(self, example_input, n_centers, kmeans):
        super().__init__()
        # Use kmeans to choose centers. Generally n_centers needs to be larger if kmeans=False.
        self.kmeans = kmeans
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
    def __init__(self, example_input, n_centers, rbf=True, kmeans=True):
        super().__init__()
        self.nystroem = Nystroem(example_input, n_centers, kmeans) if rbf else None
        self.coefs = nn.Parameter(torch.rand(n_centers) / n_centers)
        self.bias = nn.Parameter(torch.Tensor([0]))

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
