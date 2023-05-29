from importlib.metadata import requires
from typing import Any
import git
import torch
from torch import nn

N_CLASSES = 10


def activations_at(inputs, sequential, module_indices):
    """Get activations from modules inside an `nn.Sequential` at indices in `module_indices`."""

    sequential = list(sequential.modules())
    activations = []
    for i_module, module in enumerate(sequential):
        inputs = module(inputs)
        # Support negative indices
        if i_module in module_indices or i_module - len(sequential) in module_indices:
            activations.append(inputs)

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
        return activations_at(img_batch, self.seq, [3, 4])


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
        activations = activations_at(img_batch, self.convs, [3, -1])
        activations.extend(
            activations_at(activations[-1], self.fully_connected, [2, 3])
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
    Approximate a kernel by choosing (e.g. random/kmeans) centers and then normalizing.

    Currently the kernel is Gaussian, though other kernels are also possible.
    """

    def __init__(self, example_input, n_centers, kmeans=True):
        super().__init__()
        # Use kmeans to choose centers. Generally n_centers needs to be larger if kmeans=False.
        self.kmeans = kmeans
        self.n_centers = n_centers
        self.gamma = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.centers = nn.Parameter(
            torch.zeros(n_centers, example_input.numel()), requires_grad=False
        )
        self.normalization = nn.Parameter(
            torch.zeros(n_centers, n_centers), requires_grad=False
        )

    def forward(self, inputs):
        densities = gaussian_kernel(inputs, self.centers, self.gamma)
        normalized = torch.mm(densities, self.normalization)
        return normalized


class SVM(nn.Module):
    def __init__(self, example_input, n_centers, rbf=True):
        super().__init__()
        self.nystroem = Nystroem(example_input, n_centers) if rbf else None
        self.coefs = nn.Parameter(torch.rand(n_centers) - 0.5)
        self.bias = nn.Parameter(torch.Tensor([0]))

    def forward(self, inputs):
        if self.nystroem is not None:
            inputs = self.nystroem(inputs)
        return torch.mv(inputs, self.coefs) + self.bias


class Normalize(nn.Module):
    def __init__(self, mean, std):
        """
        Normalize the mean and standard deviation

        mean: Single number or mean of each channel (i.e. second dimension)
        std: Single number or standard deviation of each channel (i.e. second dimension)
        """

        super().__init__()
        self.mean = nn.Parameter(torch.tensor(mean), requires_grad=False)
        self.inv_std = nn.Parameter(torch.tensor(1/std), requires_grad=False)

    def forward(self, inputs):
        size = [1] * inputs.ndim
        size[1] = len(self.mean)
        return (inputs - self.mean.view(size)) * self.inv_std.view(size)


class Elemwise(nn.Module):
    def __init__(self, modules):
        super().__init__()
        self.modules = modules

    def forward(self, elems):
        return [m(elem) for m, elem in zip(self.modules, elems)]


def cat_layer_pairs(layers):
    """
    Concatenate features of each consecutive pair of layers.

    Layers must be flattened, i.e. each layer's size must be (n_inputs, -1).

    Returns list with `len(layers) - 1` elements.
    """

    cat_all = torch.cat(layers, dim=1)
    with torch.no_grad():
        n_layer_features = torch.Tensor([layer.size(1) for layer in layers])
        ends = n_layer_features.cumsum()
    return [
        cat_all[:, ends[i] - n_layer_features[i]:ends[i+1]] for i in range(len(layers) - 1)
    ]


class NIC(nn.Module):
    @staticmethod
    def load(filename):
        n = NIC(None, [], [], [], None, None)
        load(n, filename)
        return n

    def __init__(
        self,
        trained_model,
        layers_normalize,
        value_svms,
        provenance_svms,
        density_normalize,
        final_svm
    ):
        super().__init__()
        self.trained_model = trained_model
        self.layers_normalize = Elemwise(layers_normalize)
        self.value_svms = Elemwise(value_svms)
        self.provenance_svms = Elemwise(provenance_svms)
        self.density_normalize = density_normalize
        self.final_svm = final_svm

    def forward(self, batch):
        """
        Higher output means a higher probability of the input image being within the training
        distribution, i.e. non-adversarial.
        """
        layers = [batch] + self.trained_model.activations(batch)
        layers = [layer.flatten(1) for layer in self.layers_normalize(layers)]
        value_densities = self.value_svms(layers)
        provenance_densities = self.provenance_svms(cat_layer_pairs(layers))
        densities = value_densities + provenance_densities
        densities = torch.cat([d.unsqueeze(1) for d in densities], dim=1)
        return self.final_svm(self.density_normalize(densities))


def save(model, model_name):
    # Can look at exact parameters and data that the model was trained on using the commit.
    last_commit = git.Repo().head.object.hexsha
    torch.save(model.state_dict(), f"models/{model_name}-{last_commit}.pt")


def load(model, filename):
    model.load_state_dict(torch.load(f"models/{filename}"))
