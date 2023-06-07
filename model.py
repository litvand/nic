import math
import random

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


def pairwise_sqr_dists(points, centers):
    """Square L2 distance between each point and each center (dists[i_point, i_center])"""
    size = (
        len(points),
        len(centers),
        points.size(1),
    )  # n_points, n_centers, n_features
    diffs = points.unsqueeze(1).expand(size) - centers.unsqueeze(0).expand(size)
    square_distances = diffs.pow(2).sum(-1)  # n_points, n_centers
    return square_distances


def gaussian_kernel(inputs, centers, gamma):
    square_distances = pairwise_sqr_dists(inputs.flatten(1), centers)
    densities = torch.exp(-gamma * square_distances)  # n_inputs, n_centers
    return densities


def kmeans_farthest(train_points, n_centers):
    """
    train_points: Training points (size = n_points, n_features)
    n_centers: Number of centers

    Returns: Cluster centers (size = n_centers, train_points.size(1))
    """

    assert len(train_points) >= n_centers, (train_points, n_centers)

    dists = None  # Distance of each point from its closest center
    centers = torch.empty(n_centers, train_points.size(1), device=train_points.device)
    centers[0] = train_points[random.randint(0, len(train_points) - 1)]
    for i_center in range(1, n_centers):
        dists = pairwise_sqr_dists(train_points, centers[:i_center]).min(1)[0]

        # Point that is farthest from all previous centers
        centers[i_center] = train_points[dists.argmax()[0]]

    avg_dist = torch.inf
    while True:
        dists, closest_centers = pairwise_sqr_dists(train_points, centers).min(1)
        new_avg_dist = dists.mean()
        if new_avg_dist >= avg_dist * (1 - 1e-4):
            # new_avg_dist isn't much better than avg_dist
            break

        avg_dist = new_avg_dist
        for i_center in range(n_centers):
            # OPTIM: train_points[nn.functional.one_hot(closest_centers)[:, i_center]]?
            centers[i_center] = train_points[closest_centers == i_center].mean(0)

    return centers


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
        self.inv_std = nn.Parameter(torch.tensor(1 / std), requires_grad=False)

    def forward(self, inputs):
        size = [1] * inputs.ndim
        size[1] = len(self.mean)
        return (inputs - self.mean.expand(size)) * self.inv_std.expand(size)


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
        cat_all[:, ends[i] - n_layer_features[i] : ends[i + 1]]
        for i in range(len(layers) - 1)
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
        final_svm,
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


def get_optimizer(Optimizer, model, weight_decay=0, **kwargs):
    """
    Split parameters of `model` into those that will experience weight decay and those that won't
    (biases, batch norm, layer norm, embedding) and those that won't be updated at all. Based on
    https://github.com/karpathy/minGPT/blob/3ed14b2cec0dfdad3f4b2831f2b4a86d11aef150/mingpt/model.py#L136
    """

    decay, no_decay = [], []
    whitelist_modules = (torch.nn.Linear,)
    blacklist_modules = ("LayerNorm", "BatchNorm", "Embedding")
    for module in model.modules():
        for param_name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue  # Won't be updated.

            elif any(n in type(module).__name__ for n in blacklist_modules):
                no_decay.append(param)  # Don't decay blacklist.

            elif param_name.endswith("bias"):
                no_decay.append(param)  # Don't decay biases.

            elif param_name.endswith("weight") and isinstance(
                module, whitelist_modules
            ):
                decay.append(param)  # Decay whitelist weights.

            elif param_name == "coefs" and isinstance(module, SVM):
                decay.append(param)  # Decay SVM coefficients.

            else:
                print(f"Warning: assuming weight_decay=0 for {param_name}")
                no_decay.append(param)

    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0},
    ]
    optimizer = Optimizer(groups, **kwargs)
    return optimizer


def gradient_noise(model, i_batch, initial_variance=0.01):
    with torch.no_grad():
        sd = math.sqrt(initial_variance / (1 + i_batch) ** 0.55)
        for param in model.parameters():
            param.grad.add_(torch.randn_like(param.grad), alpha=sd)


def save(model, model_name):
    # Can look at exact parameters and data that the model was trained on using the commit.
    last_commit = git.Repo().head.object.hexsha
    torch.save(model.state_dict(), f"models/{model_name}-{last_commit}.pt")


def load(model, filename):
    model.load_state_dict(torch.load(f"models/{filename}"))
