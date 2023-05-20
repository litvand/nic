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
        '''Returns activations of hidden layers along with the output'''
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


class DistanceSVM(nn.Module):
    # Linear RBF
    def __init__(self, example_input, n_centers):
        super().__init__()

        # Approximate support vectors
        self.centers = nn.Parameter(torch.randn(n_centers, example_input.numel()))

        # Relative importances of centers
        self.coefs = nn.Parameter(torch.rand(n_centers) / n_centers)

        # If an input is farther from centers than `max_avg_distance`, then the input is outside the
        # learned distribution, i.e. classified as negative.
        self.max_avg_distance = nn.Parameter(torch.ones(1))

    def forward(self, inputs):
        inputs = torch.flatten(inputs, 1)  # n_inputs, input_numel
        size = (len(inputs), len(self.centers), len(inputs[0]))  # n_inputs, n_centers, input_numel
        diffs = inputs.unsqueeze(1).expand(size) - self.centers.unsqueeze(0).expand(size)
        distances = diffs.pow(2).sum(-1).sqrt()  # n_inputs, n_centers

        # Only relative magnitudes of coefficients matter, not absolute magnitudes
        coefs = self.coefs.abs()
        coef_sum = coefs.sum()
        coefs = coefs if coef_sum.item() == 0.0 else coefs / coef_sum

        weighted_avg_distances = torch.mv(distances, coefs)  # n_inputs
        return self.max_avg_distance - weighted_avg_distances

    def regularization_loss(self):
        # Minimize `max_avg_distance`, so that the area inside the learned distribution is as small
        # as possible.
        return torch.clamp(self.max_avg_distance, min=0.0)


class DensitySVM(nn.Module):
    # Gaussian RBF
    def __init__(self, example_input, n_centers):
        super().__init__()

        # Approximate support vectors
        self.centers = nn.Parameter(torch.randn(n_centers, example_input.numel()))

        # Relative importances of centers
        self.coefs = nn.Parameter(torch.rand(n_centers) / n_centers)

        # The distributions around centers become wider and flatter as `scale` increases.
        # This parameter is called `gamma` in scikit-learn and libsvm.
        self.scale = nn.Parameter(torch.Tensor([1.0 / example_input.numel()]))

        # If an input is less probable than `min_avg_density`, then the input is outside the learned
        # distribution, i.e. classified as negative.
        self.min_avg_density = nn.Parameter(torch.Tensor([0.5]))

    def forward(self, inputs):
        square_distances = pairwise_square_L2(inputs, self.centers)
        

        # Only relative magnitudes of coefficients matter, not absolute magnitudes
        coefs = self.coefs.abs()
        coef_sum = coefs.sum()
        coefs = coefs if coef_sum.item() == 0.0 else coefs / coef_sum

        weighted_avg_densities = torch.mv(densities, coefs)  # n_inputs
        return weighted_avg_densities - self.min_avg_density

    def regularization_loss(self):
        # Maximize `min_avg_density`, so that the area inside the learned distribution is as small
        # as possible.
        return -torch.clamp(self.min_avg_density, max=1.0)


def gaussian_kernel(inputs, centers, gamma):
    '''Square L2 distance between each input and each center'''
    inputs = torch.flatten(inputs, 1)  # n_inputs, input_numel
    size = (len(inputs), len(centers), inputs[0].numel())  # n_inputs, n_centers, input_numel
    diffs = inputs.unsqueeze(1).expand(size) - centers.unsqueeze(0).expand(size)
    square_distances = diffs.pow(2).sum(-1)   # n_inputs, n_centers
    densities = torch.exp(-gamma * square_distances)  # n_inputs, n_centers
    return densities


class Nystroem(nn.Module):
    '''Approximate kernel by choosing (e.g. random) centers and then normalizing'''
    def __init__(self, n_centers):
        super().__init__()
        self.n_centers = n_centers
        self.gamma = None    # TODO: Make this a `torch.Parameter`?
        self.centers = None
        self.normalization = None

    def forward(self, inputs):
        densities = gaussian_kernel(inputs, self.centers, self.gamma)
        normalized = torch.mm(densities, self.normalization)
        return normalized


def train_nystroem(nystroem, train_inputs):  # TODO: kmeans=False
    assert len(train_inputs) <= nystroem.n_centers, (len(train_inputs), nystroem.n_centers)

    with torch.no_grad():
        # TODO: if kmeans
        center_indices = torch.randperm(len(train_inputs))[:nystroem.n_centers]
        nystroem.centers = torch.flatten(train_inputs[center_indices], 1)

        n_features = train_inputs[0].numel()
        var = train_inputs.var().item()
        nystroem.gamma = 1.0/(n_features * var) if var > 0.0 else 1.0/n_features

        # TODO: Could we use a faster matrix decomposition instead of SVD, since `center_densities`
        #       is Hermitian?
        center_densities = gaussian_kernel(nystroem.centers, nystroem.centers, nystroem.gamma)
        u, s, vh = torch.linalg.svd(center_densities, driver='gesvd')
        s = torch.clamp(s, min=1e-12)
        nystroem.normalization = torch.mm(u / s.sqrt(), vh).t()


def save(model, model_name):
    # Can look at exact parameters and data that the model was trained on using the commit.
    last_commit = git.Repo().head.object.hexsha
    torch.save(model.state_dict(), f"models/{model_name}-{last_commit}.pt")


def load(model, filename):
    model.load_state_dict(torch.load(f"models/{filename}"))
