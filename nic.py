import torch
from torch import nn


class Zip(nn.Module):
    def __init__(self, modules):
        super().__init__()
        self.modules = modules

    def forward(self, inputs_list):
        return [m(inputs) for m, inputs in zip(self.modules, inputs_list)]


def cat_layer_pairs(layers):
    """
    Concatenate activations of each consecutive pair of layers.

    Layers must be flattened, i.e. each layer's size must be (n_inputs, -1).

    Returns list with `len(layers) - 1` elements.
    """

    cat_all = torch.cat(layers, dim=1)
    with torch.no_grad():
        n_layer_features = torch.Tensor([layer.size(1) for layer in layers])
        ends = n_layer_features.cumsum()
    return [cat_all[:, ends[i] - n_layer_features[i] : ends[i + 1]] for i in range(len(layers) - 1)]


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
        self.layers_normalize = Zip(layers_normalize)
        self.value_svms = Zip(value_svms)
        self.provenance_svms = Zip(provenance_svms)
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


def train_layer_svms(train_layers, val_layers, svm_name):
    svms, train_densities, val_densities = [], [], []
    for i_layer, train_inputs in enumerate(train_layers):
        print(f"--- {svm_name} SVM #{i_layer} ---")
        svm = model.SVM(train_inputs[0], 100).to(train_inputs.device)
        train_one_class(svm, train_inputs, val_layers[i_layer])
        svms.append(svm)
        train_densities.append(svm(train_inputs))
        val_densities.append(svm(val_layers[i_layer]))

    return svms, train_densities, val_densities


def train_nic(trained_model, data):
    """
    trained_model: Model that is already trained to classify images. NIC detects whether an image is
                   an adversarial image for `trained_model`.
    data: ((training_images, _), (validation_images, _)). Image tensors should have size
          (n_images, height, width, n_channels).
    """
    with torch.no_grad():
        (train_imgs, _), (val_imgs, _) = data
        assert train_imgs.ndim == 4, train_imgs.size()

        train_layers = [train_imgs] + trained_model.activations(train_imgs)
        normalization = [model.Normalize(layer.mean(-1), layer.std(-1)) for layer in train_layers]
        train_layers = [n(layer).flatten(1) for n, layer in zip(normalization, train_layers)]

        val_layers = [val_imgs] + trained_model.activations(val_imgs)
        val_layers = [n(layer).flatten(1) for n, layer in zip(normalization, val_layers)]

    train_densities, val_densities = [], []

    value_svms = []

    # logits = []
    # The last layer of `train_layers` already contains logits, and the last layer of
    # `train_layers` is a linear transformation of the next to last layer.

    provenance_svms = []
    train_pairs = model.cat_layer_pairs(train_layers)
    val_pairs = model.cat_layer_pairs(val_layers)
    for i_layer, inputs in enumerate(train_pairs):
        print(f"--- Provenance SVM #{i_layer} ---")
        svm = model.SVM(inputs[0], 100).to(inputs.device)
        train_one_class(svm, inputs, val_pairs[i_layer])
        provenance_svms.append(svm)

    final_svm = model.SVM()


if __name__ == "__main__":
    detector = Detector(imgs[0]).to(device)
    model.load(detector, "detect-18dab86434e82bce7472c09da5f82864a6424e86.pt")
    detector.eval()

    with torch.no_grad():
        prs_original_adv = prs_adv(detector, imgs)
        prs_adv_adv = prs_adv(detector, untargeted_imgs)

    print(
        f"Predicted probability that original images are adversarial {torch.mean(prs_original_adv)}"
    )
    print(
        f"Predicted probability that adversarial images are adversarial {torch.mean(prs_adv_adv)}"
    )
    plot_distr_overlap(prs_original_adv, prs_adv_adv)

    for threshold in [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.05, 0.99, 0.995, 0.999]:
        print(
            f"Detector accuracy on original images with threshold {threshold}:",
            torch.sum(prs_original_adv < threshold) / len(imgs),
        )
        print(
            f"Detector accuracy on adversarial images with threshold {threshold}:",
            torch.sum(prs_adv_adv > threshold) / len(imgs),
        )

# Fully connected detector taking just the raw image as input can detect 90% of adversarial images
# while classifying 90% of normal images correctly, or detect 50% of adversarial images while
# classifying 99.5% of normal images correctly. Detecting 99% of adversarial images would mean
# classifying only 3% of normal images correctly.
