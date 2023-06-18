import matplotlib.pyplot as plt
import torch
from torch import nn

import adversary
import classifier
import data2d
import mnist
import train
from eval import acc, percent
from mixture import DetectorKmeans
from train import Normalize, Whiten


class ZipN(nn.Module):
    def __init__(self, modules):
        super().__init__()
        self.modules = modules

    def __iter__(self):
        return iter(self.modules)

    def __len__(self):
        return len(self.modules)

    def forward(self, inputs_iter):
        return [m(inputs) for m, inputs in zip(self.modules, inputs_iter)]


def cat_layer_pairs(layers):
    """
    Concatenate activations of each consecutive pair of layers.

    Layers must be flattened, i.e. each layer's size must be (n_inputs, -1).

    Returns list with `len(layers) - 1` elements.
    """

    cat_all = torch.cat(layers, dim=1)
    with torch.no_grad():
        n_layer_features = torch.tensor([layer.size(1) for layer in layers], dtype=torch.int32)
        ends = n_layer_features.cumsum(dim=0, dtype=torch.int32)
    return [cat_all[:, ends[i] - n_layer_features[i] : ends[i + 1]] for i in range(len(layers) - 1)]


def get_whitened_layers(whitening, trained_model, batch):
    layers = [batch] + trained_model.activations(batch)
    return whitening(layer.flatten(1) for layer in layers)


def cat_features(features):
    """
    List of n_features tensors with shape (n_points) --> tensor with shape (n_points, n_features)
    """
    return torch.cat([f.unsqueeze(1) for f in features], dim=1)


class NIC(nn.Module):
    def __init__(self, example_img, trained_model, n_centers):
        super().__init__()

        dtype, device = example_img.dtype, example_img.device

        trained_model.eval()
        with torch.no_grad():
            imgs = example_img.unsqueeze(0)
            layers = [imgs] + trained_model.activations(imgs)
            layers = [l.flatten(1) for l in layers]
            n_classes = layers[-1].size(1)  # Last layer's number of features

        self.whitening = ZipN([Whiten(a[0]) for a in layers])
        
        self.value_detectors = ZipN([DetectorKmeans(layers[-1][0], n_centers)])

        self.layer_classifiers = ZipN([
            nn.Linear(a.size(1), n_classes).to(device) for a in layers[:-2]
        ])
        self.prov_detectors = ZipN([
            DetectorKmeans(layers[-1][0], n_centers) for _ in self.layer_classifiers
        ])

        densities = torch.empty(1, dtype=dtype, device=device).expand(
            len(self.value_detectors) + len(self.prov_detectors)
        )
        self.final_whiten = Whiten(densities)
        self.final_detector = DetectorKmeans(densities, n_centers)
        print("inited nic")

    def forward(self, batch, trained_model):

        trained_model.eval()
        layers = get_whitened_layers(self.whitening, trained_model, batch)
        value_densities = self.value_detectors(layers[-1:])

        # Last layer already contains logits from trained model, so there's no need to use a
        # classifier.
        layer_logits = self.layer_classifiers(layers[:-2]) + [layers[-1]]
        for logits in layer_logits:
            assert logits.size(1) == layer_logits[0].size(1), (logits, layer_logits)
        prov_densities = self.prov_detectors(cat_layer_pairs(layer_logits))

        densities = cat_features(value_densities + prov_densities)
        return self.final_detector(self.final_whiten(densities))

    def fit(self, data, trained_model):

        (train_inputs, train_targets), (val_inputs, val_targets) = data
        trained_model.eval()
        n_centers = len(self.final_detector.center)

        with torch.no_grad():
            print("Whiten NIC layers")
            train_layers = [train_inputs] + trained_model.activations(train_inputs)
            train_layers = [layer.flatten(1) for layer in train_layers]
            print("train_layers", train_layers)
            for whiten, layer in zip(self.whitening, train_layers):
                whiten.fit(layer)
            train_layers = self.whitening(train_layers)

            val_layers = get_whitened_layers(self.whitening, trained_model, val_inputs)

        train_value_densities = train_layer_detectors(
            self.value_detectors, train_layers[-1:], "Value"
        )

        train_logits = train_layer_classifiers(
            self.layer_classifiers, train_layers, train_targets, val_layers, val_targets
        )
        train_prov_densities = train_layer_detectors(
            self.prov_detectors, cat_layer_pairs(train_logits), "Provenance"
        )

        print("Final whiten")
        with torch.no_grad():
            train_densities = cat_features(train_value_densities + train_prov_densities)
            self.final_whiten = Whiten(train_densities[0]).fit(train_densities)
            train_densities = self.final_whiten(train_densities)

        self.final_detector = DetectorKmeans(train_densities[0], n_centers).fit(train_densities)
        return self


def train_layer_classifiers(classifiers, train_layers, train_targets, val_layers, val_targets):
    # The last layer of `train_layers` already contains logits, and the next-to-last layer is
    # just a linear transformation of the last layer, so exclude the last two layers.
    train_logits, classifiers = [], []
    for i_layer, (train_layer, val_layer) in enumerate(zip(train_layers[:-2], val_layers[:-2])):
        print(f"--- Classifier {i_layer}/{len(train_layers)} ---")
        classifier = nn.Linear(train_layer.size(1), n_classes).to(train_layer.device)
        data = ((train_layer, train_targets), (val_layer, val_targets))
        train.logistic_regression(classifier, data, init=True, n_epochs=10)

        with torch.no_grad():
            train_logits.append(classifier(train_layer))
        classifiers.append(classifier)

    train_logits.append(train_layers[-1])  # Last layer already contains logits
    classifiers = ZipN(classifiers)
    return classifiers, train_logits


def train_layer_detectors(detectors, train_layers, detector_name):
    train_densities = []
    for i_layer, train_layer in enumerate(train_layers):
        print(f"--- {detector_name} detector {i_layer}/{len(train_layers)} ---")
        train_densities.append(detectors[i_layer].fit_predict(train_layer))
    
    return train_densities


class DetectorNIC(nn.Module):
    """
    Higher output means a higher probability of the input image being within the training
    distribution, i.e. non-adversarial.

    trained_model: Model that is already trained to classify something. Should output logits for
                    each class. NIC detects whether an input is an adversarial input for
                    `trained_model`. Must have `activations` method that returns activations of
                    last layer and optionally activations of some earlier layers.
    """

    def __init__(self, example_img, trained_model, n_centers):
        super().__init__()
        imgs = example_img.unsqueeze(0)
        trained_model.eval()

        with torch.no_grad():
            layers = [imgs] + trained_model.activations(imgs)
            layers = [l.flatten(1) for l in layers]
            self.whiten = nn.ModuleList([Whiten(l[0]) for l in layers])
            self.value_detectors = nn.ModuleList([DetectorKmeans(l[0], n_centers) for l in layers])
            density = torch.empty(
                1, dtype=example_img.dtype, device=example_img.device   # rm
            )
            self.final_whiten = Normalize(density)
            self.final_detector = DetectorKmeans(density, n_centers)
    
    def forward(self, inputs, trained_model):
        trained_model.eval()

        layers = [inputs] + trained_model.activations(inputs)
        layers = [l.flatten(1) for l in layers]
        print("whiten")
        layers = [whiten(layer) for whiten, layer in zip(self.whiten, layers)]
        print([layer.shape for layer in layers])
        
        print("value detectors")
        densities = []
        for value_detector, layer in zip(self.value_detectors, layers):
            densities.append(value_detector(layer))

        return densities[0]  # rm
        densities = self.final_whiten(densities[0].view(-1, 1))
        print("whitened densities", densities.max(), densities.mean(), densities.min())

        print("final")
        return self.final_detector(densities), densities   # rm

    def fit(self, train_inputs, trained_model):
        """
        data: ((training_inputs, training_targets), (validation_inputs, validation_targets))
              Targets should be class indices.
        """
        trained_model.eval()
        
        with torch.no_grad():
            train_layers = [train_inputs] + trained_model.activations(train_inputs)
            train_layers = [l.flatten(1) for l in train_layers]
            print("whiten")
            for whiten, layer in zip(self.whiten, train_layers):
                whiten.fit(layer)
            train_layers = [whiten(layer) for whiten, layer in zip(self.whiten, train_layers)]

        print("value detectors")
        densities = []
        for value_detector, layer in zip(self.value_detectors, train_layers):
            densities.append(value_detector.fit_predict(layer))

        with torch.no_grad():
            # densities = cat_features(densities)
            print("final whiten")
            self.final_whiten.fit(densities[0].view(-1, 1), unit_range=True)   # rm
            densities = self.final_whiten(densities[0].view(-1, 1))   # rm
        
        print("final detector")
        self.final_detector.fit(densities)
        return self


if __name__ == "__main__":
    torch.manual_seed(98765)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("data")
    data = mnist.load_data(n_train=20000, n_val=2000, device=device)
    train_inputs = data[0][0]

    print("model")
    trained_model = classifier.FullyConnected(train_inputs[0]).to(device)
    train.load(trained_model, "fc20k-dc84d9b97f194b36c1130a5bc82eda5d69a57ad2")

    print("detector")
    n_centers = 2 + len(train_inputs) // 100
    detector = DetectorNIC(train_inputs[0], trained_model, n_centers)
    detector.fit(train_inputs, trained_model)
    
    print("fgsm")
    val_inputs_neg = data[1][0].clone()
    adversary.fgsm_(val_inputs_neg, data[1][1], trained_model, 0.2)
    with torch.no_grad():
        val_outputs_pos = detector(data[1][0], trained_model)
        val_outputs_neg = detector(val_inputs_neg, trained_model)
    print(
        "val outputs pos", val_outputs_pos.max(), val_outputs_pos.mean(), val_outputs_pos.min(),
        "neg", val_outputs_neg.max(), val_outputs_neg.mean(), val_outputs_neg.min())
    print(
        "max, min neg < 0",
        val_outputs_neg[val_outputs_neg < 0].max(),
        val_outputs_neg[val_outputs_neg < 0].min()
    )
    print("val acc", percent(acc(val_outputs_pos >= 0)), percent(acc(val_outputs_neg < 0)))
    data2d.scatter_outputs_y(
        val_outputs_pos.view(-1, 1).expand(-1, 2),
        val_outputs_pos,
        val_outputs_neg.view(-1, 1).expand(-1, 2),
        val_outputs_neg,
        f"{type(detector).__name__} validation",
        # centers=detector.final_detector.center.expand(-1, 2)
    )
    plt.show()
    # train.save(detector, f"nic{n_centers}-onfc20k")

    # print("nic")
    # nic = NIC(
    #     train_inputs[0], trained_model, n_centers=2 + len(train_inputs) // 50
    # ).fit(data, trained_model)
    # train.save(nic, "nic-onfc20k")
    # # train.load(nic, "nic-onfc20k-036721ae464e534d53585e0a62bf0ecc6c25405f")

    # detector_val_imgs, detector_val_targets = adversary.fgsm_detector_data(
    #     data[1][0], data[1][1], trained_model, 0.2
    # )
    # with torch.no_grad():
    #     nic.eval()
    #     eval.print_bin_acc(
    #         nic(detector_val_imgs, trained_model), detector_val_targets == 1, "NIC"
    #     )
