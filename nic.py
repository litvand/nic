import torch
from torch import nn

import adversary
import classifier
import eval
import mnist
import train
from cavegmm import DetectorMixture
from train import Normalize


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
    return [cat_all[:, ends[i] - n_layer_features[i]: ends[i + 1]] for i in range(len(layers) - 1)]


def get_whitened_layers(whitening, trained_model, batch):
    layers = [batch] + trained_model.activations(batch)
    return whitening(layer.flatten(1) for layer in layers)


def cat_features(features):
    """
    List of n_features tensors with shape (n_points) --> tensor with shape (n_points, n_features)
    """
    return torch.cat([f.unsqueeze(1) for f in features], dim=1)


class NIC(nn.Module):
    def __init__(self, example_img, trained_model):
        super().__init__()

        trained_model.eval()
        with torch.no_grad():
            imgs = example_img.unsqueeze(0)
            layers = [imgs] + trained_model.activations(imgs)
            n_classes = layers[-1].size(1)  # Last layer's number of features

        self.whitening = ZipN([Normalize(a[0]) for a in layers])
        self.value_detectors = ZipN([DetectorMixture()])
        self.layer_classifiers = ZipN([nn.Linear(a.size(1), n_classes) for a in layers[:-2]])
        self.prov_detectors = ZipN([DetectorMixture() for _ in self.layer_classifiers])
        self.final_whiten = Normalize(
            torch.zeros(len(self.value_detectors) + len(self.prov_detectors))
        )
        self.final_detector = DetectorMixture()

    def forward(self, batch, trained_model):
        """
        Higher output means a higher probability of the input image being within the training
        distribution, i.e. non-adversarial.

        trained_model: Model that is already trained to classify something. Should output logits for
                       each class. NIC detects whether an input is an adversarial input for
                       `trained_model`. Must have `activations` method that returns activations of
                       last layer and optionally activations of some earlier layers.
        """

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
        """
        data: ((training_inputs, training_targets), (validation_inputs, validation_targets))
              Targets should be class indices.

        trained_model: Model that is already trained to classify something. Should output logits for
                       each class. NIC detects whether an input is an adversarial input for
                       `trained_model`. Must have `activations` method that returns activations of
                       last layer and optionally activations of some earlier layers.
        """
        (train_inputs, train_targets), (val_inputs, val_targets) = data
        trained_model.eval()

        detector_args = {
            "num_components": min(10, 2 + len(train_inputs) // 100),
            "covariance_type": "spherical",
            "init_strategy": "kmeans",
            "batch_size": 5000,  # Largest size that fits in memory
        }

        with torch.no_grad():
            print("Whiten NIC layers")
            train_layers = [train_inputs] + trained_model.activations(train_inputs)
            self.whitening = ZipN([Normalize(layer[0]).fit(layer) for layer in train_layers])
            train_layers = self.whitening(layer.flatten(1) for layer in train_layers)
            cat_layer_pairs(train_layers)
            val_layers = get_whitened_layers(self.whitening, trained_model, val_inputs)
        
        print("train_layers", train_layers)

        self.value_detectors, train_value_densities = train_layer_detectors(
            train_layers[-1:], detector_args, "Value"
        )

        self.layer_classifiers, train_logits = train_layer_classifiers(
            train_layers, train_targets, val_layers, val_targets, n_classes=train_layers[-1].size(1)
        )
        self.prov_detectors, train_prov_densities = train_layer_detectors(
            cat_layer_pairs(train_logits), detector_args, "Provenance"
        )

        print("Final whiten")
        with torch.no_grad():
            train_densities = cat_features(train_value_densities + train_prov_densities)
            self.final_whiten = Normalize(train_densities[0]).fit(train_densities)
            train_densities = self.final_whiten(train_densities)

        self.final_detector = DetectorMixture(**detector_args).fit(train_densities)
        return self


def train_layer_classifiers(train_layers, train_targets, val_layers, val_targets, n_classes):
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


def train_layer_detectors(train_layers, detector_args, detector_name):
    train_densities, detectors = [], []
    for i_layer, train_layer in enumerate(train_layers):
        print(f"--- {detector_name} detector {i_layer}/{len(train_layers)} ---")
        detector = DetectorMixture(**detector_args)

        train_densities.append(detector.fit_predict(train_layer))
        detectors.append(detector)

    detectors = ZipN(detectors)
    return detectors, train_densities


if __name__ == '__main__':
    torch.manual_seed(98765)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = mnist.load_data(n_train=20000, n_val=2000, device=device)

    trained_model = classifier.FullyConnected(data[0][0][0]).to(device)
    train.load(trained_model, "fc20k-dc84d9b97f194b36c1130a5bc82eda5d69a57ad2")

    nic = NIC(data[0][0][0], trained_model).to(device)  # .fit(data, trained_model)
    # train.save(nic, "nic-onfc20k")
    train.load(nic, "nic-onfc20k-036721ae464e534d53585e0a62bf0ecc6c25405f")

    detector_val_imgs, detector_val_targets = adversary.fgsm_detector_data(
        data[1][0], data[1][1], trained_model, 0.2
    )
    with torch.no_grad():
        nic.eval()
        eval.print_bin_acc(nic(detector_val_imgs, trained_model), detector_val_targets == 1, "NIC")
