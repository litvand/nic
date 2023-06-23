import matplotlib.pyplot as plt
import torch
from torch import nn

import adversary
import classifier
import data2d
import mnist
import train
from eval import acc, percent, round_tensor
from mixture import DetectorKmeans
from svm import SVM
from train import Normalize, Whiten


def cat_layer_pairs(layers):
    """
    Concatenate activations of each consecutive pair of layers.

    Layers must be flattened, i.e. each layer's size must be (n_imgs, -1).

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


# class NIC(nn.Module):
#     def __init__(self, example_x, trained_model, n_centers):
#         super().__init__()

#         dtype, device = example_x.dtype, example_x.device

#         trained_model.eval()
#         with torch.no_grad():
#             X = example_x.unsqueeze(0)
#             layers = [X] + trained_model.activations(X)
#             layers = [l.flatten(1) for l in layers]
#             n_classes = layers[-1].size(1)  # Last layer's number of features

#         self.whitening = ZipN([Whiten(a[0]) for a in layers])

#         self.value_detectors = ZipN([DetectorKmeans(layers[-1][0], n_centers)])

#         self.layer_classifiers = ZipN(
#             [nn.Linear(a.size(1), n_classes).to(device) for a in layers[:-2]]
#         )
#         self.prov_detectors = ZipN(
#             [DetectorKmeans(layers[-1][0], n_centers) for _ in self.layer_classifiers]
#         )

#         densities = torch.empty(1, dtype=dtype, device=device).expand(
#             len(self.value_detectors) + len(self.prov_detectors)
#         )
#         self.final_whiten = Whiten(densities)
#         self.final_detector = DetectorKmeans(densities, n_centers)
#         print("inited nic")

#     def forward(self, batch, trained_model):
#         trained_model.eval()
#         layers = get_whitened_layers(self.whitening, trained_model, batch)
#         value_densities = self.value_detectors(layers[-1:])

#         # Last layer already contains logits from trained model, so there's no need to use a
#         # classifier.
#         layer_logits = self.layer_classifiers(layers[:-2]) + [layers[-1]]
#         for logits in layer_logits:
#             assert logits.size(1) == layer_logits[0].size(1), (logits, layer_logits)
#         prov_densities = self.prov_detectors(cat_layer_pairs(layer_logits))

#         densities = cat_features(value_densities + prov_densities)
#         return self.final_detector(self.final_whiten(densities))

#     def fit(self, data, trained_model):
#         (train_X_pos, train_targets), (val_imgs, val_targets) = data
#         trained_model.eval()
#         n_centers = len(self.final_detector.center)

#         with torch.no_grad():
#             print("Whiten NIC layers")
#             train_layers = [train_X_pos] + trained_model.activations(train_X_pos)
#             train_layers = [layer.flatten(1) for layer in train_layers]
#             print("train_layers", train_layers)
#             for whiten, layer in zip(self.whitening, train_layers):
#                 whiten.fit(layer)
#             train_layers = self.whitening(train_layers)

#             val_layers = get_whitened_layers(self.whitening, trained_model, val_imgs)

#         train_value_densities = train_layer_detectors(
#             self.value_detectors, train_layers[-1:], "Value"
#         )

#         train_logits = train_layer_classifiers(
#             self.layer_classifiers, train_layers, train_targets, val_layers, val_targets
#         )
#         train_prov_densities = train_layer_detectors(
#             self.prov_detectors, cat_layer_pairs(train_logits), "Provenance"
#         )

#         print("Final whiten")
#         with torch.no_grad():
#             train_densities = cat_features(train_value_densities + train_prov_densities)
#             self.final_whiten = Whiten(train_densities[0]).fit(train_densities)
#             train_densities = self.final_whiten(train_densities)

#         self.final_detector = DetectorKmeans(train_densities[0], n_centers).fit(train_densities)
#         return self


class NIC(nn.Module):
    """
    Higher output means a higher probability of the img image being within the training
    distribution, i.e. non-adversarial.

    trained_model: Model that is already trained to classify something. Should output logits for
                    each class. NIC detects whether an img is an adversarial img for
                    `trained_model`. Must have `activations` method that returns activations of
                    last layer and optionally activations of some earlier layers.
    """

    def __init__(self, example_x, trained_model, n_centers):
        super().__init__()
        X = example_x.unsqueeze(0)
        trained_model.eval()

        with torch.no_grad():
            layers = [X] + trained_model.activations(X)
            layers = [l.flatten(1) for l in layers]
            self.whiten = nn.ModuleList([Whiten(l[0]) for l in layers])
            self.value_detectors = nn.ModuleList([DetectorKmeans(l[0], n_centers) for l in layers])
            densities = torch.zeros(len(self.value_detectors), dtype=X.dtype, device=X.device)
            self.final_whiten = Normalize(densities)
            self.final_detector = SVM(densities)

    def forward(self, X, trained_model):
        trained_model.eval()

        layers = [X] + trained_model.activations(X)
        layers = [l.flatten(1) for l in layers]
        layers = [whiten(layer) for whiten, layer in zip(self.whiten, layers)]

        densities = []
        for value_detector, layer in zip(self.value_detectors, layers):
            densities.append(value_detector(layer))

        densities = self.final_whiten(cat_features(densities))
        print("NIC.forward layer densities")
        print([(round_tensor(densities[:, i].max()), round_tensor(densities[:, i].mean()), round_tensor(densities[:, i].min())) for i in range(densities.size(1))])

        return self.final_detector(densities)

    def fit(self, train_X_pos, train_X_neg, trained_model):
        trained_model.eval()

        with torch.no_grad():
            train_layers = [train_X_pos] + trained_model.activations(train_X_pos)
            train_layers = [l.flatten(1) for l in train_layers]
            for whiten, layer in zip(self.whiten, train_layers):
                whiten.fit(layer)
            train_layers = [whiten(layer) for whiten, layer in zip(self.whiten, train_layers)]

        densities = []
        for value_detector, layer in zip(self.value_detectors, train_layers):
            densities.append(value_detector.fit_predict(layer))

        with torch.no_grad():
            densities = cat_features(densities)
            self.final_whiten.fit(densities)
            densities = self.final_whiten(densities)

            layers_neg = [train_X_neg] + trained_model.activations(train_X_neg)
            densities_neg = self.final_whiten(cat_features(
                v(w(l.flatten(1))) for v, w, l in zip(self.value_detectors, self.whiten, layers_neg)
            ))

            print("NIC.fit layer densities")
            print([(round_tensor(densities[:, i].max()), round_tensor(densities[:, i].mean()), round_tensor(densities[:, i].min())) for i in range(densities.size(1))])
            print("neg")
            print([(round_tensor(densities_neg[:, i].max()), round_tensor(densities_neg[:, i].mean()), round_tensor(densities_neg[:, i].min())) for i in range(densities_neg.size(1))])

        self.final_detector.fit(densities, densities_neg, n_epochs=1000, margin=0.5, lr=0.1)
        print("final detector params:", *self.final_detector.named_parameters())
        return self


if __name__ == "__main__":
    torch.manual_seed(98765)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("data")
    (train_X_pos, train_y), (val_X_pos, val_y) = mnist.load_data(
        n_train=5000, n_val=2000, device=device
    )

    print("model")
    trained_model = classifier.FullyConnected(train_X_pos[0]).to(device)
    train.load(trained_model, "fc20k-dc84d9b97f194b36c1130a5bc82eda5d69a57ad2")

    print("detector")
    train_X_neg = train_X_pos.clone()
    adversary.fgsm_(train_X_neg, train_y, trained_model, 0.2)
    detector = NIC(train_X_pos[0], trained_model, n_centers=1 + len(train_X_pos)//100)
    detector.fit(train_X_pos, train_X_neg, trained_model)
    print(
        "train acc",
        percent(acc(detector(train_X_pos, trained_model) >= 0)),
        percent(acc(detector(train_X_neg, trained_model) < 0))
    )

    print("fgsm")
    val_X_neg = val_X_pos.clone()
    adversary.fgsm_(val_X_neg, val_y, trained_model, 0.2)
    with torch.no_grad():
        val_outputs_pos = detector(val_X_pos, trained_model)
        val_outputs_neg = detector(val_X_neg, trained_model)
    print(
        "val outputs pos",
        round_tensor(val_outputs_pos.max()),
        round_tensor(val_outputs_pos.mean()),
        round_tensor(val_outputs_pos.min()),
        "neg",
        round_tensor(val_outputs_neg.max()),
        round_tensor(val_outputs_neg.mean()),
        round_tensor(val_outputs_neg.min()),
    )
    print("val acc", percent(acc(val_outputs_pos >= 0)), percent(acc(val_outputs_neg < 0)))
    # train.save(detector, f"nic{n_centers}-onfc20k")
