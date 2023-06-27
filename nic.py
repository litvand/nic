import matplotlib.pyplot as plt
import torch
from torch import nn

import adversary
import classifier
import mnist
import train
from eval import acc, percent, print_balanced_acc, round_tensor
from mixture import DetectorKmeans
from svm import SVM
from train import logistic_regression, Normalize, Whiten



def cat_layer_pairs(layers):
    """
    Concatenate features of each consecutive pair of layers.

    Layers must be flattened, i.e. each layer's size must be (n_imgs, n_features).

    Returns list with `len(layers) - 1` elements.
    """

    # Concatenate all features and then take slices.
    cat_all = torch.cat(layers, dim=1)
    with torch.no_grad():
        n_features = torch.tensor([layer.size(1) for layer in layers], dtype=torch.int32)
        i_next_features = n_features.cumsum(dim=0, dtype=torch.int32)
    
    pairs = []
    for i_layer in range(len(layers) - 1):
        i_cur = i_next_features[i_layer] - n_features[i_layer]
        i_next_next = i_next_features[i_layer + 1]
        pairs.append(cat_all[:, i_cur : i_next_next].contiguous())

    return pairs


def whitened_layers(whiten, X, trained_model):
    layers = [X] + trained_model.activations(X)
    return [w(l.flatten(1)) for w, l in zip(whiten, layers)]


def cat_features(features):
    """
    List of n_features tensors with shape (n_points) --> tensor with shape (n_points, n_features)
    """
    return torch.cat([f.unsqueeze(1) for f in features], dim=1)


class NIC(nn.Module):
    """
    Higher output means a higher probability of the image being within the training
    distribution, i.e. non-adversarial.

    trained_model: Model that is already trained to classify images. Should output logits for
                   each class. NIC detects whether an image is an adversarial image for
                   `trained_model`. Must have `activations` method that returns activations of
                   last layer and optionally activations of some earlier layers.
    """

    detector_types = ["kmeans", "svm"]
    final_types = ["vote", "logistic", "svm"]

    def __init__(
        self, example_x, trained_model, n_centers, detector_type="kmeans", final_type="vote"
    ):
        super().__init__()
        X = example_x.unsqueeze(0)
        trained_model.eval()
        self.final_type = final_type
        self.detector_type = detector_type
        assert final_type in NIC.final_types, (final_type, NIC.final_types)
        assert detector_type in NIC.detector_types, (detector_type, NIC.detector_types)

        with torch.no_grad():
            layers = [X] + trained_model.activations(X)
            layers = [l.flatten(1) for l in layers]
            self.whiten = nn.ModuleList([Whiten(l[0]) for l in layers])

            k = detector_type == "kmeans"
            self.value_detectors = nn.ModuleList()
            for l in layers:
                self.value_detectors.append(DetectorKmeans(l[0], n_centers) if k else SVM(l[0]))
            
            # Last layer already contains logits from trained model, so there's no need to use a
            # classifier. Last layer is also a transformation of the second-to-last layer, so it
            # isn't useful to train a classifier based on the second-to-last layer.
            logits = layers[-1][0]
            n_classes = len(logits)
            self.classifiers = nn.ModuleList(
                [nn.Linear(l.size(1), n_classes).to(X.device) for l in layers[:-2]]
            )

            pair = torch.cat((logits, logits))
            self.prov_detectors = nn.ModuleList()
            for _ in self.classifiers:
                self.prov_detectors.append(DetectorKmeans(pair, n_centers) if k else SVM(pair))

            densities = torch.zeros(1, dtype=X.dtype, device=X.device).expand(
                len(self.value_detectors) + len(self.prov_detectors)
            )
            self.final_normalize = Normalize(densities)
            if final_type in ["vote", "logistic"]:
                self.final_detector = nn.Linear(len(densities), 1).to(X.device)
            elif final_type == "svm":
                self.final_detector = SVM(densities)
            else:
                assert False, "unreachable"

    def activations(self, X, trained_model):
        trained_model.eval()

        layers = whitened_layers(self.whiten, X, trained_model)
        value_densities = [v(l) for v, l in zip(self.value_detectors, layers)]
        logits = [c(l) for c, l in zip(self.classifiers, layers[:-2])] + [layers[-1]]
        prov_densities = [p(l) for p, l in zip(self.prov_detectors, cat_layer_pairs(logits))]

        densities = value_densities + prov_densities
        densities.append(
            self.final_detector(self.final_normalize(cat_features(densities))).view(-1)
        )
        return densities
    
    def forward(self, X, trained_model):
        return self.activations(X, trained_model)[-1]

    def fit(self, train_X_pos, train_X_neg, train_y, trained_model):
        """
        Train detector to classify images as adversarial or non-adversarial

        train_X_pos: Non-adversarial images
        train_X_neg: Adversarial images
        train_y: Class indices of images in train_X_pos
        trained_model: Model already trained to classify images in train_X_pos
        """
        trained_model.eval()

        with torch.no_grad():
            train_layers = [train_X_pos] + trained_model.activations(train_X_pos)
            train_layers = [l.flatten(1) for l in train_layers]
            for whiten, layer in zip(self.whiten, train_layers):
                whiten.fit(layer)
            train_layers = [whiten(layer) for whiten, layer in zip(self.whiten, train_layers)]

            layers_neg = whitened_layers(self.whiten, train_X_neg, trained_model)

        print("--- Value detectors ---")
        value_densities, value_densities_neg = [], []
        for value_detector, layer, layer_neg in zip(self.value_detectors, train_layers, layers_neg):
            if self.detector_type == "kmeans":
                d = value_detector.fit_predict(layer)
                # d, d_neg = value_detector.fit_predict(layer, layer_neg)
                # value_densities_neg.append(d_neg)
            else:
                value_detector.fit_one_class(layer)
                d = value_detector(layer)
            value_densities.append(d)
            value_densities_neg.append(value_detector(layer_neg))
        
        print("--- NIC classifiers ---")
        logits, logits_neg = [], []
        for classifier, layer, layer_neg in zip(self.classifiers, train_layers, layers_neg):
            data = ((layer, train_y), (None, None))
            logistic_regression(classifier, data, init=True, n_epochs=10)
            logits.append(classifier(layer))
            logits_neg.append(classifier(layer_neg))
        
        logits.append(train_layers[-1])
        logits_neg.append(layers_neg[-1])
        pairs, pairs_neg = cat_layer_pairs(logits), cat_layer_pairs(logits_neg)
        
        print("--- Provenance detectors ---")
        prov_densities, prov_densities_neg = [], []
        for prov_detector, pair, pair_neg in zip(self.prov_detectors, pairs, pairs_neg):
            if self.detector_type == "kmeans":
                d = prov_detector.fit_predict(pair)
                # d, d_neg = prov_detector.fit_predict(pair, pair_neg)
                # prov_densities_neg.append(d_neg)
            else:
                prov_detector.fit_one_class(pair)
                d = prov_detector(pair)
            prov_densities.append(d)
            prov_densities_neg.append(prov_detector(pair_neg))
        
        print("--- NIC final ---")
        densities = value_densities + prov_densities
        densities_neg = value_densities_neg + prov_densities_neg

        if self.final_type == "vote":
            with torch.no_grad():
                # Accuracy on positive training examples is always 100%, so only calculate accuracy
                # on negative examples.
                self.final_detector.weight[0, :] = torch.tensor(
                    [acc(d_neg < 0.) for d_neg in densities_neg]
                )
                self.final_detector.bias.copy_(0.)

            # Default `self.final_normalize` leaves inputs unchanged.
            return self

        with torch.no_grad():
            densities = cat_features(densities)
            self.final_normalize.fit(densities)
            densities = self.final_normalize(densities)

            densities_neg = self.final_normalize(cat_features(densities_neg))

        if self.final_type == "logistic":
            is_pos = torch.zeros(
                len(densities) + len(densities_neg), dtype=densities.dtype, device=densities.device
            )
            is_pos[: len(densities)].fill_(1.)

            densities = torch.cat((densities, densities_neg))
            regression_data = ((densities, is_pos), (None, None))

            with torch.no_grad():
                self.final_detector.weight.fill_(1. / densities.size(1))
                self.final_detector.bias.copy_(0.5 * densities.size(1))

            logistic_regression(self.final_detector, regression_data, n_epochs=100)
            return self

        self.final_detector.fit_one_class(densities)
        return self


if __name__ == "__main__":
    torch.manual_seed(98765)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("--- Data ---")
    (train_X_pos, train_y), (val_X_pos, val_y) = mnist.load_data(
        n_train=20000, n_val=2000, device=device
    )

    print("--- Model ---")
    trained_model = classifier.PoolNet(train_X_pos[0])
    train.load(trained_model, "pool20k-1ce6321a452d629b14cf94ad9266ad584cd36e85")
    trained_model.to(device)

    print("--- Detector ---")
    train_X_neg = train_X_pos.clone()
    adversary.fgsm_(train_X_neg, train_y, trained_model, 0.2)
    n_centers = 1 + len(train_X_pos) // 100
    detector = NIC(train_X_pos[0], trained_model, n_centers)
    # detector.fit(train_X_pos, train_X_neg, train_y, trained_model)
    # train.save(detector, f"nic{n_centers}-onpool20k")
    train.load(detector, "nic201-onpool20k-ce97eb7455a89a045849b0ccd55c3e6a3bc62763")

    print("--- Validation ---")
    val_X_neg = val_X_pos.clone()
    adversary.fgsm_(val_X_neg, val_y, trained_model, 0.2)

    with torch.no_grad():
        train_a_pos = detector.activations(train_X_pos, trained_model)
        train_a_neg = detector.activations(train_X_neg, trained_model)
        val_a_pos = detector.activations(val_X_pos, trained_model)
        val_a_neg = detector.activations(val_X_neg, trained_model)
        print("Index of first provenance detector:", len(detector.value_detectors))
        for i_detector in range(len(train_a_pos)):
            print("i_detector", i_detector)
            print_balanced_acc(train_a_pos[i_detector], train_a_neg[i_detector], "Training")
            print_balanced_acc(val_a_pos[i_detector], val_a_neg[i_detector], "Validation")
