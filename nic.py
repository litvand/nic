import foolbox as fb
import sklearn
import torch
from torch import cuda, nn

import adversary
import classifier
import mnist
import train
from cache import TensorCache
from eval import acc, batched_activations, percent, print_balanced_acc, print_multi_acc
from mixture import DetectorKmeans
from train import Normalize, logistic_regression


class SkSVM:
    def __init__(self, n_centers=None):
        self.nystroem = (
            None
            if n_centers is None
            else sklearn.kernel_approximation.Nystroem(n_components=n_centers)
        )
        self.svm = sklearn.linear_model.SGDOneClassSVM()

    def fit_predict(self, X):
        dtype, device = X.dtype, X.device
        X = X.detach().cpu().numpy()
        if self.nystroem is not None:
            X = self.nystroem.fit_transform(X)
        return torch.tensor(self.svm.fit_predict(X), dtype=dtype, device=device)

    def __call__(self, X):
        dtype, device = X.dtype, X.device
        X = X.detach().cpu().numpy()
        if self.nystroem is not None:
            X = self.nystroem.transform(X)
        return torch.tensor(self.svm.predict(X), dtype=dtype, device=device)


def batched_whiten_fit(whiten, X, batch_size):
    for i_x in range(0, len(X), batch_size):
        i_next = i_x + batch_size
        whiten.fit(X[i_x:i_next], finish=i_next >= len(X))
    return whiten


def cache_layers(whiten, X, trained_model, fit=False):
    batch_size = 10000
    cache = TensorCache(dir="./tmp")
    X_flat = X.flatten(1)
    if fit:
        batched_whiten_fit(whiten[0], X_flat, batch_size)
    cache.append(whiten[0](X_flat))

    # Assumes each layer can fit in memory.
    for i_a, a in enumerate(batched_activations(trained_model, X, batch_size)):
        i_layer = i_a + 1
        a = a.flatten(1).to(X.device)
        if fit:
            batched_whiten_fit(whiten[i_layer], a, batch_size)

        n_nan = a.sum(1).isnan().count_nonzero().item()
        if n_nan > 0:
            print(f"ERROR: layer {i_layer} before whiten, n_nan {n_nan}, numel {a.numel()}")
        whiten[i_layer](a, inplace=True)

        n_nan = a.sum(1).isnan().count_nonzero().item()
        if n_nan > 0:
            print(f"ERROR: layer {i_layer} after whiten, n_nan {n_nan}, numel {a.numel()}")
        cache.append(a)

    return cache


def maybe_load(module, filename):
    return module if module is not None else torch.load("./tmp/" + filename)


def fit_detectors(detectors, layers, layers_neg, _detector_type, filename):
    was_none = detectors is None
    detectors = maybe_load(detectors, filename)

    densities, densities_neg = [], []
    for i_layer, (detector, layer) in enumerate(zip(detectors, layers)):
        print(f"- Layer {i_layer} -")
        densities.append(detector.fit_predict(layer.contiguous().cuda()))
        if layers_neg is not None:
            densities_neg.append(detector(layers_neg[i_layer].contiguous().cuda()))

    if was_none:
        torch.save(detectors, "./tmp/" + filename)
    return densities, densities_neg


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
        pairs.append(cat_all[:, i_cur:i_next_next])

    return pairs


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
    final_types = ["min", "vote", "logistic", "svm"]

    def __init__(
        self, example_x, trained_model, n_centers, detector_type="kmeans", final_type="min"
    ):
        super().__init__()
        X = example_x.unsqueeze(0)
        trained_model.eval()
        self.final_type = final_type
        self.detector_type = detector_type
        assert final_type in NIC.final_types, (final_type, NIC.final_types)
        assert detector_type in NIC.detector_types, (detector_type, NIC.detector_types)

        with torch.no_grad():
            layers = [X] + list(trained_model.activations(X))
            layers = [l.flatten(1) for l in layers]
            self.whiten = nn.ModuleList(
                [
                    # (Whiten if l.size(1) < 1000 else Normalize)(l[0]) for l in layers
                    Normalize(l[0])
                    for l in layers
                ]
            )

            k = detector_type == "kmeans"
            self.value_detectors = [] if detector_type == "svm" else nn.ModuleList()
            for l in layers:
                self.value_detectors.append(
                    DetectorKmeans(l[0], n_centers) if k else SkSVM(n_centers)
                )

            # Last layer already contains logits from trained model, so there's no need to use a
            # classifier. Last layer is also a transformation of the second-to-last layer, so it
            # isn't useful to train a classifier based on the second-to-last layer.
            logits = layers[-1][0]
            n_classes = len(logits)
            self.classifiers = nn.ModuleList(
                [nn.Linear(l.size(1), n_classes).to(X.device) for l in layers[:-2]]
            )

            pair_example = torch.cat((logits, logits))
            self.prov_detectors = [] if detector_type == "svm" else nn.ModuleList()
            for _ in layers[:-2]:
                self.prov_detectors.append(
                    DetectorKmeans(pair_example, n_centers) if k else SkSVM(n_centers)
                )

            densities = torch.zeros(1, dtype=X.dtype, device=X.device).expand(
                len(layers) + len(layers[:-2])
            )
            self.final_normalize = Normalize(densities, channelwise=False)
            if final_type == "min":
                self.final_detector = None
            elif final_type in ["vote", "logistic"]:
                self.final_detector = nn.Linear(len(densities), 1).to(X.device)
            elif final_type == "svm":
                self.final_detector = SkSVM(None)
            else:
                assert False, "unreachable"

    def activations(self, X, trained_model):
        trained_model.eval()

        print("NIC.activations whiten")
        whiten = maybe_load(self.whiten, "w")
        layers = cache_layers(whiten, X, trained_model)
        whiten = None

        print("NIC.activations value detectors")
        value_detectors = maybe_load(self.value_detectors, "vd")
        value_densities = [v(l.to(X.device)) for v, l in zip(value_detectors, layers)]
        value_detectors = None

        classifiers = maybe_load(self.classifiers, "c")
        logits = [classifiers[i](layers[i].to(X.device)) for i in range(len(classifiers))] + [
            layers[-1].to(X.device)
        ]
        classifiers = None
        logits = cat_layer_pairs(logits)

        print("NIC.activations provenance detectors")
        prov_detectors = maybe_load(self.prov_detectors, "pd")
        prov_densities = [p(l.contiguous()) for p, l in zip(prov_detectors, logits)]
        prov_detectors = None

        densities = value_densities + prov_densities
        d = self.final_normalize(cat_features(densities))
        densities.append(
            d.min(-1)[0] if self.final_type == "min" else self.final_detector(d).view(-1)
        )
        return iter(densities)

    def forward(self, X, trained_model):
        return tuple(self.activations(X, trained_model))[-1]

    def fit(self, train_X_pos, train_y, trained_model, train_X_neg=None):
        """
        Train detector to classify images as adversarial or non-adversarial

        train_X_pos: Non-adversarial images
        train_X_neg: Adversarial images; can be `None` depending on final_type
                     (Learning is unsupervised aside from optionally the final detector.)
        train_y: Class indices of images in train_X_pos
        trained_model: Model already trained to classify images in train_X_pos
        """
        print("--- NIC.fit ---")
        trained_model.eval()

        with torch.no_grad():
            whiten = maybe_load(self.whiten, "w")
            layers = cache_layers(whiten, train_X_pos, trained_model, fit=True)

            print(f"whiten neg: {round(cuda.memory_allocated() / 1e9, 2)} GB")
            layers_neg = (
                None
                if self.final_type == "min"
                else cache_layers(whiten, train_X_neg, trained_model)
            )

            if self.whiten is None:
                torch.save(whiten, "./tmp/w")
            whiten = None

        print("--- Value detectors ---")
        value_densities, value_densities_neg = fit_detectors(
            self.value_detectors, layers, layers_neg, self.detector_type, "vd"
        )

        print("--- NIC classifiers ---")
        logits, logits_neg = [], []
        classifiers = maybe_load(self.classifiers, "c")
        for i_layer, c in enumerate(classifiers):
            data = ((layers[i_layer].to(train_y.device), train_y), (None, None))
            logistic_regression(c, data, init=True, batch_size=256, n_epochs=100, lr=1e-2)
            logits.append(c(data[0][0]))
            if layers_neg is not None:
                logits_neg.append(c(layers_neg[i_layer].to(train_y.device)))
        if self.classifiers is None:
            torch.save(classifiers, "./tmp/c")
        classifiers = None

        logits.append(layers[len(layers) - 1].to(train_y.device))
        pairs = cat_layer_pairs(logits)
        if layers_neg is not None:
            logits_neg.append(layers_neg[len(layers) - 1].to(train_y.device))
            pairs_neg = cat_layer_pairs(logits_neg)
        else:
            pairs_neg = None

        print("--- Provenance detectors ---")
        prov_densities, prov_densities_neg = fit_detectors(
            self.prov_detectors, pairs, pairs_neg, self.detector_type, "pd"
        )

        print("--- NIC final ---")
        if self.final_type == "min":
            return self

        densities = value_densities + prov_densities
        densities_neg = value_densities_neg + prov_densities_neg
        if self.final_type == "vote":
            with torch.no_grad():
                # Accuracy on positive training examples is always 100%, so only calculate accuracy
                # on negative examples.
                self.final_detector.weight[0, :] = torch.tensor(
                    [acc(d_neg < 0.0) for d_neg in densities_neg]
                )
                self.final_detector.bias.copy_(0.0)

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
            is_pos[: len(densities)].fill_(1.0)

            densities = torch.cat((densities, densities_neg))
            regression_data = ((densities, is_pos), (None, None))

            with torch.no_grad():
                self.final_detector.weight.fill_(1.0 / densities.size(1))
                self.final_detector.bias.copy_(0.5 * densities.size(1))

            logistic_regression(
                self.final_detector, regression_data, batch_size=5096, n_epochs=100, lr=0.1
            )
            return self

        self.final_detector.fit_predict(densities)
        return self


if __name__ == "__main__":
    torch.manual_seed(98765)
    device = "cuda" if cuda.is_available() else "cpu"

    print("--- Data ---")
    (train_X_pos, train_y), (val_X_pos, val_y) = mnist.load_data(
        n_train=20000, n_val=2000, device=device
    )

    print("--- Model ---")
    trained_model = classifier.PoolNet(train_X_pos[0])
    train.load(trained_model, "pool-norestart20k-2223b6b48b3680297dda4cb0f644d39268753dca")
    trained_model.to(device)
    print_multi_acc(trained_model(val_X_pos), val_y, "trained_model validation")

    print("--- Detector ---")

    n_centers = 1 + len(train_X_pos) // 100
    detector = NIC(train_X_pos[0], trained_model, n_centers, detector_type="kmeans")

    # detector.fit(train_X_pos, train_y, trained_model)
    # train.save(detector, f"nic{n_centers}-onPool20k")
    train.load(detector, f"nic201-onPool20k-5113214ed23a3231ef757ad0b073e21fc780b95a")

    print("--- Validation ---")
    eps = 0.1
    sanity_check = False
    if sanity_check:
        # Should be equivalent to foolbox FGSM:
        val_X_neg = val_X_pos.clone()
        adversary.fgsm_(val_X_neg, val_y, trained_model, eps)
        print_multi_acc(trained_model(val_X_neg), val_y, "trained_model on val_X_neg")

    val_X_fb, _, success = fb.attacks.FGSM()(
        fb.PyTorchModel(trained_model, (-100.0, 100.0)),
        val_X_pos,
        fb.Misclassification(val_y),
        epsilons=[eps],
    )
    val_X_fb = val_X_fb[0]
    print(
        "trained_model acc on val_X_fb:",
        percent(1.0 - success[0].count_nonzero() / float(len(success[0])))
    )
    if sanity_check:
        # Should be same as `1 - success`:
        print_multi_acc(trained_model(val_X_neg), val_y, "trained_model on val_X_fb")

    with torch.no_grad():
        if sanity_check:
            train_a_pos = tuple(detector.activations(train_X_pos, trained_model))
            val_a_neg = tuple(detector.activations(val_X_neg, trained_model))
            print("Index of first provenance detector:", len(detector.value_detectors))
        
        val_a_pos = tuple(detector.activations(val_X_pos, trained_model))
        val_a_fb = tuple(detector.activations(val_X_fb, trained_model))
        print(
            "Index of first provenance detector:",
            (len(val_a_pos) + 1) // 2 if len(val_a_pos) > 3 else None,
        )
        for i_detector in range(len(val_a_pos)):
            print("i_detector", i_detector)            
            if sanity_check:
                # Accuracy should be 100% (non-adversarial training images):
                print("Detector train pos acc", percent(acc(train_a_pos[i_detector] >= 0.0)))
                # Should be same as accuracy on foolbox images:
                print_balanced_acc(
                    val_a_pos[i_detector], val_a_neg[i_detector], "Detector validation"
                )
            
            print_balanced_acc(
                val_a_pos[i_detector], val_a_fb[i_detector], "Detector validation"
            )

"""
fgsm(eps=0.2)

Sklearn SVM (value detectors, provenance detectors and final detector) on PoolNet 20k/2k:
--- Validation ---
Index of first provenance detector: 5
i_detector 0
Training balanced accuracy; true positives and negatives: 75.12% 50.26% 99.97%
Validation balanced accuracy; true positives and negatives: 75.3% 50.65% 99.95%
i_detector 1
Training balanced accuracy; true positives and negatives: 71.17% 42.38% 99.95%
Validation balanced accuracy; true positives and negatives: 58.42% 16.85% 100.0%
i_detector 2
Training balanced accuracy; true positives and negatives: 66.63% 49.58% 83.67%
Validation balanced accuracy; true positives and negatives: 64.5% 44.05% 84.95%
i_detector 3
Training balanced accuracy; true positives and negatives: 64.74% 49.77% 79.71%
Validation balanced accuracy; true positives and negatives: 63.8% 47.15% 80.45%
i_detector 4
Training balanced accuracy; true positives and negatives: 67.72% 50.1% 85.32%
Validation balanced accuracy; true positives and negatives: 67.58% 47.35% 87.8%
i_detector 5
Training balanced accuracy; true positives and negatives: 50.5% 1.0% 100.0%
Validation balanced accuracy; true positives and negatives: 50.0% 0% 100.0%
i_detector 6
Training balanced accuracy; true positives and negatives: 50.49% 0.99% 99.99%
Validation balanced accuracy; true positives and negatives: 50.0% 0% 100.0%
i_detector 7
Training balanced accuracy; true positives and negatives: 50.0% 100.0% 0%
Validation balanced accuracy; true positives and negatives: 50.0% 100.0% 0%
i_detector 8
Training balanced accuracy; true positives and negatives: 33.87% 67.73% 0.01%
Validation balanced accuracy; true positives and negatives: 43.48% 86.95% 0%


Kmeans detector on same net & data:
--- Validation ---
Index of first provenance detector: 5
i_detector 0
Training balanced accuracy; true positives and negatives: 99.9%       100.0% 99.79%
Validation balanced accuracy; true positives and negatives: 99.8%       99.9% 99.7%
i_detector 1
Training balanced accuracy; true positives and negatives: 99.9%       100.0% 99.79%
Validation balanced accuracy; true positives and negatives: 95.65%       91.6% 99.7%
i_detector 2
Training balanced accuracy; true positives and negatives: 50.05%       100.0% 0.09%
Validation balanced accuracy; true positives and negatives: 50.05%       100.0% 0.1%
i_detector 3
Training balanced accuracy; true positives and negatives: 50.08%       100.0% 0.15%
Validation balanced accuracy; true positives and negatives: 50.08%       100.0% 0.15%
i_detector 4
Training balanced accuracy; true positives and negatives: 50.59%       100.0% 1.17%
Validation balanced accuracy; true positives and negatives: 50.75%       100.0% 1.5%
i_detector 5
Training balanced accuracy; true positives and negatives: 99.9%       100.0% 99.79%
Validation balanced accuracy; true positives and negatives: 98.33%       96.95% 99.7%
i_detector 6
Training balanced accuracy; true positives and negatives: 99.9%       100.0% 99.79%
Validation balanced accuracy; true positives and negatives: 97.28%       94.85% 99.7%
i_detector 7
Training balanced accuracy; true positives and negatives: 50.01%       100.0% 0.02%
Validation balanced accuracy; true positives and negatives: 50.0%       100.0% 0%
i_detector 8
Training balanced accuracy; true positives and negatives: 99.9%       100.0% 99.79%
Validation balanced accuracy; true positives and negatives: 99.05%       98.4% 99.7%


Sklearn SVM except final detector vote:
--- Validation ---
Index of first provenance detector: 5
i_detector 0
Training balanced accuracy; true positives and negatives: 74.97%       49.95% 99.98%
Validation balanced accuracy; true positives and negatives: 75.08%       50.2% 99.95%
i_detector 1
Training balanced accuracy; true positives and negatives: 70.69%       41.41% 99.96%
Validation balanced accuracy; true positives and negatives: 58.5%       17.0% 100.0%
i_detector 2
Training balanced accuracy; true positives and negatives: 66.81%       50.59% 83.03%
Validation balanced accuracy; true positives and negatives: 64.83%       45.0% 84.65%
i_detector 3
Training balanced accuracy; true positives and negatives: 64.76%       49.57% 79.95%
Validation balanced accuracy; true positives and negatives: 63.88%       47.05% 80.7%
i_detector 4
Training balanced accuracy; true positives and negatives: 67.55%       49.28% 85.82%
Validation balanced accuracy; true positives and negatives: 67.48%       46.7% 88.25%
i_detector 5
Training balanced accuracy; true positives and negatives: 50.5%       1.0% 100.0%
Validation balanced accuracy; true positives and negatives: 50.0%       0% 100.0%
i_detector 6
Training balanced accuracy; true positives and negatives: 50.0%       100.0% 0%
Validation balanced accuracy; true positives and negatives: 50.0%       100.0% 0%
i_detector 7
Training balanced accuracy; true positives and negatives: 50.49%       0.99% 99.99%
Validation balanced accuracy; true positives and negatives: 50.0%       0% 100.0%
i_detector 8
Training balanced accuracy; true positives and negatives: 67.72%       35.44% 99.99%
Validation balanced accuracy; true positives and negatives: 63.08%       26.15% 100.0%
"""
