from copy import deepcopy

import git
import torch
import torch.nn.functional as F

import mnist
import model
from adversary import fgsm_detector_data
from eval import print_accuracy
from lsuv_init import LSUV_
from svm import train_one_class


def logistic_regression(net, data, lr=1e-2, batch_size=100, n_epochs=100, weight_decay=1e-9):
    net.train()
    (train_inputs, train_labels), (valid_inputs, valid_labels) = data

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    min_valid_loss = float("inf")
    min_valid_state = net.state_dict()

    for epoch in range(n_epochs):
        perm = torch.randperm(len(train_inputs))
        train_inputs, train_labels = train_inputs[perm], train_labels[perm]

        loss = None
        for i_input in range(0, len(train_inputs), batch_size):
            batch_inputs = train_inputs[i_input:i_input + batch_size]
            batch_labels = train_labels[i_input:i_input + batch_size]
            batch_outputs = net(batch_inputs)
            loss = F.cross_entropy(batch_outputs, batch_labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            net.eval()
            print(f"Epoch {epoch} ({len(train_inputs)//1000}k samples per epoch)")
            print(f"Last batch loss {loss.item()}")
            print_accuracy("Batch accuracy", batch_outputs, batch_labels)

            valid_outputs = net(valid_inputs)
            valid_loss = F.cross_entropy(valid_outputs, valid_labels).item()
            print("Validation loss", valid_loss)
            print_accuracy("Validation accuracy", valid_outputs, valid_labels)

            if valid_loss <= min_valid_loss:
                if min_valid_loss < float("inf"):
                    # Don't overwrite saved model if loss was just < infinity.
                    min_valid_state = deepcopy(net.state_dict())
                min_valid_loss = valid_loss

            net.train()
    net.eval()
    net.load_state_dict(min_valid_state)


def train_detector(trained_model, data, load_name=None, eps=0.2):
    detector_data = (
        fgsm_detector_data(data[0], trained_model, eps),
        fgsm_detector_data(data[1], trained_model, eps),
    )
    detector = model.Detector(detector_data[0][0][0]).to(data[0][0].device)
    if load_name is not None:
        model.load(detector, load_name)
    else:
        LSUV_(detector, detector_data[0][0][:2000])
    logistic_regression(detector, detector_data, "detect")


def train_layer_svms(train_layers, valid_layers, svm_name):
    svms, train_densities, valid_densities = [], [], []
    for i_layer, train_inputs in enumerate(train_layers):
        print(f"--- {svm_name} SVM #{i_layer} ---")
        svm = model.SVM(train_inputs[0], 100).to(train_inputs.device)
        train_one_class(svm, train_inputs, valid_layers[i_layer])
        svms.append(svm)
        train_densities.append(svm(train_inputs))
        valid_densities.append(svm(valid_layers[i_layer]))
    
    return svms, train_densities, valid_densities


def train_nic(trained_model, data):
    """
    trained_model: Model that is already trained to classify images. NIC detects whether an image is
                   an adversarial image for `trained_model`.
    data: ((training_images, _), (validation_images, _)). Image tensors should have size
          (n_images, height, width, n_channels).
    """
    with torch.no_grad():
        (train_imgs, _), (valid_imgs, _) = data
        assert train_imgs.ndim == 4, train_imgs.size()

        train_layers = [train_imgs] + trained_model.activations(train_imgs)
        normalization = [model.Normalize(layer.mean(-1), layer.std(-1)) for layer in train_layers]
        train_layers = [n(layer).flatten(1) for n, layer in zip(normalization, train_layers)]

        valid_layers = [valid_imgs] + trained_model.activations(valid_imgs)
        valid_layers = [n(layer).flatten(1) for n, layer in zip(normalization, valid_layers)]

    train_densities, valid_densities = [], []

    value_svms = []

    # logits = []
    # The last layer of `train_layers` already contains logits, and the last layer of
    # `train_layers` is a linear transformation of the next to last layer.

    provenance_svms = []
    train_pairs = model.cat_layer_pairs(train_layers)
    valid_pairs = model.cat_layer_pairs(valid_layers)
    for i_layer, inputs in enumerate(train_pairs):
        print(f"--- Provenance SVM #{i_layer} ---")
        svm = model.SVM(inputs[0], 100).to(inputs.device)
        train_one_class(svm, inputs, valid_pairs[i_layer])
        provenance_svms.append(svm)


    final_svm = model.SVM()


def git_commit():
    repo = git.Repo()
    if repo.active_branch.name == "main":  # Don't clutter the main branch.
        print("NOTE: No automated commit, because on main branch")
        return

    repo.git.add(".")
    repo.git.commit("-m", "_Automated commit")


if __name__ == "__main__":
    git_commit()
    torch.manual_seed(98765)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = mnist.load_data(n_train=20000, n_valid=2000, device=device)
    train_imgs = data[0][0]

    m = model.PoolNet(train_imgs[0]).to(device)
    # LSUV_(m, train_imgs[:2000])
    model.load(m, "pool20k-18dab86434e82bce7472c09da5f82864a6424e86.pt")
    # train_model(m, data, 'pool20k')
    # train_detector(m, data)
    train_nic(m, data)
