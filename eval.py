import torch

import mnist
import model


def print_accuracy(msg, outputs, labels):
    # Convert logits to class indices.
    output_classes = torch.argmax(outputs, 1)
    accuracy = torch.sum(output_classes == labels) / float(len(labels))
    print(msg, round(accuracy.item(), 3))
    return accuracy.item()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, (val_imgs, val_labels) = mnist.load_data(n_train=22000, n_valid=2000, device=device)
    m = model.PoolNet(val_imgs[0]).to(device)
    model.load(m, "pool20k-18dab86434e82bce7472c09da5f82864a6424e86.pt")
    m.eval()
    print_accuracy("Validation accuracy", m(val_imgs[:10000]), val_labels[:10000])


# FullyConnected, 50k training images, 0.984 max validation accuracy
# PoolNet BatchNorm, 20k training images, 0.99 max validation accuracy
