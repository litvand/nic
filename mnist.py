import matplotlib.pyplot as plt
import torch

TRAIN_PIXEL_MEAN = 33.0
TRAIN_PIXEL_STDDEV = 79.0


def targets_from_file(filename, n_imgs=-1):
    with open(filename, "rb") as f:
        magic = int.from_bytes(f.read(4), byteorder="big")
        assert magic == 2049, magic

        max_imgs = int.from_bytes(f.read(4), byteorder="big")
        if n_imgs < 0:
            n_imgs = max_imgs
        assert n_imgs <= max_imgs, (n_imgs, max_imgs)

        targets = f.read(n_imgs)

    return torch.frombuffer(targets, dtype=torch.uint8).clone()


def imgs_from_file(filename, n_imgs=-1):
    with open(filename, "rb") as f:
        magic = int.from_bytes(f.read(4), byteorder="big")
        assert magic == 2051, magic

        max_imgs = int.from_bytes(f.read(4), byteorder="big")
        if n_imgs < 0:
            n_imgs = max_imgs
        assert n_imgs <= max_imgs, (n_imgs, max_imgs)

        n_rows = int.from_bytes(f.read(4), byteorder="big")
        n_cols = int.from_bytes(f.read(4), byteorder="big")

        imgs = f.read(n_imgs * n_rows * n_cols)

    n_channels = 1
    imgs = torch.frombuffer(imgs, dtype=torch.uint8).view(n_imgs, n_channels, n_rows, n_cols)
    imgs = imgs - TRAIN_PIXEL_MEAN
    imgs.mul_(1.0 / TRAIN_PIXEL_STDDEV)
    print("MNIST whitened mean, std", torch.mean(imgs), torch.std(imgs))
    return imgs


def load_data(n_train, n_val, device):
    """Returns (train_imgs, train_targets), (validation_imgs, validation_targets)."""
    targets = targets_from_file("data/train-labels.idx1-ubyte", n_train + n_val).to(device)
    imgs = imgs_from_file("data/train-images.idx3-ubyte", n_train + n_val).to(device)
    return (imgs[:n_train], targets[:n_train]), (imgs[n_train:], targets[n_train:])


if __name__ == "__main__":
    targets = targets_from_file("data/train-labels.idx1-ubyte")
    imgs = imgs_from_file("data/train-images.idx3-ubyte", 1)

    print(f"n_imgs: {len(targets)}, targets:", targets)
    classes, n_with_class = torch.unique(targets, return_counts=True)
    print("classes:", classes, n_with_class / float(len(targets)))

    plt.imshow(imgs[0][0], cmap="gray")
    plt.show()
