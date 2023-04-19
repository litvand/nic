import matplotlib.pyplot as plt
import torch


TRAIN_PIXEL_MEAN = 33.0
TRAIN_PIXEL_STDDEV = 79.0


def labels_from_file(filename, n_imgs=-1):
    with open(filename, 'rb') as f:
        magic = int.from_bytes(f.read(4), byteorder='big')
        assert magic == 2049, magic

        max_imgs = int.from_bytes(f.read(4), byteorder='big')
        if n_imgs < 0:
            n_imgs = max_imgs
        assert n_imgs <= max_imgs, (n_imgs, max_imgs)

        labels = f.read(n_imgs)
    
    return torch.frombuffer(labels, dtype=torch.uint8).clone()


def imgs_from_file(filename, n_imgs=-1):
    with open(filename, 'rb') as f:
        magic = int.from_bytes(f.read(4), byteorder='big')
        assert magic == 2051, magic

        max_imgs = int.from_bytes(f.read(4), byteorder='big')
        if n_imgs < 0:
            n_imgs = max_imgs
        assert n_imgs <= max_imgs, (n_imgs, max_imgs)

        n_rows = int.from_bytes(f.read(4), byteorder='big')
        n_cols = int.from_bytes(f.read(4), byteorder='big')

        imgs = f.read(n_imgs * n_rows * n_cols)
    
    n_channels = 1
    imgs = torch.frombuffer(imgs, dtype=torch.uint8).view(n_imgs, n_channels, n_rows, n_cols)
    imgs = imgs - TRAIN_PIXEL_MEAN # Whiten
    imgs.mul_(1.0 / TRAIN_PIXEL_STDDEV)
    #print(torch.mean(imgs), torch.std(imgs))
    return imgs


def load_data(n_train, n_valid, device):
    '''Returns (train_imgs, train_labels), (validation_imgs, validation_labels).'''
    labels = labels_from_file('data/train-labels.idx1-ubyte', n_train + n_valid).to(device)
    imgs = imgs_from_file('data/train-images.idx3-ubyte', n_train + n_valid).to(device)
    return (imgs[:n_train], labels[:n_train]), (imgs[n_train:], labels[n_train:])


if __name__ == '__main__':
    labels = labels_from_file('data/train-labels.idx1-ubyte')
    imgs = imgs_from_file('data/train-images.idx3-ubyte', 1)

    print(f'n_imgs: {len(labels)}, labels:', labels)
    classes, n_with_class = torch.unique(labels, return_counts=True)
    print('classes:', classes, n_with_class / float(len(labels)))

    plt.imshow(imgs[0][0], cmap='gray')
    plt.show()

