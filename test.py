import torch

import mnist
import model


def print_accuracy(msg, outputs, labels):
    output_classes = torch.argmax(outputs, 1) # Convert logits to class indices.
    accuracy = torch.sum(output_classes == labels) / float(len(labels))
    print(msg, round(accuracy.item(), 3))


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _, (valid_imgs, valid_labels) = mnist.load_data(n_train=20000, n_valid=5000, device=device)
    m = model.PoolNet(valid_imgs[0]).to(device)
    model.load(m, 'pool-bnorm-20k-faeca80e1ca4e0d35fe14157fdb4f02183d3d3cd.pt')
    m.eval()
    print_accuracy("Validation accuracy", m(valid_imgs[:10000]), valid_labels[:10000])


# FullyConnected, 50k training images, 0.984 max validation accuracy
# PoolNet BatchNorm, 20k training images, 0.99 max validation accuracy, 0.988 min validation accuracy


