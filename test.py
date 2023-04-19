import torch

import mnist
import model


def print_accuracy(msg, outputs, labels):
    output_classes = torch.argmax(outputs, 1) # Convert logits to class indices.
    accuracy = torch.sum(output_classes == labels) / float(len(labels))
    print(msg, round(accuracy.item(), 3))


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _, (valid_imgs, valid_labels) = mnist.load_data(n_train=20000, n_valid=10000, device=device)
    m = model.PoolNet(valid_imgs[0]).to(device)
    model.load(m, 'pool-8315f5fc0646ced32418bc011f52a47c121edfcc.pt')
    m.eval()
    print_accuracy("Validation accuracy", m(valid_imgs[:10000]), valid_labels[:10000])


