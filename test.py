import torch

import mnist
import model


def print_accuracy(msg, outputs, labels):
    output_classes = torch.argmax(outputs, 1) # Convert logits to class indices.
    accuracy = torch.sum(output_classes == labels) / float(len(labels))
    print(msg, round(accuracy.item(), 3))


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _, (valid_imgs, valid_labels) = mnist.load_data(n_train=10000, n_valid=2000, device=device)
    m = model.Model(valid_imgs[0]).to(device)
    model.load(m, 'fc-7922c1dad3adddb9eabbee32e378ea3e1bd2bcd2.pt')
    m.eval()
    print_accuracy("Validation accuracy", m(valid_imgs), valid_labels)


