import matplotlib.pyplot as plt
import torch
from torch import nn

import art_example
import mnist
import model
from test import print_accuracy


N_CLASSES = 10


def fgsm_(imgs, labels, trained_model, eps, target_class=None):
    imgs.requires_grad = True
    imgs.grad = None

    # Freeze model
    required_grad = []
    for p in trained_model.parameters():
        required_grad.append(p.requires_grad)
        p.requires_grad = False

    outputs = trained_model(imgs)
    
    if target_class is None: # Untargeted adversary: Make output differ from the correct label.
        loss = nn.functional.cross_entropy(outputs, labels)

    else: # Targeted adversary: Make output equal to the target class.
        output_prs = nn.functional.softmax(outputs, dim=1)
        loss = torch.mean(output_prs[:, target_class])

    print('FGSM loss', loss.item())
    loss.backward()

    with torch.no_grad():
        imgs += eps * imgs.grad.sign()
    
    # Unfreeze model if it wasn't frozen before
    for i, p in enumerate(trained_model.parameters()):
        p.requires_grad = required_grad[i]
    
    imgs.requires_grad = False
    imgs.grad = None


def cmp_targeted(imgs, labels, trained_model, eps):
    '''Compare accuracies with different target classes. Accuracy with an
    untargeted adversary should be lower than accuracy with any target class.'''
    for c in range(N_CLASSES):
        targeted_imgs = imgs.clone()
        fgsm_(targeted_imgs, labels, trained_model, eps, target_class=c)
        print_accuracy(f'{c} targeted accuracy', trained_model(targeted_imgs), labels)


def cmp_single(i_img, imgs, labels, trained_model, eps):
    '''Compare a single image with its adversarially modified version.'''

    adv_img = imgs[i_img].clone()
    fgsm_(
        adv_img.unsqueeze(0),
        labels[i_img].unsqueeze(0),
        trained_model,
        eps,
        target_class=None
    )

    original_class = torch.argmax(trained_model(imgs[i_img].unsqueeze(0)), 1).item()
    adv_class = torch.argmax(trained_model(adv_img.unsqueeze(0)), 1).item()
    print(f'Label {labels[i_img].item()}, original {original_class}, adversarial {adv_class}')
    
    plt.imshow(imgs[i_img][0].cpu(), cmap='gray')
    plt.subplots()
    plt.imshow(adv_img[0].cpu(), cmap='gray')
    plt.show()


# n_train=58000, n_valid=2000, art_example.Net(), n_epochs=6, eps=0.07 gives:
# Original accuracy 0.984
# Untargeted accuracy 0.531

# art_example with same parameters except for eps=0.2:
# Accuracy on benign test examples: 98.45%
# Accuracy on adversarial test examples: 54.1%

# I guess pixel scaling is different somewhere between the example and this repo by
# approximately a factor of 3, since the accuracy on benign examples is the same.

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _, (imgs, labels) = mnist.load_data(n_train=58000, n_valid=2000, device=device) 
    
    m = art_example.Net().to(device)
    model.load(m, 'art-d0ef50afe3c1a6cb47b069d3c42a6c832f3c1ed9.pt')
    m.eval()
    print_accuracy('Original accuracy', m(imgs), labels)
    
    eps = 0.07
    untargeted_imgs = imgs.clone()
    fgsm_(untargeted_imgs, labels, m, eps)
    print_accuracy('Untargeted accuracy', m(untargeted_imgs), labels)
    
    #cmp_targeted(imgs, labels, m, eps)
    cmp_single(-1, imgs, labels, m, eps)
    

