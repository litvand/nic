import matplotlib.pyplot as plt
import torch
from torch import nn

import mnist
import model
from test import print_accuracy


N_CLASSES = 10


# max_dist=10, norm=2 or max_dist=2, norm=float('inf') gives
# PoolNet accuracy <10% on modified training images.


def fgsm_(imgs, labels, trained_model, max_dist, norm, target_class=None):
    '''`max_dist` is the distance (according to the given norm) by which `imgs` will be modified.'''

    imgs.requires_grad = True
    imgs.grad = None

    # Freeze model
    required_grad = []
    for p in trained_model.parameters():
        required_grad.append(p.requires_grad)
        p.requires_grad = False

    outputs = trained_model(imgs)
    output_prs = nn.functional.softmax(outputs, dim=1)
    
    if target_class is None: # Untargeted adversary: Make output differ from the correct label.
        #loss = nn.functional.cross_entropy(outputs, labels) # Harder to interpret loss

        flat_indices = torch.arange(len(labels)).cuda()
        flat_indices.mul_(len(outputs[0]))
        flat_indices += labels # Convert labels to indices into flattened output tensor.

        correct_label_prs = output_prs.view(-1)[flat_indices]
        loss = -torch.mean(correct_label_prs)

    else: # Targeted adversary: Make output equal to the target class.
        loss = torch.mean(output_prs[:, target_class])

    print('FGSM loss', loss.item())
    loss.backward()

    with torch.no_grad():
        flat_grads = imgs.grad.view(len(imgs), -1)
        dist_each_img = torch.linalg.vector_norm(flat_grads, ord=norm, dim=1)

        # Scale gradients to max_dist
        float_error = 1e-15
        flat_grads.mul_((max_dist / (dist_each_img + float_error)).unsqueeze(1))
        assert torch.all(flat_grads <= max_dist), (flat_grads, dist_each_img, max_dist)
        
        # Add gradients to maximize loss.
        imgs += imgs.grad
    
    # Unfreeze model if it wasn't frozen before
    for i, p in enumerate(trained_model.parameters()):
        p.requires_grad = required_grad[i]
    
    imgs.requires_grad = False
    imgs.grad = None


def cmp_targeted(imgs, labels, trained_model, max_dist, norm):
    '''Compare accuracies with different target classes. Accuracy with an
    untargeted adversary should be lower than accuracy with any target class.'''
    for c in range(N_CLASSES):
        targeted_imgs = imgs.clone()
        fgsm_(targeted_imgs, labels, trained_model, max_dist, norm, target_class=c)
        print_accuracy(f'{c} targeted accuracy', trained_model(targeted_imgs), labels)


def cmp_single(i_img, imgs, labels, trained_model, max_dist, norm):
    '''Compare a single image with its adversarially modified version.'''

    adv_img = imgs[i_img].clone()
    fgsm_(
        adv_img.unsqueeze(0),
        labels[i_img].unsqueeze(0),
        trained_model,
        max_dist,
        norm,
        target_class=None
    )

    original_class = torch.argmax(trained_model(imgs[i_img].unsqueeze(0)), 1).item()
    adv_class = torch.argmax(trained_model(adv_img.unsqueeze(0)), 1).item()
    print(f'Label {labels[i_img].item()}, original {original_class}, adversarial {adv_class}')
    
    plt.imshow(imgs[i_img][0].cpu(), cmap='gray')
    plt.subplots()
    plt.imshow(adv_img[0].cpu(), cmap='gray')
    plt.show()


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    (train_imgs, train_labels), _ = mnist.load_data(n_train=1000, n_valid=0, device=device) 
    
    m = model.PoolNet(train_imgs[0]).to(device)
    model.load(m, 'pool-bnorm-20k-faeca80e1ca4e0d35fe14157fdb4f02183d3d3cd.pt')
    m.eval()

    print_accuracy('Original accuracy', m(train_imgs), train_labels)
    
    max_dist = 10.0
    norm = 2

    untargeted_imgs = train_imgs.clone()
    fgsm_(untargeted_imgs, train_labels, m, max_dist, norm)
    print_accuracy('Untargeted accuracy', m(untargeted_imgs), train_labels)
    
    cmp_targeted(train_imgs, train_labels, m, max_dist, norm)
    cmp_single(-1, train_imgs, train_labels, m, max_dist, norm)
    

