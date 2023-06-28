import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

import classifier
import eval
import mnist
import train


def fgsm_detector_data(imgs, y, trained_model, eps):
    """
    Generate data for training/evaluating FGSM detector.

    imgs: Original dataset images (won't be modified)
    y: Original dataset labels as class indices
    trained_model: Model trained to classify the original dataset
    eps: FGSM epsilon to use when generating the new dataset

    Returns: detector_imgs, detector_y
             Detector label is 0 if the image is adversarial and 1 otherwise.
    """

    n_imgs = len(imgs)
    n_adv = n_imgs // 2  # Number of images to adversarially modify

    detector_imgs = imgs.clone()
    fgsm_(detector_imgs[:n_adv], y[:n_adv], trained_model, eps)

    detector_labels = torch.zeros_like(y)
    detector_labels[n_adv:].fill_(1)

    perm = torch.randperm(n_imgs)  # Don't have all adversarial images at the start
    return detector_imgs[perm], detector_labels[perm]


def fgsm_(imgs, y, trained_model, eps, target_class=None):
    """
    imgs: Float tensor with size (n_images, n_channels, height, width)
    y: 1d int tensor with each image's class index
    trained_model: Outputs label logits
    eps: Modification L1 norm
    target_class: Class to maximize probability of; if `None`, just minimizes probability of the
                  correct class.
    """

    imgs.grad = None
    imgs.requires_grad = True

    # Freeze model
    required_grad = []
    for p in trained_model.parameters():
        required_grad.append(p.requires_grad)
        p.requires_grad = False

    trained_model.eval()

    chunk_size = 2000  # TODO: Choose maximum size that fits in GPU memory
    for i_first in range(0, len(imgs), chunk_size):
        outputs = trained_model(imgs[i_first : i_first + chunk_size])

        if target_class is None:
            # Untargeted adversary: Make output differ from the correct label.
            loss = F.cross_entropy(outputs, y[i_first : i_first + chunk_size])
        else:
            # Aimed adversary: Make output equal to the targeted class.
            output_prs = F.softmax(outputs, dim=1)
            loss = torch.mean(output_prs[:, target_class])

        loss.backward()

    with torch.no_grad():
        imgs += eps * imgs.grad.sign()

    # Unfreeze model if it wasn't frozen before
    for i, p in enumerate(trained_model.parameters()):
        p.requires_grad = required_grad[i]

    imgs.requires_grad = False
    imgs.grad = None


def cmp_by_target(imgs, y, trained_model, eps):
    """Compare accuracies with different targeted classes. Accuracy with an
    untargeted adversary should be lower than accuracy with any targeted class."""

    with torch.no_grad():
        n_classes = trained_model(imgs[:1]).size(1)

    for c in range(n_classes):
        target_imgs = imgs.clone()
        fgsm_(target_imgs, y, trained_model, eps, target_class=c)
        with torch.no_grad():
            eval.print_multi_acc(trained_model(target_imgs), y, f"{c} targeted accuracy")


def cmp_single(i_img, imgs, y, trained_model, eps):
    """Compare a single image with its adversarially modified version."""

    adv_img = imgs[i_img].clone()
    fgsm_(
        adv_img.unsqueeze(0),
        y[i_img].unsqueeze(0),
        trained_model,
        eps,
        target_class=None,
    )

    with torch.no_grad():
        original_class = trained_model(imgs[i_img].unsqueeze(0)).argmax(1).item()
        adv_class = trained_model(adv_img.unsqueeze(0)).argmax(1).item()
        print(f"label {y[i_img].item()}, original {original_class}, adversarial {adv_class}")

        plt.imshow(imgs[i_img][0].cpu(), cmap="gray")
        plt.subplots()
        plt.imshow(adv_img[0].cpu(), cmap="gray")
        plt.show()


# n_train=58000, n_val=2000, art_example.Net(), n_epochs=6, eps=0.2 gives:
# Original accuracy 0.98
# Untargeted accuracy 0.524

# art_example with same parameters:
# Accuracy on benign test examples: 98.45%
# Accuracy on adversarial test examples: 54.1%

# I guess pixel scaling is slightly different somewhere between ART's example and this repo.

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, (imgs, y) = mnist.load_data(n_train=20000, n_val=2000, device=device)

    net = classifier.PoolNet(imgs[0]).to(device)
    train.load(net, "pool20k-1ce6321a452d629b14cf94ad9266ad584cd36e85")
    net.eval()
    with torch.no_grad():
        eval.print_multi_acc(net(imgs), y, "Original")

    eps = 0.2
    untargeted_imgs = imgs.clone()
    fgsm_(untargeted_imgs, y, net, eps)
    with torch.no_grad():
        eval.print_multi_acc(net(untargeted_imgs), y, f"Untargeted (eps={eps})")

    # cmp_by_target(imgs, y, net, eps)
    cmp_single(-1, imgs, y, net, eps)
