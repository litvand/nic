import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

import classifier
import eval
import mnist
import train


def fgsm_detector_data(imgs, targets, trained_model, eps):
    """
    Generate data for training/evaluating FGSM detector.

    imgs: Original dataset images (won't be modified)
    targets: Original dataset targets as class indices
    trained_model: Model trained to classify the original dataset
    eps: FGSM epsilon to use when generating the new dataset

    Returns: detector_imgs, detector_targets
             Detector target is 0 if the image is adversarial and 1 otherwise.
    """

    n_imgs = len(imgs)
    n_adv = n_imgs // 2  # Number of images to adversarially modify

    detector_imgs = imgs.clone()
    fgsm_(detector_imgs[:n_adv], targets[:n_adv], trained_model, eps)

    detector_targets = torch.zeros_like(targets)
    detector_targets[n_adv:].fill_(1)

    perm = torch.randperm(n_imgs)  # Don't have all adversarial images at the start
    return detector_imgs[perm], detector_targets[perm]


def fgsm_(imgs, targets, trained_model, eps, aim_class=None):
    """
    imgs: Float tensor with size (n_images, n_channels, height, width)
    targets: 1d int tensor with each image's class index
    trained_model: Outputs target logits
    eps: Modification L1 norm
    aim_class: Class to maximize probability of; if `None`, just minimizes probability of the
               correct class.
    """

    imgs.requires_grad = True
    imgs.grad = None

    # Freeze model
    required_grad = []
    for p in trained_model.parameters():
        required_grad.append(p.requires_grad)
        p.requires_grad = False

    trained_model.eval()

    chunk_size = 2000  # TODO: Choose maximum size that fits in GPU memory
    for i_first in range(0, len(imgs), chunk_size):
        outputs = trained_model(imgs[slice(i_first, i_first + chunk_size)])

        if aim_class is None:
            # Unaimed adversary: Make output differ from the correct target.
            loss = F.cross_entropy(outputs, targets[slice(i_first, i_first + chunk_size)])
        else:
            # Aimed adversary: Make output equal to the aimed class.
            output_prs = F.softmax(outputs, dim=1)
            loss = torch.mean(output_prs[:, aim_class])

        loss.backward()

    with torch.no_grad():
        imgs += eps * imgs.grad.sign()

    # Unfreeze model if it wasn't frozen before
    for i, p in enumerate(trained_model.parameters()):
        p.requires_grad = required_grad[i]

    imgs.requires_grad = False
    imgs.grad = None


def cmp_by_aim(imgs, targets, trained_model, eps):
    """Compare accuracies with different aimed classes. Accuracy with an
    unaimed adversary should be lower than accuracy with any aimed class."""

    with torch.no_grad():
        n_classes = trained_model(imgs[:1]).size(1)

    for c in range(n_classes):
        aim_imgs = imgs.clone()
        fgsm_(aim_imgs, targets, trained_model, eps, aim_class=c)
        with torch.no_grad():
            eval.print_multi_acc(trained_model(aim_imgs), targets, f"{c} aimed accuracy")


def cmp_single(i_img, imgs, targets, trained_model, eps):
    """Compare a single image with its adversarially modified version."""

    adv_img = imgs[i_img].clone()
    fgsm_(
        adv_img.unsqueeze(0),
        targets[i_img].unsqueeze(0),
        trained_model,
        eps,
        aim_class=None,
    )

    with torch.no_grad():
        original_class = trained_model(imgs[i_img].unsqueeze(0)).argmax(1).item()
        adv_class = trained_model(adv_img.unsqueeze(0)).argmax(1).item()
        print(f"target {targets[i_img].item()}, original {original_class}, adversarial {adv_class}")

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
    _, (imgs, targets) = mnist.load_data(n_train=22000, n_val=5000, device=device)

    net = classifier.PoolNet(imgs[0]).to(device)
    train.load(net, "pool-norestart20k-2223b6b48b3680297dda4cb0f644d39268753dca")
    net.eval()
    with torch.no_grad():
        eval.print_multi_acc(net(imgs), targets, "Original")

    eps = 0.2
    unaimed_imgs = imgs.clone()
    fgsm_(unaimed_imgs, targets, net, eps)
    with torch.no_grad():
        eval.print_multi_acc(net(unaimed_imgs), targets, f"Unaimed (eps={eps})")

    # cmp_by_aim(imgs, targets, net, eps)
    cmp_single(-1, imgs, targets, net, eps)
