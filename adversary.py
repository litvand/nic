import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# import art_example
import mnist
import model
from eval import print_accuracy

N_CLASSES = 10


def fgsm_detector_data(data_pair, trained_model, eps):
    '''
    Generate data for training FGSM detector.

    data_pair: Original dataset images and labels
    trained_model: Model to classify original dataset
    eps: FGSM eps to use when generating new dataset

    returns: detector_imgs, detector_labels
             Detector label is 1 if the image is adversarial and 0 otherwise.
    '''

    imgs, labels = data_pair
    detector_imgs = imgs.clone()
    n_adv = len(imgs)//2  # Number of images to adversarially modify

    fgsm_(detector_imgs[:n_adv], labels[:n_adv], trained_model, eps)
    detector_labels = torch.zeros(len(imgs), dtype=torch.uint8, device=labels.device)
    torch.fill_(detector_labels[:n_adv], 1)  # Label these images as modified by FGSM
    perm = torch.randperm(len(imgs))  # Don't have all adversarial images at the start
    detector_imgs = detector_imgs[perm]
    detector_labels = detector_labels[perm]

    return detector_imgs, detector_labels


def fgsm_(imgs, labels, trained_model, eps, target_class=None):
    imgs.requires_grad = True
    imgs.grad = None

    # Freeze model
    required_grad = []
    for p in trained_model.parameters():
        required_grad.append(p.requires_grad)
        p.requires_grad = False

    chunk_size = 2000  # Choose maximum size that fits in GPU memory
    for i_first in range(0, len(imgs), chunk_size):
        outputs = trained_model(imgs[i_first:i_first + chunk_size])

        if target_class is None:
            # Untargeted adversary: Make output differ from the correct label.
            loss = F.cross_entropy(outputs, labels[i_first:i_first + chunk_size])
        else:
            # Targeted adversary: Make output equal to the target class.
            output_prs = F.softmax(outputs, dim=1)
            # Maximize probability of the target class.
            loss = torch.mean(output_prs[:, target_class])

        loss.backward()

    with torch.no_grad():
        imgs += eps * imgs.grad.sign()

    # Unfreeze model if it wasn't frozen before
    for i, p in enumerate(trained_model.parameters()):
        p.requires_grad = required_grad[i]

    imgs.requires_grad = False
    imgs.grad = None


def cmp_targeted(imgs, labels, trained_model, eps):
    """Compare accuracies with different target classes. Accuracy with an
    untargeted adversary should be lower than accuracy with any target class."""
    for c in range(N_CLASSES):
        targeted_imgs = imgs.clone()
        fgsm_(targeted_imgs, labels, trained_model, eps, target_class=c)
        print_accuracy(f"{c} targeted accuracy", trained_model(targeted_imgs), labels)


def cmp_single(i_img, imgs, labels, trained_model, eps):
    """Compare a single image with its adversarially modified version."""

    adv_img = imgs[i_img].clone()
    fgsm_(
        adv_img.unsqueeze(0),
        labels[i_img].unsqueeze(0),
        trained_model,
        eps,
        target_class=None,
    )

    original_class = torch.argmax(trained_model(imgs[i_img].unsqueeze(0)), 1).item()
    adv_class = torch.argmax(trained_model(adv_img.unsqueeze(0)), 1).item()
    print(f"Label {labels[i_img].item()}, original {original_class}, adversarial" f" {adv_class}")

    plt.imshow(imgs[i_img][0].cpu(), cmap="gray")
    plt.subplots()
    plt.imshow(adv_img[0].cpu(), cmap="gray")
    plt.show()


# n_train=58000, n_valid=2000, art_example.Net(), n_epochs=6, eps=0.2 gives:
# Original accuracy 0.98
# Untargeted accuracy 0.524

# art_example with same parameters:
# Accuracy on benign test examples: 98.45%
# Accuracy on adversarial test examples: 54.1%

# I guess pixel scaling is slightly different somewhere between ART's example and this repo.

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, (imgs, labels) = mnist.load_data(n_train=58000, n_valid=2000, device=device)

    m = model.PoolNet(imgs[0]).to(device)
    model.load(m, "pool-bnorm-58k-f656fe761c9714c8ea6ac237d4b9d6a1ccb683c8.pt")
    m.eval()
    print_accuracy("Original accuracy", m(imgs), labels)

    eps = 0.2
    untargeted_imgs = imgs.clone()
    fgsm_(untargeted_imgs, labels, m, eps)
    print_accuracy("Untargeted accuracy", m(untargeted_imgs), labels)

    # cmp_targeted(imgs, labels, m, eps)
    # cmp_single(-1, imgs, labels, m, eps)

    detector = model.Detector(imgs[0]).to(device)
    model.load(detector, "detect6-9f9acdd513fd4da6920d657a85c392f01297daca.pt")
    detector.eval()

    with torch.no_grad():
        pr_original_adv = torch.mean(F.softmax(detector(imgs), dim=1)[:, 1])
        pr_adv_adv = torch.mean(F.softmax(detector(imgs), dim=1)[:, 1])

    print("Predicted probability that original images are adversarial:", pr_original_adv)
    print("Predicted probability that adversarial images are adversarial:", pr_adv_adv)
