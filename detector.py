import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn

import adversary
import classifier
import eval
import mnist
import train


def balanced_acc_threshold(train_outputs, is_positive_target):
    """
    Estimate the threshold that maximizes balanced accuracy

    train_outputs: 1d tensor; larger output --> should be positive target on average
    is_positive_target: 1d bool tensor
    """

    pos_target_min = train_outputs[is_positive_target].min()
    neg_target_max = train_outputs[~is_positive_target].max()
    return (pos_target_min + neg_target_max) / 2


class DetectorNet(nn.Module):
    def __init__(self, example_img):
        super().__init__()
        n_hidden = 800
        self.seq = nn.Sequential(
            nn.Flatten(),
            nn.Linear(example_img.numel(), n_hidden),
            nn.GELU(),
            nn.BatchNorm1d(n_hidden),
            nn.Linear(n_hidden, 2),
        )
        self.threshold = nn.Parameter(torch.tensor(torch.nan), requires_grad=False)
        self.last_detector_data = None

    def forward(self, img_batch):
        """Positive --> original image, negative --> adversarial image"""
        return self.prs(img_batch) - self.threshold

    def prs(self, img_batch):
        """Returns probability of being an original image (i.e. non-adversarial) for each image"""
        return F.softmax(self.seq(img_batch), dim=1)[:, 1]

    def fit(self, data, trained_model, eps):
        """
        data: Original data ((train_imgs, train_targets), (val_imgs, val_targets))
        trained_model: Model trained to classify the original data
        eps: FGSM epsilon to use when generating new data
        """

        self.last_detector_data = detector_data = (
            fgsm_detector_data(*data[0], trained_model, eps),
            fgsm_detector_data(*data[1], trained_model, eps),
        )
        train.logistic_regression(self.seq, detector_data, init=True)
        with torch.no_grad():
            self.threshold = nn.Parameter(
                balanced_acc_threshold(self.prs(detector_data[0][0]), detector_data[0][1] == 1),
                requires_grad=False
            )

        return self


def fgsm_detector_data(imgs, targets, trained_model, eps):
    """
    Generate data for training/evaluating FGSM detector.

    imgs: Original dataset images
    targets: Original dataset targets as class indices
    trained_model: Model trained to classify the original dataset
    eps: FGSM epsilon to use when generating the new dataset

    Returns: detector_imgs, detector_targets
             Detector target is 0 if the image is adversarial and 1 otherwise.
    """

    n_imgs = len(imgs)
    n_adv = n_imgs // 2  # Number of images to adversarially modify

    detector_imgs = imgs.clone()
    adversary.fgsm_(detector_imgs[:n_adv], targets[:n_adv], trained_model, eps)

    detector_targets = torch.zeros_like(targets)
    detector_targets[n_adv:].fill_(1)

    perm = torch.randperm(n_imgs)  # Don't have all adversarial images at the start
    return detector_imgs[perm], detector_targets[perm]


if __name__ == "__main__":
    train.git_commit()
    torch.manual_seed(98765)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = mnist.load_data(n_train=20000, n_val=2000, device=device)
    example_img = data[0][0][0]

    trained_model = classifier.FullyConnected(example_img).to(device)
    train.load(trained_model, "fc20k-dc84d9b97f194b36c1130a5bc82eda5d69a57ad2.pt")

    detector = DetectorNet(example_img).to(device)  # .fit(data, trained_model, eps=0.2)
    # train.save(detector, "detector-net20k")
    # detector_val_imgs, detector_val_targets = detector.last_detector_data[1]
    train.load(detector, "detector-net-offc20k-63bc3202b7f53ba1bc0adadfcf906a6f784494a5.pt")
    detector_val_imgs, detector_val_targets = fgsm_detector_data(
        data[1][0], data[1][1], trained_model, 0.2
    )

    with torch.no_grad():
        detector.eval()
        val_prs = detector.prs(detector_val_imgs)
        thresholds = [detector.threshold.item()] + [i/10 for i in range(1, 10)] + [0.99, 0.999]
        for threshold in thresholds:
            eval.print_bin_acc(
                val_prs - threshold,
                detector_val_targets == 1,
                f"Threshold {threshold}"
            )

    # eval.plot_distr_overlap(
    #     prs_on_adv,
    #     prs_on_original,
    #     "Detector net on validation adversarial",
    #     "original images"
    # )
    plt.show()

# Fully connected detector taking just the raw image as input can detect 90% of adversarial images
# while classifying 90% of normal images correctly, or detect 50% of adversarial images while
# classifying 99.5% of normal images correctly. Detecting 99% of adversarial images would mean
# classifying only 3% of normal images correctly.
