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
            adversary.fgsm_detector_data(*data[0], trained_model, eps),
            adversary.fgsm_detector_data(*data[1], trained_model, eps),
        )
        train.logistic_regression(self.seq, detector_data, init=True)
        with torch.no_grad():
            self.threshold = nn.Parameter(
                balanced_acc_threshold(self.prs(detector_data[0][0]), detector_data[0][1] == 1),
                requires_grad=False,
            )

        return self


if __name__ == "__main__":
    torch.manual_seed(98765)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = mnist.load_data(n_train=20000, n_val=2000, device=device)
    example_img = data[0][0][0]

    trained_model = classifier.FullyConnected(example_img).to(device)
    train.load(trained_model, "fc20k-dc84d9b97f194b36c1130a5bc82eda5d69a57ad2")

    detector = DetectorNet(example_img).to(device)  # .fit(data, trained_model, eps=0.2)
    # train.save(detector, "detector-net20k")
    train.load(detector, "detector-net-offc20k-63bc3202b7f53ba1bc0adadfcf906a6f784494a5")

    # detector_val_imgs, detector_val_targets = detector.last_detector_data[1]
    detector_val_imgs, detector_val_targets = adversary.fgsm_detector_data(
        data[1][0], data[1][1], trained_model, 0.2
    )
    with torch.no_grad():
        detector.eval()
        val_prs = detector.prs(detector_val_imgs)
        thresholds = [detector.threshold.item()] + [i / 10 for i in range(1, 10)] + [0.99, 0.999]
        for threshold in thresholds:
            eval.print_bin_acc(
                val_prs - threshold, detector_val_targets == 1, f"Threshold {threshold}"
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

# fc20k-dc84d9b97f194b36c1130a5bc82eda5d69a57ad2
# detector-net-onfc20k-63bc3202b7f53ba1bc0adadfcf906a6f784494a5
# (All nets trained with restarts.)
# Threshold 0.4996219873428345 accuracy: 87.35%
# Threshold 0.4996219873428345 true positives (as fraction of positive targets): 90.6%
# Threshold 0.4996219873428345 true negatives (as fraction of negative targets): 84.1%
# Threshold 0.1 accuracy: 82.6%
# Threshold 0.1 true positives (as fraction of positive targets): 99.0%
# Threshold 0.1 true negatives (as fraction of negative targets): 66.2%
# Threshold 0.2 accuracy: 86.0%
# Threshold 0.2 true positives (as fraction of positive targets): 97.6%
# Threshold 0.2 true negatives (as fraction of negative targets): 74.4%
# Threshold 0.7 accuracy: 86.0%
# Threshold 0.7 true positives (as fraction of positive targets): 82.5%
# Threshold 0.7 true negatives (as fraction of negative targets): 89.5%
# Threshold 0.99 accuracy: 56.3%
# Threshold 0.99 true positives (as fraction of positive targets): 13.5%
# Threshold 0.99 true negatives (as fraction of negative targets): 99.1%
