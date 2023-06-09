import torch
from torch import nn

import adversary
import train


class Detector(nn.Module):
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

    def forward(self, img_batch):
        return self.seq(img_batch)


def prs_adv(detector, imgs):
    """Returns probability of being adversarial for each image"""
    return F.softmax(detector(imgs), dim=1)[:, 1]


def classify_adv(detector, imgs, threshold):
    """Classifies images as adversarial or not based on the threshold"""
    return prs_adv(detector, imgs) > threshold


def fgsm_detector_data(data_pair, trained_model, eps):
    """
    Generate data for training FGSM detector.

    data_pair: Original dataset images and targets
    trained_model: Model to classify original dataset
    eps: FGSM epsilon to use when generating new dataset

    returns: detector_imgs, detector_targets
             Detector target is 1 if the image is adversarial and 0 otherwise.
    """

    imgs, targets = data_pair
    detector_imgs = imgs.clone()
    n_adv = len(imgs) // 2  # Number of images to adversarially modify

    fgsm_(detector_imgs[:n_adv], targets[:n_adv], trained_model, eps)
    detector_targets = torch.zeros(len(imgs), dtype=torch.uint8, device=targets.device)
    torch.fill_(detector_targets[:n_adv], 1)  # target these images as modified by FGSM
    perm = torch.randperm(len(imgs))  # Don't have all adversarial images at the start
    detector_imgs = detector_imgs[perm]
    detector_targets = detector_targets[perm]

    return detector_imgs, detector_targets


def train_detector(trained_model, data, load_name=None, eps=0.2):
    detector_data = (
        fgsm_detector_data(data[0], trained_model, eps),
        fgsm_detector_data(data[1], trained_model, eps),
    )
    detector = model.Detector(detector_data[0][0][0]).to(data[0][0].device)
    if load_name is not None:
        model.load(detector, load_name)
    else:
        LSUV_(detector, detector_data[0][0][:2000])
    logistic_regression(detector, detector_data, "detect")
