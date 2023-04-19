import torch
from torch import nn


N_CLASSES = 10


class Model(nn.Module):
    def __init__(self, example_img):
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(example_img.numel(), N_CLASSES)
        )
        #self.conv1 = nn.Conv2d(in_channels=example_img.size[0], out_channels=)


    def forward(self, img_batch):
        return self.fc(img_batch)

