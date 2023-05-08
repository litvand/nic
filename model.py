import git
import torch
from torch import nn

N_CLASSES = 10


class FullyConnected(nn.Module):
    def __init__(self, example_img):
        super().__init__()
        n_hidden = 800
        self.seq = nn.Sequential(
            nn.Flatten(),
            nn.Linear(example_img.numel(), n_hidden),
            nn.GELU(),
            nn.BatchNorm1d(n_hidden),
            nn.Linear(n_hidden, N_CLASSES),
        )

    def forward(self, img_batch):
        return self.seq(img_batch)


class PoolNet(nn.Module):
    # http://yann.lecun.com/exdb/publis/pdf/ranzato-cvpr-07.pdf
    def __init__(self, example_img):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(
                in_channels=example_img.size()[0],
                out_channels=50,
                kernel_size=7,
                padding=2,
            ),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(50),
            nn.Conv2d(in_channels=50, out_channels=128, kernel_size=7),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(128),
            nn.Flatten(),
        )

        with torch.no_grad():
            n_fc_inputs = self.convs(example_img.cpu().unsqueeze(0)).numel()

        print("Number of input features to fully connected layers:", n_fc_inputs)
        self.fully_connected = nn.Sequential(
            nn.Linear(n_fc_inputs, 200),
            nn.BatchNorm1d(200),
            nn.GELU(),
            nn.Linear(200, N_CLASSES),
        )

    def forward(self, img_batch):
        return self.fully_connected(self.convs(img_batch))


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


def chunked_forward(model, imgs, chunk_size=100):
    '''Split images into chunks of chunk_size images and evaluate one chunk at a time'''

    chunk_outputs = model(imgs[:chunk_size])  # Find out output shape
    outputs = torch.empty(len(imgs), *chunk_outputs.size()[1:], device='cpu', requires_grad=True)
    outputs[:chunk_size] = chunk_outputs.cpu()

    for i_first in range(chunk_size, len(imgs), chunk_size):
        torch.cuda.empty_cache()
        i_end = min(i_first + chunk_size, len(imgs))
        outputs[i_first:i_end] = model(imgs[i_first:i_end]).cpu()

    return outputs


def save(model, model_name):
    # Can look at exact parameters and data that the model was trained on using the commit.
    last_commit = git.Repo().head.object.hexsha
    torch.save(model.state_dict(), f"models/{model_name}-{last_commit}.pt")


def load(model, filename):
    model.load_state_dict(torch.load(f"models/{filename}"))
