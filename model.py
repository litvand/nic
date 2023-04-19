import git
import torch
from torch import nn


N_CLASSES = 10


class Model(nn.Module):
    def __init__(self, example_img):
        super().__init__()
        n_hidden = 800
        self.seq = nn.Sequential(
            nn.Flatten(),
            nn.Linear(example_img.numel(), n_hidden),
            nn.GELU(),
            nn.Linear(n_hidden, N_CLASSES)
        )


    def forward(self, img_batch):
        return self.seq(img_batch)


def save(model, model_name):
    # Can look at exact parameters and data that the model was trained on using the commit.
    last_commit = git.Repo().head.object.hexsha
    torch.save(model.state_dict(), f'models/{model_name}-{last_commit}.pt')


def load(model, filename):
    model.load_state_dict(torch.load(f'models/{filename}'))

