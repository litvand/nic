import git
import torch
from torch import nn

import mnist
import model
from adversary import fgsm_detector_data
from eval import print_accuracy
from lsuv_init import LSUV_


def train_model(m, data, name):
    m.train()
    (train_imgs, train_labels), (valid_imgs, valid_labels) = data

    batch_size = 64
    n_epochs = 6
    optimizer = torch.optim.Adam(m.parameters(), lr=1e-2, weight_decay=1e-9)

    for iter in range(1 + n_epochs * len(train_imgs) // batch_size):
        indices = torch.randint(high=len(train_imgs), size=(batch_size,))
        batch_imgs = train_imgs[indices]
        batch_labels = train_labels[indices]

        batch_outputs = m(batch_imgs)
        loss = nn.functional.cross_entropy(batch_outputs, batch_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if iter % 500 == 0:
            with torch.no_grad():
                m.eval()
                print(f"Iter {iter} \tloss {loss.item()}")
                print_accuracy("Train batch accuracy", batch_outputs, batch_labels)
                valid_outputs = m(valid_imgs)
                print_accuracy("Validation accuracy", valid_outputs, valid_labels)
                print(
                    "Validation loss",
                    nn.functional.cross_entropy(valid_outputs, valid_labels),
                )
                model.save(m, name)
                m.train()

    m.eval()


def git_commit():
    repo = git.Repo()
    if repo.active_branch.name == "main":  # Don't clutter the main branch.
        print("NOTE: No automated commit, because on main branch")
        return

    repo.git.add(".")
    repo.git.commit("-m", "_Automated commit")


if __name__ == "__main__":
    git_commit()
    torch.manual_seed(98765)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = mnist.load_data(n_train=58000, n_valid=2000, device=device)
    train_imgs = data[0][0]

    m = model.PoolNet(train_imgs[0]).to(device)
    model.load(m, 'pool-bnorm-58k-f656fe761c9714c8ea6ac237d4b9d6a1ccb683c8.pt')

    if False:
        LSUV_(m, train_imgs[:2000])
        train_model(m, data, 'pool')
    else:
        detector_data = (
            fgsm_detector_data(data[0], m, eps=0.2),
            fgsm_detector_data(data[1], m, eps=0.2)
        )
        detector = model.Detector(detector_data[0][0][0]).to(device)
        train_model(detector, detector_data, 'detect')
