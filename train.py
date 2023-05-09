import git
import torch
import torch.nn.functional as F

import mnist
import model
from adversary import fgsm_detector_data
from eval import print_accuracy
from lsuv_init import LSUV_


def train_model(m, data, name):
    m.train()
    (train_imgs, train_labels), (valid_imgs, valid_labels) = data
    min_valid_loss = float("inf")

    batch_size = 150
    n_epochs = 50
    optimizer = torch.optim.Adam(m.parameters(), lr=1e-2, weight_decay=1e-9)

    for i_img in range(0, n_epochs * len(train_imgs), batch_size):
        indices = torch.randint(high=len(train_imgs), size=(batch_size,))
        batch_imgs = train_imgs[indices]
        batch_labels = train_labels[indices]

        batch_outputs = m(batch_imgs)
        loss = F.cross_entropy(batch_outputs, batch_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i_img % 20000 < batch_size:  # <batch_size instead of ==0 because might not be exactly 0
            with torch.no_grad():
                print(f"{i_img//1000}k images processed, batch loss {loss.item()}")
                print_accuracy("Training batch accuracy", batch_outputs, batch_labels)

                m.eval()
                valid_outputs = m(valid_imgs)
                valid_loss = F.cross_entropy(valid_outputs, valid_labels).item()
                print("Validation loss", valid_loss)
                print_accuracy("Validation accuracy", valid_outputs, valid_labels)
                if valid_loss < min_valid_loss:
                    if min_valid_loss < float("inf"):
                        # Don't overwrite saved model if loss was just < infinity.
                        model.save(m, name)
                    min_valid_loss = valid_loss

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

    data = mnist.load_data(n_train=20000, n_valid=2000, device=device)
    train_imgs = data[0][0]

    m = model.PoolNet(train_imgs[0]).to(device)
    # LSUV_(m, train_imgs[:2000])
    model.load(m, "pool20k-18dab86434e82bce7472c09da5f82864a6424e86.pt")
    # train_model(m, data, 'pool20k')

    detector_data = (
        fgsm_detector_data(data[0], m, eps=0.2),
        fgsm_detector_data(data[1], m, eps=0.2),
    )
    detector = model.Detector(detector_data[0][0][0]).to(device)
    LSUV_(detector, detector_data[0][0][:2000])
    # model.load(detector, 'detect5-9f9acdd513fd4da6920d657a85c392f01297daca.pt')
    train_model(detector, detector_data, "detect")
