import git
import torch
from lsuv_init import LSUV_

import mnist
import model
from test import print_accuracy


def train_model(m, data):
    m.train()
    (train_imgs, train_labels), (valid_imgs, valid_labels) = data

    batch_size = 300
    n_iter = 50000
    optimizer = torch.optim.Adam(m.parameters(), lr=0.0003)
    
    for iter in range(n_iter):
        indices = torch.randint(high=len(train_imgs), size=(batch_size,))
        batch_imgs = train_imgs[indices]
        batch_labels = train_labels[indices]

        batch_outputs = m(batch_imgs)
        loss = torch.nn.functional.cross_entropy(batch_outputs, batch_labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if iter % 400 == 0:
            with torch.no_grad():
                m.eval()
                print(f"Iter {iter} \tloss {loss.item()}")
                print_accuracy("Train batch accuracy", batch_outputs, batch_labels)
                print_accuracy("Valid accuracy", m(valid_imgs), valid_labels)
                model.save(m, 'fc')
                m.train()

    m.eval()


def git_commit():
    repo = git.Repo()
    if repo.active_branch.name == 'master': # Don't clutter the master branch.
        print('NOTE: No automated commit, because on master branch')
        return
    
    repo.git.add('.')
    repo.git.commit('-m', '_Automated commit') 


if __name__ == '__main__':
    git_commit()
    torch.manual_seed(412790)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data = mnist.load_data(n_train=20000, n_valid=2000, device=device)
    train_imgs = data[0][0]
    
    m = model.PoolNet(train_imgs[0]).to(device)
    LSUV_(m, train_imgs[:min(2000, len(train_imgs))])
    train_model(m, data)
 
 