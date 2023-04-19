import torch
import git

import mnist
import model


def print_accuracy(msg, outputs, labels):
    output_classes = torch.argmax(outputs, 1)
    print(msg, round(torch.mean(output_classes == labels).item(), 3))


def train_model(model, data):
    model.train()
    (train_imgs, train_labels), (valid_imgs, valid_labels) = data

    batch_size = 1000
    n_iter = 1000
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for iter in range(n_iter):
        indices = torch.randint(high=len(train_imgs), size=(batch_size,))
        batch_imgs = train_imgs[indices]
        batch_labels = train_labels[indices]

        batch_outputs = model(batch_imgs)
        loss = torch.nn.functional.cross_entropy(batch_outputs, batch_labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if iter % 100 == 0:
            with torch.no_grad():
                print(f"Iter {iter} \tloss {loss.item()}")
                print_accuracy("Train batch accuracy", batch_outputs, batch_labels)

    model.eval()


def save_model(model, name):
    # Can look at exact parameters and data that the model was trained on using the commit.
    last_commit = git.Repo().head.object.hexsha
    torch.save(model.state_dict(), f'out/{name}-{last_commit}.pt')


if __name__ == '__main__':
    repo = git.Repo()
    repo.git.add('.')
    repo.git.commit('-m', '_Automated commit')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = mnist.load_data(n_train=10000, n_valid=1000, device=device)
    model = model.Model(data[0][0][0]).to(device)
    train_model(model, data)
    save_model(model, "fc")


