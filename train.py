import torch

import mnist
import model


def train_model(model, data):
    model.train()

    batch_size = 1000
    n_iter = 1000
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    (train_imgs, train_labels), (valid_imgs, valid_labels) = data
    
    for iter in range(n_iter):
        indices = torch.randint(high=len(train_imgs), size=(batch_size,))
        batch_xs = xs[indices]
        batch_ys = ys[indices]

        batch_outputs = net(batch_xs)
        loss = get_loss(batch_outputs, batch_ys)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if iter % 100 == 0:
            print(f"Iter {iter} \tloss {loss.item()}")
            print_accuracy("Batch", batch_outputs, batch_ys)

    model.eval()


if __name__ == '__main__':
    data = mnist.load_data(n_train=10000, n_valid=1000)
    model = model.Model(data[0][0][0])

