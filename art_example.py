"""
The script demonstrates a simple example of using ART with PyTorch. The example train a small model
on the MNIST dataset and creates adversarial examples using the Fast Gradient Sign Method. Here we
use the ART classifier to train the model, it would also be possible to provide a pretrained model
to the ART classifier. The parameters are chosen for reduced computational requirements of the
script and not optimised for accuracy.
"""
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier
from art.utils import load_mnist

# Step 0: Define the neural network model, return logits instead of activation in forward method


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5, stride=1)
        self.conv_2 = nn.Conv2d(in_channels=4, out_channels=10, kernel_size=5, stride=1)
        self.fc_1 = nn.Linear(in_features=4 * 4 * 10, out_features=100)
        self.fc_2 = nn.Linear(in_features=100, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 10)
        x = F.relu(self.fc_1(x))
        x = self.fc_2(x)
        return x


if __name__ == "__main__":
    # Step 1: Load the MNIST dataset

    (
        (x_train, train_y),
        (x_test, y_test),
        min_pixel_value,
        max_pixel_value,
    ) = load_mnist()
    print(len(x_train), len(x_test))

    # Step 1a: Swap axes to PyTorch's NCHW format

    x_train = np.transpose(x_train, (0, 3, 1, 2)).astype(np.float32)
    x_test = np.transpose(x_test, (0, 3, 1, 2)).astype(np.float32)
    x_train, train_y, x_test, y_test = (
        x_train[:58000],
        train_y[:58000],
        x_train[58000:],
        train_y[58000:],
    )

    # Step 2: Create the model

    model = Net()

    # Step 2a: Define the loss function and the optimizer

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Step 3: Create the ART classifier

    classifier = PyTorchClassifier(
        model=model,
        clip_values=(min_pixel_value, max_pixel_value),
        loss=criterion,
        optimizer=optimizer,
        input_shape=(1, 28, 28),
        nb_classes=10,
    )

    # Step 4: Train the ART classifier

    classifier.fit(x_train, train_y, batch_size=64, nb_epochs=6)

    # Step 5: Evaluate the ART classifier on benign test examples

    predictions = classifier.predict(x_test)
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
    print("Accuracy on benign test examples: {}%".format(accuracy * 100))

    # Step 6: Generate adversarial test examples
    attack = FastGradientMethod(estimator=classifier, eps=0.2)
    x_test_adv = attack.generate(x=x_test)

    if False:  # Compare original image with adversarially modified image
        import matplotlib.pyplot as plt

        i_img = 0
        plt.imshow(x_test[i_img][0], cmap="gray")
        plt.subplots()
        plt.imshow(x_test_adv[i_img][0], cmap="gray")
        plt.show()

    # Step 7: Evaluate the ART classifier on adversarial test examples

    predictions = classifier.predict(x_test_adv)
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
    print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))
