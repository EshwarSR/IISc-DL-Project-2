import utils
import matplotlib.pyplot as plt
import time
import numpy as np
import pdb
import os
from tqdm import tqdm

from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision.datasets import FashionMNIST
from torchvision import transforms


class LeNet(nn.Module):

    def __init__(self, n_classes=10):
        emb_dim = 20
        '''
        Define the initialization function of LeNet, this function defines
        the basic structure of the neural network
        '''

        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.emb = nn.Linear(64*7*7, emb_dim)
        self.clf = nn.Linear(emb_dim, n_classes)

    def num_flat_features(self, x):
        '''
        Calculate the total tensor x feature amount
        '''

        size = x.size()[1:]  # All dimensions except batch dimension
        num_features = 1
        for s in size:
            num_features *= s

        return num_features

    def forward(self, x):
        # x = x.view(-1, 1, 28, 28)
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = self.emb(x)
        out = self.clf(x)

        return out


num_epochs = 100
evaluate_every = 10
print_every = 10
learning_rate = 0.05
batch_size = 128
num_classes = 10
input_size = 28*28

###################
# Loading Dataset #
###################

X_train, X_validate, y_train, y_validate = utils.get_train_valid_data()
# # Flattening
# X_train = X_train.reshape(X_train.shape[0], -1)
# X_validate = X_validate.reshape(X_validate.shape[0], -1)


models = {
    "lenet": LeNet()
}


################################
# Training & Evaluating Models #
################################
metrics = {
    "train_loss": {},
    "validation_loss": {},
    "train_accuracies": {},
    "validation_accuracies": {}
}

for model_name, model in models.items():
    print("Started training model:", model_name)
    start = time.time()
    # model = torch.nn.Sequential(*layers)

    model, train_losses, validation_losses, train_accuracies, validation_accuracies, epoch_ticks = utils.train_evaluate_model(
        model, num_classes, num_epochs, batch_size, learning_rate, X_train, X_validate, y_train, y_validate, evaluate_every, print_every)

    metrics["train_loss"][model_name] = train_losses
    metrics["validation_loss"][model_name] = validation_losses
    metrics["train_accuracies"][model_name] = train_accuracies
    metrics["validation_accuracies"][model_name] = validation_accuracies

    end = time.time()
    print("Starting Loss:", train_losses[0], validation_losses[0])
    print("Ending Loss:", train_losses[-1], validation_losses[-1])

    print("Starting Accuracy:", train_accuracies[0], validation_accuracies[0])

    print("Ending Accuracy:", train_accuracies[-1], validation_accuracies[-1])

    print("Time taken:", end-start)

    print("Saving the model")
    torch.save(model.state_dict(), "models/"+model_name+".pt")

    # Loading the models
    # model = TheModelClass(*args, **kwargs)
    # model.load_state_dict(torch.load(PATH))
    # model.eval()

    print("\n\n")


###############################
# Plotting the Metrics Graphs #
###############################

def plot_metrics(metrics_data, metrics_name):
    for model_name in models.keys():
        plt.plot(epoch_ticks, metrics_data[model_name], label=model_name)
        plt.legend(loc='best')
        plt.title(metrics_name)
    plt.savefig(metrics_name+'.png')
    plt.clf()


plot_metrics(metrics["train_loss"], "train_loss")
plot_metrics(metrics["validation_loss"], "validation_loss")
plot_metrics(metrics["train_accuracies"], "train_accuracies")
plot_metrics(metrics["validation_accuracies"], "validation_accuracies")
