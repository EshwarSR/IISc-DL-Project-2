import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.autograd import Variable
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from sklearn.model_selection import train_test_split
import time
import scikitplot as skplt
import matplotlib.pyplot as plt
import os


def get_train_valid_data():
    trans_img = transforms.Compose([transforms.ToTensor()])
    data = FashionMNIST("./data", train=True,
                        transform=trans_img, download=True)

    X = data.train_data
    X = X.reshape(60000, 1, 28, 28)  # Reshaping for use of Convolution
    X = X.type(torch.FloatTensor)
    X = X / 255  # Normalizing

    y = data.train_labels

    X_train, X_validate, y_train, y_validate = train_test_split(
        X, y, test_size=10000, stratify=y, random_state=42)

    return X_train, X_validate, y_train, y_validate


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def evaluate(model, X, y, loss_fn, batch_size):
    all_y_truth = []
    all_y_pred = []
    loss_meter = AverageMeter("loss")
    for start in range(0, len(X), batch_size):
        end = start + batch_size
        X_batch = X[start:end]
        y_batch = y[start:end]
        preds = model(X_batch)
        loss = loss_fn(preds, y_batch).item()
        _, pred_classes = torch.max(preds, 1)

        all_y_pred.extend(pred_classes)
        all_y_truth.extend(y_batch)
        loss_meter.update(loss, X_batch.shape[0])

    acc = accuracy_score(all_y_truth, all_y_pred)
    return acc, loss_meter.avg, all_y_pred, all_y_truth


def plot_confusion_matrices(model_name, epoch_num, train_y_truth, train_y_pred, validate_y_truth, validate_y_pred):
    skplt.metrics.plot_confusion_matrix(
        train_y_truth, train_y_pred, normalize=True)
    plot_folder = "plots/" + model_name + "/"
    if not os.path.exists(plot_folder):
        os.mkdir(plot_folder)
    plt.savefig(plot_folder + epoch_num + '_cm.png')
    plt.title(model_name)
    plt.clf()
    plt.close()


def train_evaluate_model(model_name, model, num_classes, num_epochs, batch_size, learning_rate, X_train, X_validate, y_train, y_validate, evaluate_every, print_every, device=None):

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    train_losses = []
    validation_losses = []
    train_accuracies = []
    validation_accuracies = []
    epoch_ticks = []

    for epoch in range(num_epochs):
        epoch_start = time.time()
        # Training
        model.train()
        for start in range(0, len(X_train), batch_size):
            end = start + batch_size
            x = X_train[start:end]
            y = y_train[start:end]

            # Moving to device if given
            if device:
                x.to(device)
                y.to(device)

            optimizer.zero_grad()

            pred_y = model(x)
            loss = loss_fn(pred_y, y)
            loss.backward()
            optimizer.step()

        # Evaluating every evaluate_every epochs
        if epoch % evaluate_every == (evaluate_every - 1):
            epoch_ticks.append(epoch)
            model.eval()

            with torch.no_grad():
                # Train data set evaluation
                train_acc, train_loss, train_y_pred, train_y_truth = evaluate(
                    model, X_train, y_train, loss_fn, batch_size)
                train_accuracies.append(train_acc)
                train_losses.append(train_loss)

                # Validation data set evaluation
                validation_acc, validation_loss, validate_y_pred, validate_y_truth = evaluate(
                    model, X_validate, y_validate, loss_fn, batch_size)
                validation_accuracies.append(validation_acc)
                validation_losses.append(validation_loss)

                # Writing Confusion matrices
                plot_confusion_matrices(
                    model_name, str(epoch+1), train_y_truth, train_y_pred, validate_y_truth, validate_y_pred)

                # Saving the model
                model_dir = "models/" + model_name + "/"
                if not os.path.exists(model_dir):
                    os.mkdir(model_dir)
                torch.save(model.state_dict(), model_dir +
                           str(epoch+1) + ".pt")

        if epoch % print_every == (print_every - 1):
            epoch_end = time.time()
            print("Epoch:", epoch+1, "Train Loss:",
                  train_loss, "Validation Loss:", validation_loss, "Time:", epoch_end - epoch_start)

    return model, train_losses, validation_losses, train_accuracies, validation_accuracies, epoch_ticks, train_y_truth, train_y_pred, validate_y_truth, validate_y_pred
