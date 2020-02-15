import utils
import time
import torch
import matplotlib.pyplot as plt
import scikitplot as skplt
import CNN_models

num_epochs = 500
evaluate_every = 20
print_every = 20
learning_rate = 0.01
batch_size = 1024
num_classes = 10
input_size = 28*28


###################
# Loading Dataset #
###################

X_train, X_validate, y_train, y_validate = utils.get_train_valid_data()
# Flattening
# X_train = X_train.reshape(X_train.shape[0], -1)
# X_validate = X_validate.reshape(X_validate.shape[0], -1)


#####################
# Model Definitions #
#####################

# Experiment 1
# models = {
#     "Conv_2L_8_12_2L_128_32_AVG_BN": CNN_models.Conv_2L_8_12_2L_128_32_AVG_BN(num_classes),
#     "Conv_2L_8_12_2L_128_32_AVG_DO": CNN_models.Conv_2L_8_12_2L_128_32_AVG_DO(num_classes),
#     "Conv_2L_8_12_2L_128_32_MAX_BN": CNN_models.Conv_2L_8_12_2L_128_32_MAX_BN(num_classes),
#     "Conv_2L_8_12_2L_128_32_MAX_DO": CNN_models.Conv_2L_8_12_2L_128_32_MAX_DO(num_classes)
# }

# Experiment 2
models = {
    "Conv_3L_8_12_16_3L_128_64_32_AVG_BN": CNN_models.Conv_3L_8_12_16_3L_128_64_32_AVG_BN(num_classes),
    "Conv_3L_8_12_16_3L_128_64_32_MAX_BN": CNN_models.Conv_3L_8_12_16_3L_128_64_32_MAX_BN(num_classes)
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

    model, train_losses, validation_losses, train_accuracies, validation_accuracies, epoch_ticks, train_y_truth, train_y_pred, validate_y_truth, validate_y_pred = utils.train_evaluate_model(
        model_name, model, num_classes, num_epochs, batch_size, learning_rate, X_train, X_validate, y_train, y_validate, evaluate_every, print_every)

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
