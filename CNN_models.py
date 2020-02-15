import torch
import torch.nn as nn
import torch.nn.functional as F

################
# Experiment 1 #
################


class Conv_2L_8_12_2L_128_32_AVG_BN(nn.Module):
    def __init__(self, n_classes=10):
        super(Conv_2L_8_12_2L_128_32_AVG_BN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, stride=1,
                      padding=2),  # (N,8,28,28)
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=4, stride=2, padding=2),  # (N,8,14,14)
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 12, kernel_size=4, stride=1,
                      padding=1),  # (N,12,14,14)
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=4, stride=2),  # (N,12,6,6)
            nn.BatchNorm2d(12)
        )
        self.flat_units = 12*6*6
        self.hidden_units = [128, 32]
        self.classifier = nn.Sequential(
            nn.Linear(self.flat_units, self.hidden_units[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_units[0], self.hidden_units[1]),
            nn.ReLU(),
            nn.Linear(self.hidden_units[1], n_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class Conv_2L_8_12_2L_128_32_AVG_DO(nn.Module):
    def __init__(self, n_classes=10):
        super(Conv_2L_8_12_2L_128_32_AVG_DO, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, stride=1,
                      padding=2),  # (N,8,28,28)
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=4, stride=2, padding=2),  # (N,8,14,14)
            nn.Dropout2d(0.2),
            nn.Conv2d(8, 12, kernel_size=4, stride=1,
                      padding=1),  # (N,12,14,14)
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=4, stride=2),  # (N,12,6,6)
            nn.Dropout2d(0.2)
        )
        self.flat_units = 12*6*6
        self.hidden_units = [128, 32]
        self.classifier = nn.Sequential(
            nn.Linear(self.flat_units, self.hidden_units[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_units[0], self.hidden_units[1]),
            nn.ReLU(),
            nn.Linear(self.hidden_units[1], n_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class Conv_2L_8_12_2L_128_32_MAX_BN(nn.Module):
    def __init__(self, n_classes=10):
        super(Conv_2L_8_12_2L_128_32_MAX_BN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, stride=1,
                      padding=2),  # (N,8,28,28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=2),  # (N,8,14,14)
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 12, kernel_size=4, stride=1,
                      padding=1),  # (N,12,14,14)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=2),  # (N,12,6,6)
            nn.BatchNorm2d(12)
        )
        self.flat_units = 12*6*6
        self.hidden_units = [128, 32]
        self.classifier = nn.Sequential(
            nn.Linear(self.flat_units, self.hidden_units[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_units[0], self.hidden_units[1]),
            nn.ReLU(),
            nn.Linear(self.hidden_units[1], n_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class Conv_2L_8_12_2L_128_32_MAX_DO(nn.Module):
    def __init__(self, n_classes=10):
        super(Conv_2L_8_12_2L_128_32_MAX_DO, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, stride=1,
                      padding=2),  # (N,8,28,28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=2),  # (N,8,14,14)
            nn.Dropout2d(0.2),
            nn.Conv2d(8, 12, kernel_size=4, stride=1,
                      padding=1),  # (N,12,14,14)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=2),  # (N,12,6,6)
            nn.Dropout2d(0.2)
        )
        self.flat_units = 12*6*6
        self.hidden_units = [128, 32]
        self.classifier = nn.Sequential(
            nn.Linear(self.flat_units, self.hidden_units[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_units[0], self.hidden_units[1]),
            nn.ReLU(),
            nn.Linear(self.hidden_units[1], n_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

################
# Experiment 2 #
################


class Conv_3L_8_12_16_3L_128_64_32_AVG_BN(nn.Module):
    def __init__(self, n_classes=10):
        super(Conv_3L_8_12_16_3L_128_64_32_AVG_BN, self).__init__()
        self.conv = nn.Sequential(
            # CNN Layer 1
            nn.Conv2d(1, 8, kernel_size=5, stride=1,
                      padding=2),  # (N,8,28,28)
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=4, stride=2, padding=2),  # (N,8,14,14)
            nn.BatchNorm2d(8),

            # CNN Layer 2
            nn.Conv2d(8, 12, kernel_size=4, stride=1,
                      padding=1),  # (N,12,14,14)
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=4, stride=2),  # (N,12,6,6)
            nn.BatchNorm2d(12),

            # CNN Layer 3
            nn.Conv2d(12, 16, kernel_size=4, stride=1,
                      padding=1),  # (N,16,5,5)
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=3, stride=2),  # (N,16,2,2)
            nn.BatchNorm2d(16)

        )
        self.flat_units = 16*2*2
        self.hidden_units = [128, 64, 32]
        self.classifier = nn.Sequential(
            nn.Linear(self.flat_units, self.hidden_units[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_units[0], self.hidden_units[1]),
            nn.ReLU(),
            nn.Linear(self.hidden_units[1], self.hidden_units[2]),
            nn.ReLU(),
            nn.Linear(self.hidden_units[2], n_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class Conv_3L_8_12_16_3L_128_64_32_MAX_BN(nn.Module):
    def __init__(self, n_classes=10):
        super(Conv_3L_8_12_16_3L_128_64_32_MAX_BN, self).__init__()
        self.conv = nn.Sequential(
            # CNN Layer 1
            nn.Conv2d(1, 8, kernel_size=5, stride=1,
                      padding=2),  # (N,8,28,28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=2),  # (N,8,14,14)
            nn.BatchNorm2d(8),

            # CNN Layer 2
            nn.Conv2d(8, 12, kernel_size=4, stride=1,
                      padding=1),  # (N,12,14,14)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=2),  # (N,12,6,6)
            nn.BatchNorm2d(12),

            # CNN Layer 3
            nn.Conv2d(12, 16, kernel_size=4, stride=1,
                      padding=1),  # (N,16,5,5)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (N,16,2,2)
            nn.BatchNorm2d(16)

        )
        self.flat_units = 16*2*2
        self.hidden_units = [128, 64, 32]
        self.classifier = nn.Sequential(
            nn.Linear(self.flat_units, self.hidden_units[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_units[0], self.hidden_units[1]),
            nn.ReLU(),
            nn.Linear(self.hidden_units[1], self.hidden_units[2]),
            nn.ReLU(),
            nn.Linear(self.hidden_units[2], n_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
