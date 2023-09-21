import torch
import torch.nn as nn


class ANN(nn.Module):
    def __init__(self, input_size, num_classes=8):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 264)
        self.fc3 = nn.Linear(264, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x