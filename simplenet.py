import torch
import torch.nn as nn


# Define a simple neural network for multiclass classification
class SimpleNet(nn.Module):
    def __init__(self, verbose=False):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(3, 8)
        self.fc2 = nn.Linear(8, 2)  # 2 output classes
        self.verbose = verbose

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.softmax(x, dim=1)
        return x
