import torch
import torch.nn as nn
from game import start_game
import time


times = [   ]

# Define a simple neural network for multiclass classification
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(3, 8)
        self.fc2 = nn.Linear(8, 2)  # 2 output classes

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.softmax(x, dim=1)
        return x


for i in range(20):
    print(f"Iteration {i}")
    model = SimpleNet()
    time_alive = start_game(model)
    times.append(time_alive)
    print("waiting for 3 seconds")
    time.sleep(3)
    # Print out the model's weights
    # for name, param in model.named_parameters():
    #     print(f"Weights of {name}:")
    #     print(param.data)

    # Test the model
    # with torch.no_grad():
    #     test_input = torch.tensor([[1.0, 2.0], [-1.5, -2.0]], dtype=torch.float32)
    #     test_outputs = model(test_input)
    #     predictions = torch.argmax(test_outputs, dim=1)
    #     print(test_outputs.numpy())
    #     print("Test predictions:", predictions.numpy())  # Output class indices
print(times)