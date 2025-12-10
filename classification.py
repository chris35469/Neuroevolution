import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple dataset (4 samples, 2 input features, 2 output classes)
X = torch.tensor([[1.0, 2.0],
                  [2.0, 1.0],
                  [1.0, -1.0],
                  [-1.0, -2.0]], dtype=torch.float32)
# Labels: 0 or 1 (for two classes) - multiclass
y = torch.tensor([0, 0, 1, 1], dtype=torch.long)

# Define a simple neural network for multiclass classification
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, 2)  # 2 output classes

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.softmax(x, dim=1)
        return x

model = SimpleNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 20 == 0:
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

# Test the model
with torch.no_grad():
    test_input = torch.tensor([[1.0, 2.0], [-1.5, -2.0]], dtype=torch.float32)
    test_outputs = model(test_input)
    predictions = torch.argmax(test_outputs, dim=1)
    print(test_outputs)
    print("Test predictions:", predictions.numpy())  # Output class indices



