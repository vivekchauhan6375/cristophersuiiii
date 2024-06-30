# import pandas as pd
# import numpy as np
# import torch
# import train_pipeline as tp

# X_test = pd.DataFrame(data={"x1": [0, 0, 1, 1], "x2": [0, 1, 0, 1]})
# X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)

# with torch.no_grad():
#     predictions = tp.sequential_nn(X_test_tensor)
#     predictions = predictions.numpy() 

# print("Predictions:")
# print(predictions)

# binary_predictions = (predictions > 0.5).astype(int)
# print("Binary Predictions:")
# print(binary_predictions)
import torch
from torch import nn
import torch.optim as optim

# Define the XOR dataset from "x1" and "x2"
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# Define the neural network model
class XORModel(nn.Module):
    def __init__(self):
        super(XORModel, self).__init__()
        self.fc1 = nn.Linear(2, 4)  # Input layer to hidden layer
        self.fc2 = nn.Linear(4, 1)  # Hidden layer to output layer

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# Instantiate the model
model = XORModel()

# Define loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error loss
optimizer = optim.Adam(model.parameters(), lr=0.1)  # Adam optimizer

# Training the model
epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad()  # Zero the gradients
    outputs = model(X)  # Forward pass
    loss = criterion(outputs, y)  # Compute the loss
    loss.backward()  # Backward pass
    optimizer.step()  # Update weights

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Evaluation and predictions
model.eval()
with torch.no_grad():
    # Use specific inputs "x1" and "x2"
    x1 = torch.tensor([0, 0, 1, 1], dtype=torch.float32).view(-1, 1)
    x2 = torch.tensor([0, 1, 0, 1], dtype=torch.float32).view(-1, 1)
    X_pred = torch.cat((x1, x2), dim=1)  # Combine x1 and x2 into input tensor
    predictions = model(X_pred)
    predictions = (predictions > 0.5).float()  # Convert probabilities to binary predictions

    print("Predictions:")
    for i in range(len(X_pred)):
        print(f"x1: {x1[i].item()}, x2: {x2[i].item()}, Predicted Output: {predictions[i].item()}")
