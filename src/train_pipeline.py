import torch
from src.config.config import epochs, bce_loss, learning_rate
from src.preprocessing.preprocessors import sequential_nn
from src.preprocessing.data_management import data_gen

optimizer = torch.optim.RMSprop(sequential_nn.parameters(), lr=learning_rate)

for e in range(epochs):
    running_loss = 0.0
    for X_train_mb, Y_train_mb in data_gen:
        optimizer.zero_grad()
        nn_out = sequential_nn(X_train_mb)
        nn_out = nn_out.view(-1)
        loss_func = bce_loss(nn_out, Y_train_mb)
        loss_func.backward()
        optimizer.step()
        running_loss += loss_func.item()

    avg_loss = running_loss / len(data_gen)
    print(f"Epoch # {e + 1}, loss function value {avg_loss:.6f}")