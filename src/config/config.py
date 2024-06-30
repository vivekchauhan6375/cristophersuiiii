import pandas as pd
import numpy as np
import torch
from torch.nn import functional as F
from torch.nn import Sequential
from torch import nn
from torch.utils.data import Dataset, DataLoader


# Model Hyperparameters
minibatch_size = 2
epochs = 199
learning_rate = 1e-3

# Loss Function and Optimizer
bce_loss = nn.BCELoss()