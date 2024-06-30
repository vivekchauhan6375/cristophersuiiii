# Data Preparation
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from src.config.config import minibatch_size


data = pd.DataFrame(data={"x1": [0, 0, 1, 1], "x2": [0, 1, 0, 1], "y": [0, 1, 1, 0]})

class xor_dataset(Dataset):
    def __init__(self, data):
        self.training_data = data

    def __len__(self):
        return len(self.training_data)
        
    def __getitem__(self, idx):
        row = self.training_data.iloc[idx]
        X_train = torch.tensor(row.iloc[0:2].values, dtype=torch.float32)
        Y_train = torch.tensor(row.iloc[2], dtype=torch.float32)
        return X_train, Y_train

# DataLoader
Xordataset = xor_dataset(data)
data_gen = DataLoader(dataset=Xordataset, batch_size=minibatch_size)