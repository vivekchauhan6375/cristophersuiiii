from torch.nn import Sequential
from torch import nn

class sequential_mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.sequential_nn = Sequential(
            nn.Linear(in_features=2, out_features=4, device="cpu"),
            nn.ReLU(),
            nn.Linear(in_features=4, out_features=2, device="cpu"),
            nn.ReLU(),
            nn.Linear(in_features=2, out_features=1, device="cpu"),
            nn.Sigmoid()
        )
    
    def forward(self, inp):
        nn_out = self.sequential_nn(inp)
        return nn_out

sequential_nn = sequential_mlp()