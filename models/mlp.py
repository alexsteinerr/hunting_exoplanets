import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim=7, h1=256, h2=256, out_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1), nn.ReLU(),
            nn.Linear(h1, h2),    nn.ReLU(),
            nn.Linear(h2, out_dim)
        )
    
    def forward(self, x):
        return self.net(x)