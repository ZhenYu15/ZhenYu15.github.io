import torch
import torch.nn as nn

class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )
    def forward(self, x):
        return self.net(x)

def build_model():
    return TinyMLP()

def build_example_input(model):
    return torch.randn(1, 10)
