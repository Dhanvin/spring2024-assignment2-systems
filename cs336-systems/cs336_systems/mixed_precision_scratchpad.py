import torch
import torch.nn as nn


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int): 
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False) 
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        print(f"input: {x.dtype}")
        print(f"output-fc1: {self.fc1(x).dtype}")
        x = self.relu(self.fc1(x)) 
        print(f"output-relu: {x.dtype}")
        x = self.ln(x)
        print(f"output-layer-norm: {x.dtype}")
        x = self.fc2(x)
        print(f"output-fc2: {x.dtype}")
        return x

batch_size = 32
in_features = 50
out_features = 10
input = torch.randn(size=(batch_size, in_features), dtype=torch.float32).to(torch.device("cuda:0"))
model = ToyModel(in_features, out_features).to(torch.device("cuda:0"))
with torch.autocast('cuda'):
    for name, p in model.named_parameters():
        print(f"{name}: {p.dtype}")
    model(input)