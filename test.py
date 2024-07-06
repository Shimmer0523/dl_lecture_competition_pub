import torch
import torch.nn as nn

x = torch.zeros(size=(3, 271*281))
s = nn.Sequential(
    nn.AdaptiveAvgPool1d(512),
    nn.Linear(512, 10),
)
out = s(x)

print(out.shape)