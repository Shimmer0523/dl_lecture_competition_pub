import torch
import torch.nn as nn

from src.models import ConvBlock

x = torch.zeros(size=(16, 271, 281))
y = ConvBlock(in_dim=271, out_dim=281)(x)

print(y.shape)
