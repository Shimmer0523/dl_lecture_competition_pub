import torch
import torch.nn as nn
import torchaudio

from src.models import ConvBlock

x = torch.zeros(size=(16, 271, 281))
x = torchaudio.transforms.Resample(orig_freq=200, new_freq=50)(x)
print(x.shape)
x = x.view(x.size(0), -1)
print(x.shape)
