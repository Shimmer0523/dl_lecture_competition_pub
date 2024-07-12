import torch
import torch.nn as nn
import torchaudio

from src.models import ConvBlock


for i in range(3):
    print(torch.cuda.is_available())
