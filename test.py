from icecream import ic
import os
import torch
import torch.nn as nn
import torchaudio
import matplotlib.pyplot as plt

from src.models import ConvBlock
from src.datasets import ThingsMEGDataset

train_set = ThingsMEGDataset("train", "/workspaces/PythonProjects/dl_lecture_competition_pub/data")

X, y, idx = train_set[0]
ic(X.shape)

X2 = torchaudio.functional.resample(X, orig_freq=200, new_freq=20)
X2 = X2 - torch.mean(X2, dim=1, dtype=torch.float32, keepdim=True)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

ax[0].imshow(X.numpy(), aspect="auto", origin="lower")
ax[1].imshow(X2.numpy(), aspect="auto", origin="lower")

plt.savefig("dl_lecture_competition_pub/plot/plot.png")
