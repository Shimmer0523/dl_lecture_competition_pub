import os, sys
from icecream import ic
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils
import torch.utils.data
from torchmetrics import Accuracy
from torchvision import transforms
from PIL import Image

from src.datasets import Image2CategoryDataset
from src.models import ResNet18
from src.utils import set_seed


def run():
    set_seed(123)

    train_dataset = Image2CategoryDataset("train", data_dir=data_dir)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True
    )
    val_dataset = Image2CategoryDataset("val", data_dir=data_dir)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=True)

    model = ResNet18(in_channels=3, cls_num=1854)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    epochs = 10
    for epoch in range(epochs):
        model.train()

        for img, y in train_loader:
            pass


if __name__ == "__main__":
    pass
