import os, sys
from icecream import ic
import numpy as np
import scipy.signal
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils
import torch.utils.data
from torchmetrics import Accuracy
from torchvision import transforms
import torchaudio.transforms as T
from PIL import Image
import hydra
from omegaconf import DictConfig

from src.datasets import MEG2ImageDataset
from src.models import MEGClip
from src.utils import set_seed


@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(args: DictConfig):
    set_seed(123)

    img_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    train_dataset = MEG2ImageDataset(
        "train", data_dir=args.data_dir, transform=img_transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True
    )
    val_dataset = MEG2ImageDataset(
        "val", data_dir=args.data_dir, transform=img_transform
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MEGClip().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0015)

    epochs = 3
    for epoch in range(epochs):

        model.train()
        for i, (X, img) in enumerate(train_loader):
            X = X.to(device)
            img = img.to(device)

            # 順伝搬
            loss = model(MEG=X, img=img)

            # 逆伝搬
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f"Epoch: {epoch}, Iteration: {i}, Loss: {loss.item()}")

    torch.save(
        model.state_dict(),
        "/content/drive/MyDrive/03_Colab Notebooks/DLBasics2023_colab/GraduationProject/model/pretrained_resnet.pth",
    )


if __name__ == "__main__":
    run()
