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
import hydra
from omegaconf import DictConfig

from src.datasets import Image2CategoryDataset
from src.models import ResNet18
from src.utils import set_seed


@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(args: DictConfig):
    set_seed(123)

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    train_dataset = Image2CategoryDataset(
        "train", data_dir=args.data_dir, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True
    )
    val_dataset = Image2CategoryDataset(
        "val", data_dir=args.data_dir, transform=transform
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet18(cls_num=1854).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 10
    for epoch in range(epochs):

        model.train()
        for i, (img, y) in enumerate(train_loader):
            img = img.to(device)
            y = F.one_hot(y, num_classes=1854).float().to(device)

            # 順伝搬
            pred = model(img)
            loss = criterion(pred, y)

            # 逆伝搬
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f"Epoch: {i / epochs}, Loss: {loss.item()}")

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for img, y in val_loader:
                img = img.to(device)
                y = y.to(device)

                pred = model(img)

                cls = torch.argmax(pred, dim=1)
                correct += (cls == y).sum().item()
                total += y.size(0)
        accuracy = correct / total
        print(f"Epoch: {epoch}, Accuracy: {accuracy:.4f}")

    torch.save(model.state_dict(), "/content/model/pretrained_resnet.pth")


if __name__ == "__main__":
    run()
