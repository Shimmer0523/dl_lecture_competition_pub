import os, sys
from icecream import ic
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics import Accuracy
from torchvision import transforms
from PIL import Image

from src.datasets import ThingsMEGDataset
from src.models import ResNet18, BasicConvClassifier
from src.utils import set_seed


def load_preprocess_image(path: str):
    transform = transforms.Compose(
        [
            transforms.Resize((244, 244)),
            transform.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image = Image.open(fp=path).convert(mode="RGB")
    image = transform(image).unsqueeze(0)
    return image

def provide_dataloader():
    train_dataset = 

def run():
    model = ResNet18(in_channels=3, cls_num=1854)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    epochs = 10
    for epoch in range(epochs):
        model.train()


if __name__ == "__main__":
    pass
