import os
from icecream import ic
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchaudio
from PIL import Image
from typing import Tuple
from termcolor import cprint
import torch.utils
import torch.utils.data


class MEG2ImageDataset(torch.utils.data.Dataset):
    """脳波MEGデータと脳波に対応する画像のデータセット"""

    def __init__(
        self,
        split: str,
        data_dir: str,
        transform: torchvision.transforms.Compose,
    ) -> None:
        super().__init__()

        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split

        # MEG
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))

        # 画像ファイルのパスをリスト化
        with open(os.path.join(data_dir, f"{split}_image_paths.txt"), "r") as file:
            lines = file.readlines()
            lines = [
                (
                    line.strip()
                    if "/" in line
                    else f"{'_'.join(line.split('_')[:-1])}/{line}"
                )
                for line in lines
            ]
        self.image_paths = [f"/content/data/Images/{line.strip()}" for line in lines]

        self.transform = transform

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, Image.Image]:
        x = self.X[i].reshape(1, -1)
        img = Image.open(self.image_paths[i]).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return x, img


class Image2CategoryDataset(torch.utils.data.Dataset):
    """画像と画像に対応するカテゴリラベルのデータセット"""

    def __init__(
        self,
        split: str,
        data_dir: str,
        transform: torchvision.transforms.Compose = None,
    ):
        super().__init__()

        # トランスフォーム
        self.transform = transform

        # 画像ファイルのパスをリスト化
        with open(os.path.join(data_dir, f"{split}_image_paths.txt"), "r") as file:
            lines = file.readlines()
            lines = [
                (
                    line.strip()
                    if "/" in line
                    else f"{'_'.join(line.split('_')[:-1])}/{line}"
                )
                for line in lines
            ]
        self.image_paths = [f"/content/data/Images/{line.strip()}" for line in lines]

        # 画像のカテゴリラベル
        self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index):
        img = Image.open(self.image_paths[index]).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, self.y[index]


class ThingsMEGDataset(torch.utils.data.Dataset):
    """脳波MEGデータと脳波に対応する画像のカテゴリラベルのデータセット"""

    def __init__(
        self,
        split: str,
        data_dir: str = "data",
    ) -> None:
        super().__init__()

        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854

        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
        self.subject_idxs = torch.load(
            os.path.join(data_dir, f"{split}_subject_idxs.pt")
        )

        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert (
                len(torch.unique(self.y)) == self.num_classes
            ), "Number of classes do not match."

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        if hasattr(self, "y"):
            return self.X[i], self.y[i], self.subject_idxs[i]
        else:
            return self.X[i], self.subject_idxs[i]

    @property
    def num_channels(self) -> int:
        return self.X.shape[1]

    @property
    def seq_len(self) -> int:
        return self.X.shape[2]
