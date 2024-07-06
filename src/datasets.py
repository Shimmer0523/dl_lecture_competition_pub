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
from glob import glob


class MEG2ImageDataset(torch.utils.data.Dataset):
    """脳波MEGデータと脳波に対応する画像のデータセット"""

    def __init__(
        self,
        split: str,
        data_dir: str,
        transform: torchvision.transforms.Compose,
    ) -> None:
        super().__init__()

        self.split = split
        self.data_dir = data_dir
        self.num_classes = 1854
        self.num_samples = len(glob(os.path.join(data_dir, f"{split}_X", "*.npy")))

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
        return self.num_samples

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, Image.Image]:
        X_path = os.path.join(
            self.data_dir, f"{self.split}_X", str(i).zfill(5) + ".npy"
        )
        X = torch.from_numpy(np.load(X_path))

        subject_idx_path = os.path.join(
            self.data_dir, f"{self.split}_subject_idxs", str(i).zfill(5) + ".npy"
        )
        subject_idx = torch.from_numpy(np.load(subject_idx_path))

        img = Image.open(self.image_paths[i]).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return X, img, subject_idx


class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data") -> None:
        super().__init__()
        assert split in ["train", "val", "test"], f"Invalid split: {split}"

        self.split = split
        self.data_dir = data_dir
        self.num_classes = 1854
        self.num_samples = len(glob(os.path.join(data_dir, f"{split}_X", "*.npy")))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, i):
        X_path = os.path.join(
            self.data_dir, f"{self.split}_X", str(i).zfill(5) + ".npy"
        )
        X = torch.from_numpy(np.load(X_path))

        subject_idx_path = os.path.join(
            self.data_dir, f"{self.split}_subject_idxs", str(i).zfill(5) + ".npy"
        )
        subject_idx = torch.from_numpy(np.load(subject_idx_path))

        if self.split in ["train", "val"]:
            y_path = os.path.join(
                self.data_dir, f"{self.split}_y", str(i).zfill(5) + ".npy"
            )
            y = torch.from_numpy(np.load(y_path))

            return X, y, subject_idx
        else:
            return X, subject_idx

    @property
    def num_channels(self) -> int:
        return np.load(
            os.path.join(self.data_dir, f"{self.split}_X", "00000.npy")
        ).shape[0]

    @property
    def seq_len(self) -> int:
        return np.load(
            os.path.join(self.data_dir, f"{self.split}_X", "00000.npy")
        ).shape[1]
