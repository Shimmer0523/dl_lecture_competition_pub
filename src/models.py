import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class ResidualBlock(nn.Module):
    """残差ブロック"""

    def __init__(
        self, in_channels: int, out_channels: int, stride: int, downsampler: nn.Module
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
        )
        self.batch_norm1 = nn.BatchNorm2d(num_features=out_channels)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.batch_norm2 = nn.BatchNorm2d(num_features=out_channels)
        self.downsampler = downsampler

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        z = F.relu(self.batch_norm1(self.conv1(x)))
        z = self.batch_norm2(self.conv2(z))

        if self.downsampler:
            residual = self.downsampler(x)

        z += residual
        z = F.relu(z)
        return z


class ResNet18(nn.Module):
    def __init__(
        self,
        in_channels: int,
        emb_dim: int,
    ):
        """
        Args:
        in_channels[int]: 入力のチャネル数
        emb_dim[int]: 特徴ベクトルの次元数
        """
        super().__init__()
        self.in_channels = in_channels
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3
        )
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(out_channels=64, block_num=2, stride=1)
        self.layer2 = self._make_layer(
            out_channels=128,
            block_num=2,
            stride=2,
        )
        self.layer3 = self._make_layer(
            out_channels=256,
            block_num=2,
            stride=2,
        )
        self.layer4 = self._make_layer(
            out_channels=512,
            block_num=2,
            stride=2,
        )
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, emb_dim)

        # 重みの初期化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
        self,
        out_channels: int,
        block_num: int,
        stride: int,
    ):
        downsampler = None
        if stride != 1 or (self.in_channels != out_channels):
            downsampler = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                ),
                nn.BatchNorm2d(num_features=out_channels),
            )

        layers = []
        layers.append(
            ResidualBlock(self.in_channels, out_channels, stride, downsampler)
        )
        self.in_channels = out_channels
        for _ in range(1, block_num):
            layers.append(ResidualBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.adaptive_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class BasicConvClassifier(nn.Module):
    def __init__(
        self, num_classes: int, seq_len: int, in_channels: int, hid_dim: int = 128
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim),
            ConvBlock(hid_dim, hid_dim),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(hid_dim, num_classes),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """_summary_
        Args:
            X ( b, c, t ): _description_
        Returns:
            X ( b, num_classes ): _description_
        """
        X = self.blocks(X)

        return self.head(X)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size: int = 3,
        p_drop: float = 0.1,
    ) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")
        # self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size) # , padding="same")

        self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)

        self.dropout = nn.Dropout(p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.in_dim == self.out_dim:
            X = self.conv0(X) + X  # skip connection
        else:
            X = self.conv0(X)

        X = F.gelu(self.batchnorm0(X))

        X = self.conv1(X) + X  # skip connection
        X = F.gelu(self.batchnorm1(X))

        # X = self.conv2(X)
        # X = F.glu(X, dim=-2)

        return self.dropout(X)
