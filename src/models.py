from icecream import ic
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from einops.layers.torch import Rearrange


class MEGClip(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = 1.0
        self.img_encoder = ImageEncoder()
        self.MEG_encoder = MEGLSTM()

    def forward(self, MEG: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        img_embedding = self.img_encoder(img)
        MEG_embedding = self.MEG_encoder(MEG)

        ic(img_embedding.shape)
        ic(MEG_embedding.shape)

        logit = (img_embedding @ MEG_embedding.T) / self.temperature
        img_similarity = img_embedding @ img_embedding.T
        MEG_similarity = MEG_embedding @ MEG_embedding.T
        target = F.softmax(
            (img_similarity + MEG_similarity) / 2 * self.temperature, dim=-1
        )
        img_loss = F.cross_entropy(logit, target)
        MEG_loss = F.cross_entropy(logit.T, target.T)
        loss = (img_loss + MEG_loss) / 2
        return loss.mean()


class ImageEncoder(nn.Module):
    """画像の特徴量を抽出するモデル"""

    def __init__(self):
        super().__init__()
        self.encoder = torchvision.models.resnet34(pretrained=True)
        self.features = nn.Sequential(*list(self.encoder.children())[:-2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        return x


class MEGLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=658,
            hidden_size=1024,
            num_layers=2,
            batch_first=True,
            dropout=0.25,
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        ic(X.shape)
        _, (h, _) = self.lstm(X)
        return h[-1]


class MEGClassifier(nn.Module):
    def __init__(self, num_classes: int, state_dict: dict = None):
        super().__init__()
        self.encoder = MEGLSTM()
        self.encoder.load_state_dict(state_dict)

        self.classifier = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = F.softmax(self.classifier(x))
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
        ic(X.shape)
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
