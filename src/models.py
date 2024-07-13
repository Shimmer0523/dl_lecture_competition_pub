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

        logit = (img_embedding @ MEG_embedding.T) / self.temperature
        img_similarity = img_embedding @ img_embedding.T
        MEG_similarity = MEG_embedding @ MEG_embedding.T
        target = F.softmax((img_similarity + MEG_similarity) / 2 * self.temperature, dim=-1)
        img_loss = F.cross_entropy(logit, target)
        MEG_loss = F.cross_entropy(logit.T, target.T)
        loss = (img_loss + MEG_loss) / 2
        return loss.mean()


class ImageEncoder(nn.Module):
    """画像の特徴量を抽出するモデル"""

    def __init__(self):
        super().__init__()
        self.encoder = torchvision.models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(self.encoder.children())[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        return x


class Transformer_Classifier(nn.Module):
    def __init__(self, d_model: int, num_classes: int):
        super().__init__()
        self.rearrange1 = Rearrange("b c t -> t b c")

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=2048, dropout=0.1, activation="gelu", layer_norm_eps=1e-5
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6, enable_nested_tensor=False)

        self.rearrange2 = Rearrange("t b c -> b c t")
        self.adaptive_avg_pool1d = nn.AdaptiveAvgPool1d(1)
        self.rearrange3 = Rearrange("b c t -> b (c t)")

        self.classifier = nn.Sequential(
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor):
        x = self.rearrange1(x)
        x = self.transformer_encoder(x)
        x = self.rearrange2(x)
        x = self.adaptive_avg_pool1d(x)
        x = self.rearrange3(x)
        x = self.classifier(x)
        return x


class LSTM_Classifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int, dropout: float, state_dict: dict = None):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = 2
        self.rearrange1 = Rearrange("b c t -> b t c")
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=dropout,
        )

        if state_dict is not None:
            self.lstm.load_state_dict(state_dict)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 2048),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2048, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.randn(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.randn(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        x = self.rearrange1(x)
        x, (_, _) = self.lstm(x, (h0, c0))
        x = self.classifier(x[:, -1, :])
        return x


class BasicConvClassifier(nn.Module):
    def __init__(self, num_classes: int, seq_len: int, in_channels: int, hid_dim: int = 128) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim),
            ConvBlock(hid_dim, hid_dim),
        )

        self.adaptiveAvgPool1d = nn.AdaptiveAvgPool1d(1)
        self.rearrenge = Rearrange("b d 1 -> b d")
        self.fc = nn.Linear(hid_dim, num_classes)

        # self.head = nn.Sequential(
        #     nn.AdaptiveAvgPool1d(1),
        #     Rearrange("b d 1 -> b d"),
        #     nn.Linear(hid_dim, num_classes),
        # )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """_summary_
        Args:
            X ( b, c, t ): _description_
        Returns:
            X ( b, num_classes ): _description_
        """
        X = self.blocks(X)
        X = self.adaptiveAvgPool1d(X)
        X = self.rearrenge(X)
        X = self.fc(X)

        return X


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
        X = self.dropout(X)

        X = self.conv1(X) + X  # skip connection
        X = F.gelu(self.batchnorm1(X))

        # X = self.conv2(X)
        # X = F.glu(X, dim=-2)

        return self.dropout(X)
