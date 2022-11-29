import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models


class Efficientnet(nn.Module):
    """# Model with Pretrained Efficient Net B0

    Args:
        num_classes - number of classes in detection
    """

    def __init__(self, num_classes, dropout_rate=0.3):
        super(Efficientnet, self).__init__()
        model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )  # Pretrained Efficient Net b0 Model

        n_inputs = model.classifier[1].in_features
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.linear1 = nn.Linear(n_inputs, 256)
        self.linear2 = nn.Linear(256, 128)
        self.dp = nn.Dropout(dropout_rate)
        self.linear3 = nn.Linear(128, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):

        batch_size, frames_to_use, c, h, w = x.shape
        x = x.view(batch_size * frames_to_use, c, h, w)
        fmap = self.model(x)

        x = self.avgpool(fmap)
        x = x.view(x.shape[0], -1)
        x = F.leaky_relu(self.linear1(x))
        x = F.leaky_relu(self.dp(self.linear2(x)))
        x = self.linear3(x)

        output = F.softmax(x, dim=1)

        return output


class Efficientnet_GRU_Model(nn.Module):
    """# Model with Pretrained Efficient Net B0

    Args:
        num_classes - number of classes in detection
    """

    def __init__(
        self, num_classes, latent_dim=1280, gru_layers=2, hidden_dim=1280, bidirectional=False
    ):
        super(Efficientnet_GRU_Model, self).__init__()
        model = models.efficientnet_b0(pretrained=True)  # Pretrained Efficient Net b0 Model
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.gru = nn.GRU(latent_dim, hidden_dim, gru_layers, bidirectional)  # GRU Layer
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.3)
        self.linear1 = nn.Linear(1280, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 1280)
        x_gru, _ = self.gru(x, None)
        x = self.relu((self.linear1(torch.mean(x_gru, dim=1))))
        x = self.relu(self.dp(self.linear2(x)))
        x = self.linear3(x)
        return fmap, x
