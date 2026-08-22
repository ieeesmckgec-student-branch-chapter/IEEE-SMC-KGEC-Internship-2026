"""Classification backbones used by the benchmark.

A dropout layer sits before the final linear layer in every backbone so that
the MC-dropout deferral policy can be evaluated without changing the
architecture between policies.
"""
import torch.nn as nn
from torchvision.models import resnet18

DROPOUT_P = 0.3


class SimpleCNN(nn.Module):
    """Three-block convolutional network, used as the lightweight baseline."""

    def __init__(self, in_channels: int, n_outputs: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.dropout = nn.Dropout(DROPOUT_P)
        self.classifier = nn.Linear(128, n_outputs)

    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.classifier(self.dropout(x))


def build_resnet18(in_channels: int, n_outputs: int) -> nn.Module:
    model = resnet18(weights=None)
    if in_channels != 3:
        model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(DROPOUT_P), nn.Linear(in_features, n_outputs))
    return model


def build_model(backbone: str, in_channels: int, n_outputs: int) -> nn.Module:
    if backbone == "resnet18":
        return build_resnet18(in_channels, n_outputs)
    if backbone == "simplecnn":
        return SimpleCNN(in_channels, n_outputs)
    raise ValueError(f"Unknown backbone: {backbone}")


def enable_dropout_only(model: nn.Module) -> None:
    """Put the model in eval mode but re-enable dropout for MC sampling."""
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()
