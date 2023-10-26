import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, n_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            n_channels, n_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.conv2 = nn.Conv2d(
            n_channels, n_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(n_channels)

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)), inplace=True)
        h = F.relu(self.bn2(self.conv2(h)), inplace=True)
        return x + h


class MyrtleNet(nn.Module):
    def __init__(
        self,
        n_channels=3,
        n_classes=10,
        n_layers=3,
        residual_blocks=[0, 2],
        height=32,
        width=32,
        architecture="myrtle_net",
    ):
        if architecture != "myrtle_net":
            raise ValueError(
                f"Incorrect architecture specification '{architecture}' for model MyrtleNet!"
            )
        super(MyrtleNet, self).__init__()
        self.prep = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        # Add convolutional blocks that increase channels by powers of 2.
        self.layers = nn.ModuleList()
        for l in range(n_layers):
            block = [
                nn.Conv2d(
                    64 * 2**l,
                    64 * 2 ** (l + 1),
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(64 * 2 ** (l + 1)),
                nn.ReLU(True),
                nn.MaxPool2d(2),
            ]
            if l in residual_blocks:
                block.append(ResidualBlock(64 * 2 ** (l + 1)))
            self.layers.append(nn.Sequential(*block))
            height //= 2
            width //= 2

        # Compute final number of features after pooling.
        height //= 2
        width //= 2
        n_features = 64 * 2**n_layers * height * width
        self.classifier = nn.Sequential(
            nn.MaxPool2d(2), nn.Flatten(1), nn.Linear(n_features, n_classes)
        )

    def forward(self, x, y=None):
        h = self.prep(x)
        for layer in self.layers:
            h = layer(h)
        logits = self.classifier(h)
        loss = None
        if not (y is None):
            loss = F.cross_entropy(logits, y)
        return loss, logits
