import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectedResidualBlock(nn.Module):
    def __init__(self, n_channels, embed_dim):
        super(ProjectedResidualBlock, self).__init__()
        self.conv0 = nn.Conv2d(
            n_channels, embed_dim, kernel_size=1, stride=1, padding=1, bias=False
        )
        self.bn0 = nn.BatchNorm2d(embed_dim)
        self.conv1 = nn.Conv2d(
            embed_dim, embed_dim, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(embed_dim)
        self.conv2 = nn.Conv2d(
            embed_dim, n_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(n_channels)

    def forward(self, x):
        h = F.relu(self.bn0(self.conv0(x)), inplace=True)
        h = F.relu(self.bn1(self.conv1(x)), inplace=True)
        return F.relu(x + self.bn2(self.conv2(h)), inplace=True)


class MyrtleResidualBlock(nn.Module):
    def __init__(self, n_channels):
        super(MyrtleResidualBlock, self).__init__()
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
                block.append(MyrtleResidualBlock(64 * 2 ** (l + 1)))
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

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)), inplace=True)
        return F.relu(self.shortcut(x) + self.bn2(self.conv2(h)), inplace=True)

class ResNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        n_classes=10,
        n_layers=3,
        height=32,
        width=32,
        architecture="resnet",
    ):
        if architecture != "resnet":
            raise ValueError(
                f"Incorrect architecture specification '{architecture}' for model ResNet!"
            )
        super(ResNet, self).__init__()
        self.embed = nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )

        # Add convolutional blocks that increase channels by powers of 2.
        self.layers = nn.ModuleList()
        for l in range(n_layers):
            self.layers.append(ResidualBlock(64 * 2**l, 64 * 2 ** (l + 1)))
            height = (height + 1) // 2
            width  = (width + 1) // 2

        # Compute final number of features.
        n_features = 64 * 2**n_layers * height * width
        self.classifier = nn.Sequential(nn.Flatten(1), nn.Linear(n_features, n_classes))

    def forward(self, x, y=None, sample_weight=None):
        h = F.relu(self.embed(x), inplace=True)
        for layer in self.layers:
            h = layer(h)
        logits = self.classifier(h)
        loss = None
        if not (y is None):
            if not (sample_weight is None):
                loss = torch.dot(sample_weight, F.cross_entropy(logits, y, reduction="none"))
            else:
                loss = F.cross_entropy(logits, y)
        return loss, logits


# Used for unbalanced Fashion MNIST dataset from LL's code.
class ConvNet(nn.Module):
    def __init__(self, hidden_dim=512, n_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(64*3*3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_classes)

    def forward(self, x, y=None, sample_weight=None):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)  # conv1
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # conv2
        x = F.max_pool2d(F.relu(self.conv3(x)), 2) # conv3 
        x = x.view(x.shape[0], -1) # flatten
        features = self.fc1(x)
        x = F.relu(features)
        logits = self.fc2(x)
        loss = None
        if not (y is None):
            if not (sample_weight is None):
                loss = torch.dot(sample_weight, F.cross_entropy(logits, y, reduction="none"))
            else:
                loss = F.cross_entropy(logits, y)
        return loss, logits