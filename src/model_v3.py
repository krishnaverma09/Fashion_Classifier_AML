"""v3 architectures: models from the reference papers and a residual CNN; input float32 (N,1,28,28) in [0,1].

  [IJST] Sumera et al. (2022), Implementation of CNN and ANN for Fashion-MNIST using different optimizers.
  [JSR]  Kadam, Adamuthe & Patil (2020), CNN model for image classification on MNIST and Fashion-MNIST.
  [EY]   Edwards & Yen, ECE 471 final project: Fashion-MNIST.

Imports torch only, so the demo can load a model without any dataset or training module.
"""
from torch import nn
import torch.nn.functional as F


class ANN(nn.Module):
    """[IJST] 784-3000-1000-10 dense network; dropout 0.3 added because 5.4M weights overfit quickly."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(784, 3000), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(3000, 1000), nn.ReLU(), nn.Dropout(0.3), nn.Linear(1000, 10))

    def forward(self, images):
        return self.net(images)


class IJSTCNN(nn.Module):
    """[IJST] conv 32 3x3 - pool - conv 64 3x3 - pool - dense 64 (121,930 parameters, as in the paper)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(1, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
                                 nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2),
                                 nn.Flatten(), nn.Linear(64 * 5 * 5, 64), nn.ReLU(), nn.Linear(64, 10))

    def forward(self, images):
        return self.net(images)


class KadamArch3(nn.Module):
    """[JSR] Architecture 3, the paper's best on Fashion-MNIST: three 2x2 convolutions of 64, dropout 0.25."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(1, 64, 2), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout(0.25),
                                 nn.Conv2d(64, 64, 2), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout(0.25),
                                 nn.Conv2d(64, 64, 2), nn.ReLU(), nn.Dropout(0.25),
                                 nn.Flatten(), nn.Linear(64 * 5 * 5, 64), nn.ReLU(), nn.Dropout(0.25),
                                 nn.Linear(64, 10))

    def forward(self, images):
        return self.net(images)


class KadamArch5(nn.Module):
    """[JSR] Architecture 5: VGG-style 32-32 / 64-64 3x3 convolutions, dense 512, dropout 0.25 and 0.5."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(1, 32, 3), nn.ReLU(), nn.Conv2d(32, 32, 3), nn.ReLU(),
                                 nn.MaxPool2d(2), nn.Dropout(0.25),
                                 nn.Conv2d(32, 64, 3), nn.ReLU(), nn.Conv2d(64, 64, 3), nn.ReLU(),
                                 nn.MaxPool2d(2), nn.Dropout(0.25),
                                 nn.Flatten(), nn.Linear(64 * 4 * 4, 512), nn.ReLU(), nn.Dropout(0.5),
                                 nn.Linear(512, 10))

    def forward(self, images):
        return self.net(images)


class KerasCNN(nn.Module):
    """[EY] Keras MNIST example: conv 32 and 64 3x3 - pool - dropout 0.25 - dense 128 - dropout 0.5."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(1, 32, 3), nn.ReLU(), nn.Conv2d(32, 64, 3), nn.ReLU(),
                                 nn.MaxPool2d(2), nn.Dropout(0.25),
                                 nn.Flatten(), nn.Linear(64 * 12 * 12, 128), nn.ReLU(), nn.Dropout(0.5),
                                 nn.Linear(128, 10))

    def forward(self, images):
        return self.net(images)


class ResidualBlock(nn.Module):
    def __init__(self, i, o, stride=1):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(i, o, 3, stride, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(),
                                  nn.Conv2d(o, o, 3, 1, 1, bias=False), nn.BatchNorm2d(o))
        self.skip = (nn.Identity() if i == o and stride == 1 else
                     nn.Sequential(nn.Conv2d(i, o, 1, stride, bias=False), nn.BatchNorm2d(o)))

    def forward(self, x):
        return F.relu(self.body(x) + self.skip(x))


class ResNet(nn.Module):
    """BatchNorm + residual skips (Bhatnagar et al., 92.54%, cited by [JSR] and [EY]); 28 -> 14 -> 7 -> pooled."""
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU())
        self.stages = nn.Sequential(ResidualBlock(32, 32), ResidualBlock(32, 32),
                                    ResidualBlock(32, 64, 2), ResidualBlock(64, 64),
                                    ResidualBlock(64, 128, 2), ResidualBlock(128, 128))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, 10))

    def forward(self, images):
        return self.head(self.stages(self.stem(images)))
