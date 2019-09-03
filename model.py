import torch
import torch.nn as nn
import torchvision.models as models


class Baseline(nn.Module):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=hidden_size, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(hidden_size),
            nn.Conv2d(hidden_size, hidden_size, 3, 2, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(hidden_size),
            nn.Conv2d(hidden_size, hidden_size, 3, 2, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(hidden_size),
            nn.Conv2d(hidden_size, hidden_size, 3, 2, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(hidden_size),
            nn.Conv2d(hidden_size, hidden_size, 3, 2, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(hidden_size),
            nn.Conv2d(hidden_size, out_size, 4, 1),
        )

    def forward(self, image):
        return self.net(image).squeeze(-1).squeeze(-1)


class Resnet18(nn.Module):
    def __init__(self, num_classes, dropout=False):
        super().__init__()
        model = models.resnet18(pretrained=True)
        model = list(model.children())[:-1]
        if dropout:
            model.append(nn.Dropout(0.2))
        model.append(nn.Conv2d(512, num_classes, 1))
        self.net = nn.Sequential(*model)

    def forward(self, x):
        return self.net(x).squeeze(-1).squeeze(-1)


class Resnet50(nn.Module):
    def __init__(self, num_classes, dropout=False):
        super().__init__()
        model = models.resnet50(pretrained=True)
        model = list(model.children())[:-1]
        if dropout:
            model.append(nn.Dropout(0.2))
        model.append(nn.Conv2d(2048, num_classes, 1))
        self.net = nn.Sequential(*model)

    def forward(self, x):
        return self.net(x).squeeze(-1).squeeze(-1)


class Resnext50(nn.Module):
    def __init__(self, num_classes, dropout=False):
        super().__init__()
        model = models.resnext50_32x4d(pretrained=True)
        model = list(model.children())[:-1]
        if dropout:
            model.append(nn.Dropout(0.2))
        model.append(nn.Conv2d(2048, num_classes, 1))
        self.net = nn.Sequential(*model)

    def forward(self, x):
        return self.net(x).squeeze(-1).squeeze(-1)


class Resnext101(nn.Module):
    def __init__(self, num_classes, dropout=False):
        super().__init__()
        model = models.resnext101_32x8d(pretrained=True)
        model = list(model.children())[:-1]
        if dropout:
            model.append(nn.Dropout(0.1))
        model.append(nn.Conv2d(2048, num_classes, 1))
        self.net = nn.Sequential(*model)

    def forward(self, x):
        return self.net(x).squeeze(-1).squeeze(-1)