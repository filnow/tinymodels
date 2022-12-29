import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
import torch.nn.functional as F
from utils import class_img


class DanseLayer(nn.Module):
    def __init__(self, inch: int, outch: int) -> None:
        super().__init__()
        self.norm1 = nn.BatchNorm2d(inch)
        self.conv1 = nn.Conv2d(inch, outch, kernel_size=1)

        self.norm2 = nn.BatchNorm2d(inch)
        self.conv2 = nn.Conv2d(inch, outch, kernel_size=3)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.conv1(self.relu(self.norm1(x)))
        x = self.conv2(self.relu(self.norm2(x)))

        return x

class DenseBlock(nn.Module):
    def __init__(self, num: int) -> None:
        super().__init__()
        self.danseblock = DanseLayer()

        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return x

    def _make_layer(inch: int, outch:int, layer_num: int) -> nn.Sequential:

        layer = []

        for _ in range(layer_num):
            layer.append([
                nn.BatchNorm2d(inch),
                nn.ReLU(inplace=True),
                nn.Conv2d(inch, outch, kernel_size=1, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(outch),
                nn.ReLU(inplace=True),
                nn.Conv2d(outch, inch//2, kernel_size=3, stride=1, bias=False)
            ])
        
        return nn.Sequential(*layer)


class Transition(nn.Module):
    def __init__(self, inch: int, outch: int) -> None:
        super().__init__()
        self.norm = nn.BatchNorm2d(inch)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(inch, outch, kernel_size=1, bias=False)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.conv(self.relu(self.norm(x)))
        x = self.avgpool(x)

        return x


class DenseNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.features = nn.Sequential(
            
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            DenseBlock(),
            Transition(256, 128),
            DenseBlock(),
            Transition(512, 256),
            DenseBlock(),
            Transition(1024, 512),
            DenseBlock(),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1024, 1000)
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x



model = DenseNet()

data = load_state_dict_from_url('https://download.pytorch.org/models/densenet121-a639ec97.pth')

model.load_state_dict(data, strict=False)

for i in data.keys():
    print(i, data[i].shape)