from typing import Type
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url


class BootleNeck(nn.Module):
    def __init__(self, inch: int, outch: int, s: int, block0 = True) -> None:
        super().__init__()

        self.s = s

        if block0:
            self.conv1 = nn.Conv2d(inch, outch, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(outch)
        
        self.conv = [
            nn.Conv2d(outch, outch, kernel_size=3, stride=s, groups=outch, bias=False),
            nn.ReLU6(),
            nn.Conv2d(outch, inch, kernel_size=1, bias=False),
            nn.BatchNorm2d(outch)
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        ident = x

        x = F.relu6(self.bn1(self.conv1(x)))
        x = F.relu6(self.conv2(x))
        x = self.bn2(self.conv3)

        if self.s != 2:
            x += ident

        return x


class MobileNetV2(nn.Module):
    def __init__(self) -> None:
        super().__init__()


        self.features = nn.Sequential(
            self.first_conv(),
            self._make_layer(BootleNeck,0,16,32,1)
            
        )

        self.avgpool = nn.AdaptiveMaxPool2d((7,7))

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1280,1000)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _make_layer(self, BootleNeck: Type[BootleNeck], num_Blocks: int, in_channels: int, out_channels: int, stride: int) -> nn.Sequential:
        
        layers = [BootleNeck(in_channels, out_channels, s=stride, block0=False)]
        
        if num_Blocks != 0:
            for _ in range(num_Blocks):
                layers.append(BootleNeck(in_channels, out_channels, s=1))
            
        return nn.Sequential(*layers)

    @staticmethod
    def first_conv():
        return nn.Sequential(*[nn.Conv2d(3, 32, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6()
            ])


data = load_state_dict_from_url('https://download.pytorch.org/models/mobilenet_v2-b0353104.pth')

model = MobileNetV2()
model.load_state_dict(data)
model.eval

for i in data.keys():
    print(i, data[i].shape)