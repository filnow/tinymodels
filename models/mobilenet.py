import torch
import torch.nn as nn


class BootleNeck(nn.Module):
    def __init__(self, inch: int, outch: int, hidden_dim: int, s: int, block0: bool = True) -> None:
        super().__init__()

        self.identity = s == 1 and inch == hidden_dim
        layers = []

        if block0:
            layers.append(
                self.conv1x1(inch,outch)    
            )
        
        layers.extend([
            self.conv3x3(outch,s),
            nn.Conv2d(outch, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        if self.identity:
            
            return x + self.conv(x)

        return self.conv(x)

    @staticmethod
    def conv1x1(inch: int, outch: int) -> nn.Sequential:

        return nn.Sequential(
            nn.Conv2d(inch, outch, kernel_size=1, bias=False),
            nn.BatchNorm2d(outch),
            nn.ReLU6(inplace=True)
        )

    @staticmethod
    def conv3x3(inch: int, s: int) -> nn.Sequential:
        
        return nn.Sequential(
            nn.Conv2d(inch, inch, kernel_size=3, stride=s, padding=1, groups=inch, bias=False),
            nn.BatchNorm2d(inch),
            nn.ReLU6(inplace=True)
        )

    
class MobileNetV2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.cfgs = [
            # t, c, n, s
            [24,  16, 1, 2],
            [32,  24, 2, 2],
            [64,  32, 3, 2],
            [96,  64, 4, 2],
            [160,  96, 3, 2],
            [320, 160, 3, 1]
        ]

        features = [self.convkxk(3, 32, 3, 2), BootleNeck(96,32,16,1, block0=False)]

        for t,c,n,s in self.cfgs:
            for i in range(n):
                if i+1 == n:
                    features.append(BootleNeck(c,c*6,t,s))
                else:
                    features.append(BootleNeck(c,c*6,c,1))

        features.append(self.convkxk(320, 1280, 1, 1))

        self.features = nn.Sequential(*features)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(1280,1000)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    @staticmethod
    def convkxk(inch: int, outch: int, k: int, s: int):
        return nn.Sequential(*[nn.Conv2d(inch, outch, kernel_size=k, stride=s, bias=False),
            nn.BatchNorm2d(outch),
            nn.ReLU6(inplace=True)
            ])