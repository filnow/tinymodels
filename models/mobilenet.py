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
            [1,  16, 1, 1],
            [6,  24, 2, 2],
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        
        #TODO: make it better

        features = [self.first_conv(3, 32, 3, 2), BootleNeck(96,32,16,1, block0=False)]

        features.append(BootleNeck(16,96,24,2)) #f2
        features.append(BootleNeck(24,144,24,1)) #f3
        features.append(BootleNeck(24,144,32,2)) #f4
        features.append(BootleNeck(32,192,32,1)) #f5
        features.append(BootleNeck(32,192,32,1)) #f6 
        features.append(BootleNeck(32,192,64,2)) #f7
        features.append(BootleNeck(64,384,64,1)) #f8
        features.append(BootleNeck(64,384,64,1)) #f9
        features.append(BootleNeck(64,384,64,1)) #f10
        features.append(BootleNeck(64,384,96,1)) #f11
        features.append(BootleNeck(96,576,96,1)) #f12
        features.append(BootleNeck(96,576,96,1)) #f13
        features.append(BootleNeck(96,576,160,2)) #f14
        features.append(BootleNeck(160,960,160,1)) #f15
        features.append(BootleNeck(160,960,160,1)) #f16
        features.append(BootleNeck(160,960,320,1)) #f17
    
        features.append(self.first_conv(320, 1280, 1, 1))

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
    def first_conv(inch: int, outch: int, k: int, s: int):
        return nn.Sequential(*[nn.Conv2d(inch, outch, kernel_size=k, stride=s, bias=False),
            nn.BatchNorm2d(outch),
            nn.ReLU6(inplace=True)
            ])