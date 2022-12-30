from collections import OrderedDict
import torch
import torch.nn as nn


class DenseLayer(nn.Module):
    def __init__(self, inch: int) -> None:
        super().__init__()
        self.norm1 = nn.BatchNorm2d(inch)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(inch, 128, kernel_size=1, bias=False)

        self.norm2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(128, 32, kernel_size=3, padding=1, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = [x] if isinstance(x, torch.Tensor) else x
        
        x = torch.cat(x, 1)

        x = self.conv1(self.relu1(self.norm1(x)))
        x = self.conv2(self.relu2(self.norm2(x)))
        
        return x


class DenseBlock(nn.Module):
    def __init__(self, inch: int, num_layer: int) -> None:
        super().__init__()
        self.num_layer = num_layer

        for i in range(num_layer):
            setattr(self, f"denselayer{i+1}", DenseLayer(inch))
            inch += 32

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        out = [x]
        
        for i in range(self.num_layer):

            layer_out = getattr(self, f"denselayer{i+1}")(out)

            out.append(layer_out)

        out = torch.cat(out, 1)

        return out


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
        self.features = nn.Sequential(
            
            OrderedDict(
                [
                    ("conv0", nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)),
                    ("norm0", nn.BatchNorm2d(64)),
                    ("relu0", nn.ReLU(inplace=True)),
                    ("pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                ]
            )
        )
        
        denseblock_ch = 64
        transition_ch = 256
        num_block = 6

        for i in range(3):
            self.features.add_module(f'denseblock{i+1}', DenseBlock(denseblock_ch,num_block))
            self.features.add_module(f'transition{i+1}', Transition(transition_ch, transition_ch//2))
            denseblock_ch *= 2
            transition_ch *= 2
            num_block *= 2
        
        self.features.add_module('denseblock4', DenseBlock(512,16))

        self.features.add_module('norm5', nn.BatchNorm2d(1024))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout()
        self.classifier = nn.Linear(1024, 1000)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(self.dropout(x))
        
        return x

