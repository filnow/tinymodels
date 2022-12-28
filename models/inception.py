import torch
import torch.nn as nn
import torch.nn.functional as F

class Mixed5(nn.Module):
    def __init__(self, inch: int, outch: int, pool: int) -> None:
        super().__init__()
        self.branch1x1 = BasicConv2d(inch, outch, kernel_size=1)
        
        self.branch5x5_1 = BasicConv2d(inch, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, outch, kernel_size=5, padding=2)
        
        self.branch3x3dbl_1 = BasicConv2d(inch, outch, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(outch, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)
            
        self.maxpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.branch_pool = BasicConv2d(inch, pool, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        branches = [
            self.branch1x1(x),
            self.branch5x5_2(self.branch5x5_1(x)),
            self.branch3x3dbl_3(self.branch3x3dbl_2(self.branch3x3dbl_1(x))),
            self.branch_pool(self.maxpool(x))
        ]
 
        return torch.cat(branches, 1)


class Miexed6a(nn.Module):
    def __init__(self, inch: int, outch: int) -> None:
        super().__init__()
        self.branch3x3 = BasicConv2d(inch, outch, kernel_size=3, stride=2)

        self.branch3x3dbl_1 = BasicConv2d(inch, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, stride=2)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        branches = [
            self.branch3x3(x),
            self.branch3x3dbl_3(self.branch3x3dbl_2(self.branch3x3dbl_1(x))),
            self.maxpool(x)
        ]

        return torch.cat(branches, 1)


class Mixed6(nn.Module):
    def __init__(self, inch: int, outch: int, size: int) -> None:
        super().__init__()
        inch_copy = inch
        self.branch1x1 = BasicConv2d(inch, outch, kernel_size=1)
        
        self.branch7x7_1 = BasicConv2d(inch, size, kernel_size=1)
        self.branch7x7_2 = BasicConv2d(size, size, kernel_size=(1,7), padding=(0, 3))
        self.branch7x7_3 = BasicConv2d(size, outch, kernel_size=(7,1), padding=(3, 0))

        kernel_sizes = [1, (7,1), (1,7), (7,1), (1,7)]
        padding_sizes = [0, (3,0), (0,3), (3,0), (0,3)]
        
        for i, (kernel_size, padding_size) in enumerate(zip(kernel_sizes, padding_sizes)):
            setattr(self, f"branch7x7dbl_{i+1}", BasicConv2d(inch, size, kernel_size=kernel_size, padding=padding_size))
            inch = size
            size = 192 if i == 3 else size

        self.maxpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.branch_pool = BasicConv2d(inch_copy, outch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        branches = [
            self.branch1x1(x),
            self.branch7x7_3(self.branch7x7_2(self.branch7x7_1(x))),
            nn.Sequential(*[getattr(self, f"branch7x7dbl_{i+1}") for i in range(5)])(x),
            self.branch_pool(self.maxpool(x))
        ]
        
        return torch.cat(branches, 1)


class AuxInception(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv0 = BasicConv2d(768, 128, kernel_size=1)
        self.conv1 = BasicConv2d(128, 768, kernel_size=5)

        self.fc = nn.Linear(768, 1000)
        self.dropout = nn.Dropout(p=0.7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = F.adaptive_avg_pool2d(x, (5, 5))
        x = self.conv1(self.conv0(x))
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


class Mixed7a(nn.Module):
    def __init__(self, inch: int, outch: int) -> None:
        super().__init__()
        self.branch3x3_1 = BasicConv2d(inch, outch, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(outch, 320, kernel_size=3, stride=2)

        kernel_sizes = [1, (1,7), (7,1), 3]
        padding_sizes = [0, (0,3), (3,0), 0]
        
        for i, (kernel_size, padding_size) in enumerate(zip(kernel_sizes, padding_sizes)):
            stride = 2 if i == 3 else 1
            setattr(self, f"branch7x7x3_{i+1}", BasicConv2d(inch, outch, kernel_size=kernel_size, padding=padding_size, stride=stride))
            inch = outch

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        branches = [
            self.branch3x3_2(self.branch3x3_1(x)),
            nn.Sequential(*[getattr(self, f"branch7x7x3_{i+1}") for i in range(4)])(x),
            self.maxpool(x)
        ]

        return torch.cat(branches, 1)


class Mixed7(nn.Module):
    def __init__(self, inch: int, outch: int) -> None:
        super().__init__()
        self.branch1x1 = BasicConv2d(inch, outch, kernel_size=1)

        self.branch3x3_1 = BasicConv2d(inch, 384, kernel_size=1)
        self.branch3x3_2a = BasicConv2d(384, 384, kernel_size=(1,3), padding=(0,1))
        self.branch3x3_2b = BasicConv2d(384, 384, kernel_size=(3,1), padding=(1,0))

        self.branch3x3dbl_1 = BasicConv2d(inch, 448, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = BasicConv2d(384, 384, kernel_size=(1,3), padding=(0,1))
        self.branch3x3dbl_3b = BasicConv2d(384, 384, kernel_size=(3,1), padding=(1,0))

        self.maxpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.branch_pool = BasicConv2d(inch, 192, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]

        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = self.branch_pool(self.maxpool(x))

        branches = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        
        return torch.cat(branches, 1)


class BasicConv2d(nn.Module):
    def __init__(self, inch: int, outch: int, **kwargs) -> None:
        super().__init__()
        self.conv = nn.Conv2d(inch, outch, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(outch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return F.relu(self.bn(self.conv(x)), inplace=True)


class InceptionV3(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)
        
        self.Mixed_5b = Mixed5(192, 64, 32)
        self.Mixed_5c = Mixed5(256, 64, 64)
        self.Mixed_5d = Mixed5(288, 64, 64)
        
        self.Mixed_6a = Miexed6a(288, 384)
        
        self.Mixed_6b = Mixed6(768, 192, 128)
        self.Mixed_6c = Mixed6(768, 192, 160)
        self.Mixed_6d = Mixed6(768, 192, 160)
        self.Mixed_6e = Mixed6(768, 192, 192)

        self.AuxLogits = AuxInception()

        self.Mixed_7a = Mixed7a(768, 192)

        self.Mixed_7b = Mixed7(1280, 320)
        self.Mixed_7c = Mixed7(2048, 320)

        self.dropout = nn.Dropout()
        self.fc = nn.Linear(2048, 1000)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        
        x = self.maxpool(x)
        
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        
        x = self.maxpool(x)
        
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
 
        x = self.Mixed_6a(x)
      
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)

        aux = self.AuxLogits(x)

        x = self.Mixed_7a(x)

        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(self.dropout(x))
        
        return x