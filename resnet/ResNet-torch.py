import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torch.hub import load_state_dict_from_url
import requests
from io import BytesIO
import sys

class Block(nn.Module):

    def __init__(self, in_channels, out_channels, strd=1, sample=False) -> None:
        super().__init__()
        
        self.sample = sample
        self.conv1 = nn.Conv2d(in_channels, out_channels,kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3, stride=strd,padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, 4 * out_channels, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(4 * out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        if sample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, 4 * out_channels,kernel_size=1,stride=strd, bias=False),
                nn.BatchNorm2d(4 * out_channels)
            )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        
        ident = x.clone()
        
        x = self.relu(self.bn1(self.conv1(x)))
        
        print(x.shape, ident.shape)
        x = self.relu(self.bn2(self.conv2(x)))
    
        x = self.bn3(self.conv3(x))
        
        if self.sample:
            ident = self.downsample(ident)
        
        x += ident
        x = self.relu(x)
       
        return x

class ResNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.in_ch = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(Block, 2, 64, 1)
        self.layer2 = self._make_layer(Block, 3, 128, 2)
        self.layer3 = self._make_layer(Block, 5, 256, 2)
        self.layer4 = self._make_layer(Block, 2, 512, 2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout()
        self.fc = nn.Linear(512 * 4, 1000)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.relu(self.bn1(self.conv1(x)))
 
        x = self.maxpool(x)
        x = self.avgpool(x)
      
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = torch.flatten(x, 1)
        x = self.fc(self.dropout(x))
        return x

    def _make_layer(self, Block, num_Blocks, in_channels, stride):
        
        layers = [Block(self.in_ch,in_channels, stride, sample=True)]
        
        self.in_ch = in_channels*4

        for _ in range(num_Blocks):
            layers.append(Block(self.in_ch,in_channels))
        
        return nn.Sequential(*layers)

data  = load_state_dict_from_url('https://download.pytorch.org/models/resnet50-0676ba61.pth')
model = ResNet()
model.load_state_dict(data)
model.eval()

transform = transforms.Compose([            
 
 transforms.Resize(256),                    
 transforms.CenterCrop(224),                
 transforms.ToTensor(),                     
 transforms.Normalize(                      
 mean=[0.485, 0.456, 0.406],                
 std=[0.229, 0.224, 0.225]                  
 
 )])

#'https://raw.githubusercontent.com/srirammanikumar/DogBreedClassifier/master/images/Labrador_retriever_06457.jpg'
response = requests.get('https://raw.githubusercontent.com/srirammanikumar/DogBreedClassifier/master/images/Labrador_retriever_06457.jpg')
img = Image.open(BytesIO(response.content))
img_t = transform(img)
batch_t = torch.unsqueeze(img_t, 0)
out = model(batch_t)
labels = requests.get('https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt').text.split('\n')
_, index = torch.max(out, 1)
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

print(labels[index[0]], percentage[index[0]].item())