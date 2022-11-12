

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torch.hub import load_state_dict_from_url
import requests
from io import BytesIO
import sys

# Define model
class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1000),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        print(x.data.shape)
        x = self.classifier(x)
        return x
    
model = AlexNet()

link = 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth'
state = load_state_dict_from_url(link)
model.load_state_dict(state)

#Freeze param
for param in model.features.parameters():
    param.requires_grad = False

model.eval

transform = transforms.Compose([            
 
 transforms.Resize(256),                    
 transforms.CenterCrop(224),                
 transforms.ToTensor(),                     
 transforms.Normalize(                      
 mean=[0.485, 0.456, 0.406],                
 std=[0.229, 0.224, 0.225]                  
 
 )])

#'https://raw.githubusercontent.com/srirammanikumar/DogBreedClassifier/master/images/Labrador_retriever_06457.jpg'
response = requests.get(sys.argv[1])
img = Image.open(BytesIO(response.content))
img_t = transform(img)
batch_t = torch.unsqueeze(img_t, 0)
out = model(batch_t)
print('Out: ', out)

labels = requests.get('https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt').text.split('\n')

_, index = torch.max(out, 1)
print('index: ', index)
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
print('percentage: ', percentage)
print(labels[index[0]], percentage[index[0]].item())