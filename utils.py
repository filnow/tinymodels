import torch
from PIL import Image
from torchvision import transforms
import requests
from torch.hub import load_state_dict_from_url
from typing import Tuple, Any, Union
from models import *

def class_img(model: type, image_path: str) -> Tuple[Any, Union[int, float]]:
  
  transform = transforms.Compose([            
 
    transforms.Resize(256),                    
    transforms.CenterCrop(224),                
    transforms.ToTensor(),                     
    transforms.Normalize(                      
    mean=[0.485, 0.456, 0.406],                
    std=[0.229, 0.224, 0.225]                  
  
  )])

  img = Image.open(image_path)
  batch_t = torch.unsqueeze(transform(img), 0)
  out = model(batch_t)
  labels = requests.get('https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt').text.split('\n')
  _, index = torch.max(out, 1)
  percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
  
  return (labels[index[0]], percentage[index[0]].item())

def run_model(model: Any) -> Tuple[type, str]:

  model_name = type(model).__name__

  pretrained_weights = {
    
    type(AlexNET()).__name__ : 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
    #EfficientNET() : 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth',
    type(GoogleNET()).__name__ : 'https://download.pytorch.org/models/googlenet-1378be20.pth',
    type(ResNet()).__name__ : 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    type(VGG19()).__name__ : 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
  
  }
  
  assert model_name in pretrained_weights.keys(), f'There is no model called {model_name}' 
  
  data = load_state_dict_from_url(pretrained_weights[model_name])
  model.load_state_dict(data)
  model.eval()
  
  return (model, model_name)