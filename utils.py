import torch
from PIL import Image
from torchvision import transforms
import requests
import re
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from typing import Tuple, Any, Union
from models import *


#copy from pytorch DenseNet to match dict
def _load_state_dict(model: nn.Module, data_link: str) -> None:

    pattern = re.compile(
        r"^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$"
    )

    state_dict = load_state_dict_from_url(data_link)
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    
    model.load_state_dict(state_dict)

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

def run_model(model_name: str) -> nn.Module:

  pretrained_weights = {
    
    'AlexNet' : 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
    'EfficientNet' : 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth',
    'GoogleNet' : 'https://download.pytorch.org/models/googlenet-1378be20.pth',
    'ResNet' : 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'VGG' : 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'InceptionV3' : 'https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth',
    'MobileNetV2' : 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
    'DenseNet' : 'https://download.pytorch.org/models/densenet121-a639ec97.pth'
  
  }
  
  assert model_name in pretrained_weights.keys(), f'There is no model called {model_name}' 
 
  model: nn.Module = globals().get(model_name)()

  if model_name == 'DenseNet':
    _load_state_dict(model, pretrained_weights[model_name])
  
  else:
    data = load_state_dict_from_url(pretrained_weights[model_name])
    model.load_state_dict(data)
  
  model.eval()
  
  return model