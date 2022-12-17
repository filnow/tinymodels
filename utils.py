import torch
from PIL import Image
from torchvision import transforms
import requests
from io import BytesIO

def class_img(model, image_path):
  
  transform = transforms.Compose([            
 
    transforms.Resize(256),                    
    transforms.CenterCrop(224),                
    transforms.ToTensor(),                     
    transforms.Normalize(                      
    mean=[0.485, 0.456, 0.406],                
    std=[0.229, 0.224, 0.225]                  
  
  )])

  img = Image.open(image_path)
  img_t = transform(img)
  batch_t = torch.unsqueeze(img_t, 0)
  out = model(batch_t)
  labels = requests.get('https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt').text.split('\n')
  _, index = torch.max(out, 1)
  percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
  
  return labels[index[0]], percentage[index[0]].item()

