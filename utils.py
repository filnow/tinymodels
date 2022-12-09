import torch
import numpy as np
from tinygrad.tensor import Tensor 
from PIL import Image
from torchvision import transforms
import requests
from io import BytesIO


'''
transform= transforms.Compose([            
 
 transforms.Resize(256),                    
 transforms.CenterCrop(224),                
 transforms.ToTensor(),                     
 transforms.Normalize(                      
 mean=[0.485, 0.456, 0.406],                
 std=[0.229, 0.224, 0.225]                  
 
 )])
'''

def infer(model, img):
  
  # preprocess image
  aspect_ratio = img.size[0] / img.size[1]
  img = img.resize((int(224*max(aspect_ratio,1.0)), int(224*max(1.0/aspect_ratio,1.0))))

  img = np.array(img)
  y0,x0=(np.asarray(img.shape)[:2]-224)//2
  retimg = img = img[y0:y0+224, x0:x0+224]

  # low level preprocess
  img = np.moveaxis(img, [2,0,1], [0,1,2])
  img = img.astype(np.float32)[:3].reshape(1,3,224,224)
  img /= 255.0
  img -= np.array([0.485, 0.456, 0.406]).reshape((1,-1,1,1)) #mean
  img /= np.array([0.229, 0.224, 0.225]).reshape((1,-1,1,1)) #std

  out = model.forward(Tensor(img)).cpu()

  return out, retimg




def class_img(model):
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
  #import matplotlib.pyplot as plt
  #plt.plot(percentage.detach().numpy())
  #plt.show()
  print(index[0])
  print(labels[index[0]], percentage[index[0]].item())