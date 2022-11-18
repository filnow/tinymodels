import numpy as np
from tinygrad.tensor import Tensor 


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