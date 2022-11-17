from tinygrad.tensor import Tensor 
from torch.hub import load_state_dict_from_url
from PIL import Image
import requests
from io import BytesIO
import numpy as np

class VGG19():

    def __init__(self):
        
        self.features = {}
        self.classifier = {}

        self.features_weight = {}
        self.classifier_weight = {}
        
        self.features_bias = {}
        self.classifier_bias = {}
               
    def forward(self, x: Tensor) -> Tensor:
        
        for i in range(16):
            if i in [1,3,7,11,15]:
                x = x.conv2d(*self.features[i], padding=1).relu().max_pool2d(kernel_size=(2,2))
            else:
                x = x.conv2d(*self.features[i], padding=1).relu()
        
        x = x.avg_pool2d(kernel_size=(1,1))
        
        x = x.flatten(1)
        #x = x.reshape(shape=(-1, x.shape[1]))
        #NOTE change shapes idk
        x = x.linear(self.classifier[0][0].transpose(), self.classifier[0][1]).relu()
        x = x.dropout(0.5)
        
        x = x.linear(*self.classifier[1]).relu()
        x = x.dropout(0.5)

        x = x.linear(self.classifier[2][0].transpose(), self.classifier[2][1])

        return x.softmax()
      
    def fake_load(self):

        data = load_state_dict_from_url('https://download.pytorch.org/models/vgg19-dcbb9e9d.pth', progress=True)

        for k,v in data.items():
 
            if 'features' in k:    
                if 'weight' in k:
                    self.features_weight[k] = Tensor.glorot_uniform(*v.detach().numpy().shape).assign(v.detach().numpy())
                else:
                    self.features_bias[k] = Tensor.zeros(*v.detach().numpy().shape).assign(v.detach().numpy())
            else:  
                if 'weight' in k:
                    self.classifier_weight[k] = Tensor.glorot_uniform(*v.detach().numpy().shape).assign(v.detach().numpy())
                else:
                    self.classifier_bias[k] = Tensor.zeros(*v.detach().numpy().shape).assign(v.detach().numpy())

        self.features = tuple(zip(self.features_weight.values(), self.features_bias.values()))
        self.classifier = tuple(zip(self.classifier_weight.values(), self.classifier_bias.values()))

def infer(model, img):
  # preprocess image
  aspect_ratio = img.size[0] / img.size[1]
  img = img.resize((int(224*max(aspect_ratio,1.0)), int(224*max(1.0/aspect_ratio,1.0))))

  img = np.array(img)
  y0,x0=(np.asarray(img.shape)[:2]-224)//2
  retimg = img = img[y0:y0+224, x0:x0+224]

  # if you want to look at the image
  '''
  import matplotlib.pyplot as plt
  plt.imshow(img)
  plt.show()
  '''
  # low level preprocess
  img = np.moveaxis(img, [2,0,1], [0,1,2])
  img = img.astype(np.float32)[:3].reshape(1,3,224,224)
  img /= 255.0
  img -= np.array([0.485, 0.456, 0.406]).reshape((1,-1,1,1))
  img /= np.array([0.229, 0.224, 0.225]).reshape((1,-1,1,1))

  # run the net
  out = model.forward(Tensor(img)).cpu()

  # if you want to look at the outputs
  '''
  import matplotlib.pyplot as plt
  plt.plot(out.data[0])
  plt.show()
  '''
  return out, retimg

model = VGG19()
model.fake_load()

response = requests.get('https://raw.githubusercontent.com/srirammanikumar/DogBreedClassifier/master/images/Labrador_retriever_06457.jpg')
labels = requests.get('https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt').text.split('\n')

img = Image.open(BytesIO(response.content))
out, _ = infer(model, img)
print(np.argmax(out.data), np.max(out.data)*100, labels[np.argmax(out.data)])
