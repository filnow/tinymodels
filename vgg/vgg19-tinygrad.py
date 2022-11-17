from tinygrad.tensor import Tensor 
import torch
from torch.hub import load_state_dict_from_url
from PIL import Image
from torchvision import transforms
import requests
from io import BytesIO
from extra.utils import get_parameters, get_child, fake_torch_load, fetch

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
                x = x.conv2d(*self.features[i], padding=1).relu().max_pool2d
            else:
                x = x.conv2d(*self.features[i], padding=1).relu()
        
        #x = x.avg_pool2d(kernel_size=(1,1))
 
        x = x.flatten(1)
        x = x.dropout(0.5)
        
        x = x.linear(self.classifier[0][0].transpose(), self.classifier[0][1]).relu()
        x = x.dropout(0.5)
        
        x = x.linear(*self.classifier[1]).relu()
        x = x.linear(self.classifier[2][0].transpose(), self.classifier[2][1])

        return x
      
    def fake_load(self):

        data = load_state_dict_from_url('https://download.pytorch.org/models/vgg19-dcbb9e9d.pth', progress=True)

        for k,v in data.items():
            
            if 'features' in k:    
                if 'weight' in k:
                    self.features_weight[k] = Tensor.uniform(*v.detach().numpy().shape).assign(v.detach().numpy())
                else:
                    self.features_bias[k] = Tensor.zeros(*v.detach().numpy().shape).assign(v.detach().numpy())
            else:  
                if 'weight' in k:
                    self.classifier_weight[k] = Tensor.uniform(*v.detach().numpy().shape).assign(v.detach().numpy())
                else:
                    self.classifier_bias[k] = Tensor.zeros(*v.detach().numpy().shape).assign(v.detach().numpy())

        self.features = tuple(zip(self.features_weight.values(), self.features_bias.values()))
        self.classifier = tuple(zip(self.classifier_weight.values(), self.classifier_bias.values()))

m = VGG19()
m.fake_load()

#'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'

transform= transforms.Compose([            
 
 transforms.Resize(256),                    
 transforms.CenterCrop(224),                
 transforms.ToTensor(),                     
 transforms.Normalize(                      
 mean=[0.485, 0.456, 0.406],                
 std=[0.229, 0.224, 0.225]                  
 
 )])

#'https://raw.githubusercontent.com/srirammanikumar/DogBreedClassifier/master/images/Labrador_retriever_06457.jpg'
response = requests.get('https://raw.githubusercontent.com/filnow/tinymodels/main/data/dog3.jpg')
img = Image.open(BytesIO(response.content))
img_t = transform(img)
batch_t = torch.unsqueeze(img_t, 0)
out = m.forward(Tensor(batch_t.detach().numpy()))
labels = requests.get('https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt').text.split('\n')
#index = int(Tensor.max(out).numpy())
_, index = torch.max(torch.tensor(out.numpy()), 1)
#percentage = out.softmax()[0] * 100
percentage = torch.nn.functional.softmax(torch.tensor(out.numpy()), dim=1)[0] * 100

print(labels[index[0]], percentage[index[0]].item())

