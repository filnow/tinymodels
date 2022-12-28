#TODO: write a test base on pytorch implementations of models and my models

from models import *
from torch.hub import load_state_dict_from_url
import unittest
import torch

def test_model(model_class, data_url):
    class TestModel(unittest.TestCase):
        def setUp(self):
            # Load the model and weights
            self.data = load_state_dict_from_url(data_url)
            self.model = model_class()
            self.model.load_state_dict(self.data)
            self.model.eval()

        def test_predict(self):
            # Generate some test data
            x_test = torch.rand((64, 3, 224, 224))

            # Make predictions with the model
            y_pred = self.model(x_test)

            # Check that the shape of the predictions is correct
            self.assertEqual(y_pred.shape, (64, 1000))

    return TestModel

TestAlexNet = test_model(AlexNet, 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth')
TestGoogleNet = test_model(GoogleNet, 'https://download.pytorch.org/models/googlenet-1378be20.pth')
TestVGG = test_model(VGG, 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth')
TestResNet = test_model(ResNet, 'https://download.pytorch.org/models/resnet50-0676ba61.pth')
TestInceptionV3 = test_model(InceptionV3, 'https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth')

if __name__ == '__main__':
    unittest.main()

