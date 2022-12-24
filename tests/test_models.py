#TODO: write a test base on pytorch implementations of models and my models
from models import *
from torch.hub import load_state_dict_from_url
import unittest
import torch

class TestAlexNet(unittest.TestCase):
    def setUp(self):
        # Load the model and weights
        self.data = load_state_dict_from_url('https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth')
        self.model = AlexNet()
        self.model.load_state_dict(self.data)
        self.model.eval()

    def test_predict(self):
        # Generate some test data
        x_test = torch.rand((64, 3, 224, 224))

        # Make predictions with the model
        y_pred = self.model(x_test)

        # Check that the shape of the predictions is correct
        self.assertEqual(y_pred.shape, (64, 1000))

class TestGoogleNet(unittest.TestCase):
    def setUp(self):
        # Load the model and weights
        self.data = load_state_dict_from_url('https://download.pytorch.org/models/googlenet-1378be20.pth')
        self.model = GoogleNet()
        self.model.load_state_dict(self.data)
        self.model.eval()

    def test_predict(self):
        # Generate some test data
        x_test = torch.rand((64, 3, 224, 224))

        # Make predictions with the model
        y_pred = self.model(x_test)

        # Check that the shape of the predictions is correct
        self.assertEqual(y_pred.shape, (64, 1000))
    
if __name__ == '__main__':
    unittest.main()
