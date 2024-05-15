# download Fashion MNIST
from torch.nn import Module, Conv2d, Linear, ReLU, MaxPool2d, CrossEntropyLoss, Sequential, Module, Softmax, LogSoftmax
from torchvision import datasets, transforms
import torch
import matplotlib.pyplot as plt
import numpy as np


class DatasetLoader:
    def __init__(self, batch_size, baseName):
        self.batch_size = batch_size
        self.baseName = baseName
        self.train_loader = None
        self.test_loader = None
        

    def load(self):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        trainset = self.baseName('data', download=False, train=True, transform=transform)
        testset = self.baseName('data', download=False, train=False, transform=transform)
        self.train_loader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size, shuffle=False)
        return self.train_loader, self.test_loader
    
    def getLayerSize(self):
        dataiter = iter(self.train_loader)
        input, output = next(dataiter)
        return input[0].size(), len(output.unique())



# class ConvolutionNeuralNetwork(Module):
#     def __init__(self, inputSize, outputSize):
#         super(ConvolutionNeuralNetwork, self).__init__()
#         self.conv1 = Conv2d(1, 32, 3)
#         self.conv2 = Conv2d(32, 64, 3)
#         self.pool = MaxPool2d(2, 2)
#         self.fcIn = Linear(inputSize, 128)
#         self.fcOut = Linear(128, outputSize)
#         self.relu = ReLU()
#         self.softmax = Softmax(dim=1)
    
    
    


if __name__ == "__main__":
    fashion_mnist = DatasetLoader(64, datasets.FashionMNIST)
    train_loader, test_loader = fashion_mnist.load()
    print(fashion_mnist.getLayerSize())