import torch
import numpy as np
from torch.utils.data import Dataset
from dataclasses import dataclass
from torchvision import datasets, transforms
# from PIL import Image
# from torchvision.transforms import ToPILImage

from DynGenModels.datamodules.toys.dataprocess import PreProcessGaussData


class MNIST_Dataset(Dataset):

    def __init__(self, configs: dataclass):
        self.configs = configs
        self.get_target_data()
        self.get_source_data()

    def __getitem__(self, idx):
        output = {}
        output['target'] = self.target[idx]
        output['source'] = self.source[idx]
        output['mask'] = torch.ones_like(self.target[idx][..., 0])
        output['context'] = torch.empty_like(self.target[idx][..., 0])
        return output

    def __len__(self):
        return len(self.target)
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def get_target_data(self):
        
        if self.configs.DATA_TARGET == 'mnist':
            self.data_1 = datasets.MNIST(root='../data', train=True, download=True, transform=transforms.ToTensor())
        
        elif self.configs.DATA_TARGET == 'emnist':
            self.data_1 = datasets.EMNIST(root='./data', split='letters', train=True, download=True, transform=transforms.ToTensor())
        
        elif self.configs.DATA_TARGET == 'fashion':
             self.data_1 = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
        
        self.target = [d[0] for d in self.data_1]
        self.target_label = [d[1] for d in self.data_1]
        
    def get_source_data(self):

        if self.configs.DATA_SOURCE == 'mnist':
            self.data_0 = datasets.MNIST(root='../data', train=True, download=True, transform=transforms.ToTensor())
       
        elif self.configs.DATA_SOURCE == 'emnist':
            self.data_0 = datasets.EMNIST(root='./data', split='letters', train=True, download=True, transform=transforms.ToTensor())
       
        elif self.configs.DATA_SOURCE == 'fashion':
             self.data_0 = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
        else:
            self.data_0 = self.data_1

        self.source = [d[0] for d in self.data_0] if self.configs.DATA_SOURCE is not None else [torch.rand_like(d[0]) for d in self.data_0]
        self.source_label = [d[1] for d in self.data_0] if self.configs.DATA_SOURCE is not None else [0] * len(self.data_0)

