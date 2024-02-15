import torch
import numpy as np
from torch.utils.data import Dataset
from dataclasses import dataclass
from torchvision import datasets, transforms
from PIL import Image

class MNISTDataset(Dataset):

    def __init__(self, config: dataclass):
        self.config = config
        self.get_target_data()
        self.get_source_data()

    def __getitem__(self, idx):
        output = {}
        output['target'] = self.target[idx]
        output['source'] = self.source[idx]
        output['context'] = self.target_label[idx]
        output['mask'] = torch.ones_like(self.target[idx][..., 0])
        return output

    def __len__(self):
        return len(self.target)
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def get_target_data(self):
        if self.config.DATA_TARGET == 'mnist':
            train = datasets.MNIST(root='../../data', 
                                   train=True, 
                                   download=True, 
                                   transform=transforms.ToTensor())
            self.target_label = train.targets.tolist() 
            self.target = train.data.unsqueeze(1).float() / 255.0 

        elif self.config.DATA_TARGET == 'emnist':
            train = datasets.EMNIST(root='../../data', 
                                    split='letters', 
                                    train=True, 
                                    download=True, 
                                    transform=transforms.ToTensor())  
            self.target_label = train.targets.tolist()
            train = train.data.unsqueeze(1).float() / 255.0 
            train = torch.flip(train, dims=[3])
            self.target = torch.rot90(train, 1, (2, 3))
             
        elif self.config.DATA_TARGET == 'fashion':
            train = datasets.FashionMNIST(root='../../data', 
                                        train=True, 
                                        download=True, 
                                        transform=transforms.ToTensor())
            self.target_label = train.targets.tolist()
            self.target = train.data.unsqueeze(1).float() / 255.0

    def get_source_data(self):
        if self.config.DATA_SOURCE == 'mnist':
            train = datasets.MNIST(root='../../data', 
                                   train=True, 
                                   download=True, 
                                   transform=transforms.ToTensor())
            test = datasets.MNIST(root='../../data', 
                                  train=False, 
                                  download=True, 
                                  transform=transforms.ToTensor())
            self.source_label = train.targets.tolist() 
            self.source_test_label = test.targets.tolist() 
            self.source = train.data.unsqueeze(1).float() / 255.0 
            self.source_test = test.data.unsqueeze(1).float() / 255.0

        elif self.config.DATA_SOURCE == 'emnist':
            train = datasets.EMNIST(root='../../data', 
                                          split='letters', 
                                          train=True, 
                                          download=True, 
                                          transform=transforms.ToTensor())
            test = datasets.EMNIST(root='../../data', 
                                          split='letters', 
                                          train=False, 
                                          download=True, 
                                          transform=transforms.ToTensor())
            self.source_label = train.targets.tolist() 
            self.source_test_label = test.targets.tolist()
            train = train.data.unsqueeze(1).float() / 255.0 
            train = torch.flip(train, dims=[3])
            self.source = torch.rot90(train, 1, (2, 3))
            test = test.data.unsqueeze(1).float() / 255.0
            test = torch.flip(test, dims=[3])
            self.source_test = torch.rot90(test, 1, (2, 3))

        elif self.config.DATA_SOURCE == 'fashion':
            train = datasets.FashionMNIST(root='../../data', 
                                          train=True, 
                                          download=True, 
                                          transform=transforms.ToTensor())
            test = datasets.FashionMNIST(root='../../data', 
                                         train=False, 
                                         download=True, 
                                         transform=transforms.ToTensor())
            self.source_label = train.targets.tolist() 
            self.source_test_label = test.targets.tolist() 
            self.source = train.data.unsqueeze(1).float() / 255.0 
            self.source_test = test.data.unsqueeze(1).float() / 255.0

        elif self.config.DATA_SOURCE == 'distorted_mnist':

            train = datasets.MNIST(root='../../data', 
                                   train=True, 
                                   download=True, 
                                   transform=transforms.ToTensor())                  
            test = datasets.MNIST(root='../../data', 
                                  train=False, 
                                  download=True, 
                                  transform=transforms.ToTensor())
            
            self.source_label = train.targets.tolist() 
            self.source = train.data.unsqueeze(1) 
            self.source_test = test.data.unsqueeze(1)
            self.source_test_label = test.targets.tolist() 
            
            mask = torch.triu(torch.ones(28, 28), diagonal=0).unsqueeze(0).unsqueeze(0)
            mask = mask.expand_as(self.source)  
            
            self.source = self.source * mask
            mask = torch.triu(torch.ones(28, 28), diagonal=0).unsqueeze(0).unsqueeze(0)
            mask = mask.expand_as(self.source_test)  
            self.source_test = self.source_test * mask

        elif self.config.DATA_SOURCE == 'noise':
            self.source = torch.rand_like(self.target)
            self.source_test = torch.rand_like(self.target)
            self.source_label = [0] * len(self.source)
            self.source_test_label = [0] * len(self.source_test)

        else:
            raise ValueError('Invalid source dataset')
