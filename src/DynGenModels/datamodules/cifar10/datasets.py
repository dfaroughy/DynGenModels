import torch
import numpy as np
from torch.utils.data import Dataset
from dataclasses import dataclass
from torchvision import datasets, transforms

class CIFARDataset(Dataset):

    def __init__(self, config: dataclass):
        self.config = config
        self.img_transform= transforms.Compose([transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
                                                 )
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
        if self.config.DATA_TARGET == 'cifar10':

            train = datasets.CIFAR10(root='../../data', 
                                     train=True, 
                                     download=True, 
                                     transform=self.img_transform)
            self.target_label = train.targets 
            self.target = train.data.transpose((0, 3, 1, 2)).astype(np.float32) / 255.0
            self.target = torch.tensor(self.target)

        elif self.config.DATA_TARGET == 'cifar100':
            train = datasets.CIFAR100(root='../../data', 
                                      train=True, 
                                      download=True, 
                                      transform=self.img_transform)
            self.target_label = train.targets 
            self.target = train.data.transpose((0, 3, 1, 2)).astype(np.float32) / 255.0
            self.target = torch.tensor(self.target)
        else:
            raise ValueError('Invalid target dataset')

    def get_source_data(self):
        if self.config.DATA_SOURCE == 'cifar10':
            train = datasets.CIFAR10(root='../../data', 
                                     train=True, 
                                     download=True, 
                                     transform=self.img_transform)
            test = datasets.CIFAR10(root='../../data', 
                                    train=False, 
                                    download=True, 
                                    transform=self.img_transform)
            self.source_label = train.targets 
            self.source_test_label = test.targets 
            self.source = train.data.transpose((0, 3, 1, 2)).astype(np.float32) / 255.0
            self.source = torch.tensor(self.source)
            self.source_test = test.data.transpose((0, 3, 1, 2)).astype(np.float32) / 255.0
            self.source_test = torch.tensor(self.source_test)

        elif self.config.DATA_SOURCE == 'noise':
            self.source = torch.rand_like(self.target)
            self.source_test = torch.rand_like(self.target)
            self.source_label = [0] * len(self.source)
            self.source_test_label = [0] * len(self.source_test)

        else:
            raise ValueError('Invalid source dataset')
