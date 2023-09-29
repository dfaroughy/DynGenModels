import torch

from nflows.utils import torchutils
from nflows.flows.base import Flow
from nflows.transforms.base import CompositeTransform
from nflows.distributions.normal import StandardNormal
from nflows.transforms.permutations import ReversePermutation
from dataclasses import dataclass

class NormalizingFlow:
    def __init__(self, net, configs: dataclass):
        self.permutation = ReversePermutation(features=configs.dim_input)
        self.net = net
        self.num_transforms = configs.num_transforms
        self.transform = self.transforms()
        self.flow = Flow(self.transform, StandardNormal(shape=[configs.dim_input]))

    def transforms(self):
        transforms = []
        for _ in range(self.num_transforms):
            transforms.append(self.permutation)
            transforms.append(self.net)
        return CompositeTransform(transforms)

    def loss(self, batch):
        ''' Negative Log-probability loss
        '''
        target = batch['target']
        loss = - self.flow.log_prob(target)
        return torch.mean(loss)


