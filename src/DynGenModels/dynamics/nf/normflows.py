import torch

from nflows.utils import torchutils
from nflows.flows.base import Flow
from nflows.transforms.base import CompositeTransform
from nflows.distributions.normal import StandardNormal
from nflows.transforms.permutations import ReversePermutation
from dataclasses import dataclass
from copy import deepcopy

class NormalizingFlow:

    def __init__(self, net, configs: dataclass):
        self.num_transforms = configs.num_transforms
        self.flow_net = net
        self.permutation = ReversePermutation(features=configs.dim_input)
        self.transforms()
        self.flows = CompositeTransform(self.transforms)
        self.base_distribution = StandardNormal(shape=[configs.dim_input])
        self.net = Flow(self.flows, self.base_distribution)

    def transforms(self):
        self.transforms = []
        for _ in range(self.num_transforms):
            net = deepcopy(self.flow_net)
            perm = deepcopy(self.permutation)
            self.transforms.append(net)
            self.transforms.append(perm)

    def loss(self, batch):
        ''' Negative Log-probability loss
        '''
        target = batch['target']
        loss = - self.net.log_prob(target)
        return torch.mean(loss)


