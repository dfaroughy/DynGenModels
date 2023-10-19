import torch
from dataclasses import dataclass

class NormalizingFlow:

    def __init__(self, configs: dataclass):
        self.dim = configs.dim_input
        self.device = configs.DEVICE
        self.num_transforms = configs.num_transforms

    def loss(self, model, batch):
        ''' Negative Log-probability loss
        '''
        target = batch['target'].to(self.device)
        loss = - model.log_prob(target)
        return torch.mean(loss)


# class NormalizingFlow:

#     def __init__(self, net, configs: dataclass):
#         self.dim = configs.dim_input
#         self.device = configs.DEVICE
#         self.num_transforms = configs.num_transforms
#         self.flow_net = net
#         self.get_permutation(configs.permutation)
#         self.transforms()
#         self.flows = CompositeTransform(self.transforms).to(configs.DEVICE)
#         self.base_distribution = StandardNormal(shape=[configs.dim_input]).to(configs.DEVICE)
#         self.net = Flow(self.flows, self.base_distribution)

#     def get_permutation(self, perm):
#         k = list(range(self.dim))
#         if 'cycle' in perm:
#             N = int(perm.split('-')[0]) 
#             assert N < self.dim 
#             self.permutation = Permutation(torch.tensor(k[-N:] + k[:-N]))
#         elif 'reverse' in perm:
#             self.permutation = Permutation(torch.tensor(k[::-1]))

#     def transforms(self):
#         self.transforms = []
#         for _ in range(self.num_transforms):
#             net = deepcopy(self.flow_net)
#             perm = deepcopy(self.permutation)
#             self.transforms.append(net)
#             self.transforms.append(perm)

#     def loss(self, batch):
#         ''' Negative Log-probability loss
#         '''
#         target = batch['target'].to(self.device)
#         loss = - self.net.log_prob(target)
#         return torch.mean(loss)


