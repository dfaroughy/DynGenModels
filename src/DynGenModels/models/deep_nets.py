import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

#...Multi-Layer Perceptron architecture:

class _MLP(torch.nn.Module):
    def __init__(self, dim, out_dim=None, w=64, time_varying=False, num_layers=3):
        super().__init__()
        self.time_varying = time_varying
        if out_dim is None: out_dim = dim
        layers = [torch.nn.Linear(dim + (1 if time_varying else 0), w), torch.nn.SELU()]
        for _ in range(num_layers-1): layers.extend([torch.nn.Linear(w, w), torch.nn.SELU()])
        layers.append(torch.nn.Linear(w, out_dim))
        self.net = torch.nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)
    
class MLP(nn.Module):
    ''' Wrapper class for the MLP architecture
    '''
    def __init__(self, configs):
        super(MLP, self).__init__()
        self.device = configs.DEVICE
        self.mlp = _MLP(dim=configs.dim_input, 
                       out_dim=None,
                       w=configs.dim_hidden, 
                       num_layers=configs.num_layers,
                       time_varying=True)
                        
    def forward(self, t, x, context=None, mask=None, sampling=False):
        x = torch.cat([x, t], dim=-1)
        x = x.to(self.device)
        self.mlp = self.mlp.to(self.device)
        return self.mlp.forward(x)
    
#...ResNet architecture:

class ResidualBlock(torch.nn.Module):
    def __init__(self, dim, num_layers_per_block=2):
        super(ResidualBlock, self).__init__()

        layers = []
        for _ in range(num_layers_per_block):
            layers.extend([torch.nn.Linear(dim, dim), torch.nn.SELU()])
        self.block = torch.nn.Sequential(*layers[:-1])

    def forward(self, x):
        return x + self.block(x)
    
class _ResNet(torch.nn.Module):
    def __init__(self, dim, out_dim=None, w=64, time_varying=False, num_blocks=3, num_layers_per_block=2):
        super(_ResNet, self).__init__()

        if out_dim is None:
            out_dim = dim

        self.input_layer = torch.nn.Sequential(
            torch.nn.Linear(dim + (1 if time_varying else 0), w),
            torch.nn.SELU()
        )
        self.blocks = torch.nn.ModuleList([ResidualBlock(w, num_layers_per_block) for _ in range(num_blocks)])
        self.output_layer = torch.nn.Linear(w, out_dim)

    def forward(self, x):
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        return self.output_layer(x)

class ResNet(nn.Module):
    ''' Wrapper class for the ResNet architecture
    '''
    def __init__(self, configs):
        super(ResNet, self).__init__()
        self.device = configs.DEVICE
        self.resnet = _ResNet(dim=configs.dim_input, 
                            out_dim=None,
                            w=configs.dim_hidden, 
                            num_blocks=configs.num_blocks,
                            num_layers_per_block=configs.num_block_layers,
                            time_varying=True)
                
    def forward(self, t, x, context=None, mask=None, sampling=False):
        x = torch.cat([x, t], dim=-1)
        x = x.to(self.device)
        self.resnet = self.resnet.to(self.device)
        return self.resnet.forward(x)
