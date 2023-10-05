import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

#...Multi-Layer Perceptron architecture:

class _MLP(torch.nn.Module):
    def __init__(self, dim, out_dim=None, w=64, time_varying=False, num_layers=3, device='cpu'):
        super().__init__()
        
        self.time_varying = time_varying
        if out_dim is None: out_dim = dim
        
        layers = [torch.nn.Linear(dim + (1 if time_varying else 0), w), torch.nn.SELU()]
        for _ in range(num_layers-1): layers.extend([torch.nn.Linear(w, w), torch.nn.SELU()])
        layers.append(torch.nn.Linear(w, out_dim))
        self.net = torch.nn.Sequential(*layers)
        self.to(device)
    def forward(self, x):
        return self.net(x)
    
class MLP(nn.Module):
    ''' Wrapper class for the MLP architecture
    '''
    def __init__(self, config):
        super(MLP, self).__init__()

        self.mlp = _MLP(dim=config.dim_input, 
                       out_dim=None,
                       w=config.dim_hidden, 
                       num_layers=config.num_layers,
                       time_varying=True,
                       device=config.device)
                
    def forward(self, t, x, context=None, mask=None):
        x = torch.cat([x, t], dim=-1)
        return self.mlp.forward(x)
    
#...ResNet architecture:

class ResidualBlock(torch.nn.Module):
    def __init__(self, dim, num_layers_per_block=2):
        super(ResidualBlock, self).__init__()

        layers = []
        for _ in range(num_layers_per_block):
            layers.extend([
                torch.nn.Linear(dim, dim),
                torch.nn.SELU()
            ])
        self.block = torch.nn.Sequential(*layers[:-1])

    def forward(self, x):
        return x + self.block(x)
    
class _ResNet(torch.nn.Module):
    def __init__(self, dim, out_dim=None, w=64, time_varying=False, num_blocks=3, num_layers_per_block=2, device='cpu'):
        super(_ResNet, self).__init__()

        if out_dim is None:
            out_dim = dim

        self.input_layer = torch.nn.Sequential(
            torch.nn.Linear(dim + (1 if time_varying else 0), w),
            torch.nn.SELU()
        )
        self.blocks = torch.nn.ModuleList([ResidualBlock(w, num_layers_per_block) for _ in range(num_blocks)])
        self.output_layer = torch.nn.Linear(w, out_dim)
        self.to(device)

    def forward(self, x):
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        return self.output_layer(x)

class ResNet(nn.Module):
    ''' Wrapper class for the ResNet architecture
    '''
    def __init__(self, config):
        super(ResNet, self).__init__()

        self.resnet = _ResNet(dim=config.dim_input, 
                            out_dim=None,
                            w=config.dim_hidden, 
                            num_blocks=config.num_blocks,
                            num_layers_per_block=config.num_block_layers,
                            time_varying=True,
                            device=config.device)
                
    def forward(self, t, x, context=None, mask=None):
        x = torch.cat([x, t], dim=-1)
        return self.resnet.forward(x)
