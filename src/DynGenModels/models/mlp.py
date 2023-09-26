import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

class _MLP(torch.nn.Module):
    def __init__(self, dim, out_dim=None, w=64, time_varying=False, device='cpu'):
        super().__init__()

        self.time_varying = time_varying
        if out_dim is None: out_dim = dim

        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim + (1 if time_varying else 0), w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, out_dim)).to(device)

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
                       time_varying=True,
                       device=config.device)
                
    def forward(self, t, x, context=None, mask=None):
        x = torch.cat([x, t], dim=-1)
        return self.mlp.forward(x)