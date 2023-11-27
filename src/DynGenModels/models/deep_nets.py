import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from DynGenModels.models.utils import get_activation_function, transformer_timestep_embedding, loretz_scalar_embedding

#...Multi-Layer Perceptron architecture:


class MLP(nn.Module):

    def __init__(self, 
                 configs):
        
        super().__init__()
        self.device = configs.DEVICE
        self.define_deep_models(configs)
        self.init_weights()
        self.to(self.device)

    def define_deep_models(self, configs):
        self.dim_input = configs.dim_input
        self.dim_output = configs.dim_input
        self.dim_hidden = configs.dim_hidden 
        self.dim_time_emb = configs.dim_time_emb
        self.num_layers = configs.num_layers
        self.act_fn = get_activation_function(configs.activation)
        # layers:
        layers = [nn.Linear(self.dim_input + 3 + self.dim_time_emb, self.dim_hidden), self.act_fn]
        for _ in range(self.num_layers - 2):
            layers.extend([nn.Linear(self.dim_hidden, self.dim_hidden), self.act_fn])
        layers.append(nn.Linear(self.dim_hidden, self.dim_output))
        self.model = nn.Sequential(*layers)

    def forward(self, t, x, context=None, mask=None):
        time_embeddings = transformer_timestep_embedding(t.squeeze(1), embedding_dim=self.dim_time_emb) if t is not None else t
        x = loretz_scalar_embedding(x)
        x = torch.concat([x, time_embeddings], dim=1)
        x = x.to(self.device)
        return self.model(x)

    def init_weights(self):
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)


# class _MLP(torch.nn.Module):
#     def __init__(self, 
#                  dim_input, 
#                  dim_output=None, 
#                  dim_hidden=64, 
#                  dim_time_emb=16,
#                  num_layers=3,
#                  activation=torch.nn.ReLU()
#                  ):
#         super().__init__()

#         if dim_output is None: dim_output = dim_input
        
#         layers = [torch.nn.Linear(dim_input + dim_time_emb, dim_hidden), activation]
#         for _ in range(num_layers-1): layers.extend([torch.nn.Linear(dim_hidden, dim_hidden), activation])
#         layers.append(torch.nn.Linear(dim_hidden, dim_output))
#         self.net = torch.nn.Sequential(*layers)
#     def forward(self, x):
#         return self.net(x)
    
# class MLP(nn.Module):
#     ''' Wrapper class for the MLP architecture
#     '''
#     def __init__(self, configs):
#         super(MLP, self).__init__()
#         self.device = configs.DEVICE
#         self.mlp = _MLP(dim_input=configs.dim_input, 
#                         dim_output=None,
#                         dim_hidden=configs.dim_hidden, 
#                         dim_time_emb=configs.dim_tim_emb,
#                         num_layers=configs.num_layers)
                        
#     def forward(self, t, x, context=None, mask=None, sampling=False):
#         t = time_embedding(t,)
#         x = torch.cat([x, t], dim=-1)
#         x = x.to(self.device)
#         self.mlp = self.mlp.to(self.device)
#         return self.mlp.forward(x)
    
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
