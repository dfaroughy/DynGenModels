import torch
from torch import nn
from DynGenModels.models.architectures.utils import (fc_block, get_activation_function, 
                                                     timestep_sinusoidal_embedding, 
                                                     GaussianFourierProjection)

class DeepSets(nn.Module):
    ''' Wrapper class for the DeepSets architecture
    '''
    def __init__(self, config):
        super(DeepSets, self).__init__()
        self.device = config.DEVICE
        self.deepsets = DeepSets_Network(dim_features=config.DIM_INPUT, 
                                         dim_hidden=config.DIM_HIDDEN, 
                                         dim_time_embedding=config.DIM_TIME_EMB,
                                         num_layers_phi=config.NUM_LAYERS_PHI,
                                         num_layers_rho=config.NUM_LAYERS_RHO,
                                         dropout=config.DROPOUT,
                                         activation=get_activation_function(config.ACTIVATION),
                                         pool=config.POOLING)
                        
    def forward(self, t, x, mask=None):
        t = t.to(self.device)   
        x = x.to(self.device)
        self.deepsets = self.deepsets.to(self.device)
        return self.deepsets.forward(t, x)


class DeepSets_Network(torch.nn.Module):
    def __init__(self, 
                 dim_features, 
                 dim_hidden=64, 
                 activation = torch.nn.SELU(),
                 time_embedding='sinusoidal',
                 dim_time_embedding=10,
                 num_layers_phi=3,
                 num_layers_rho=3,
                 dropout=0.1,
                 pool='sum'
                 ):
        
        super(DeepSets_Network).__init__()
        self.pool = pool
        self.time_embedding = time_embedding
        self.dim_time_embedding = dim_time_embedding
        factor = 3 if pool == 'mean_sum' else 2  

        self.phi = fc_block(dim_input = dim_features + dim_time_embedding, 
                            dim_output = dim_hidden, 
                            dim_hidden = dim_hidden, 
                            num_layers = num_layers_phi,
                            activation = activation, 
                            dropout = dropout)
        
        self.rho = fc_block(dim_input = factor * dim_hidden, 
                            dim_hidden = dim_hidden,
                            dim_output = dim_features, 
                            num_layers = num_layers_rho,
                            activation = activation, 
                            dropout = dropout)

    def forward(self, t, x):

        if self.time_embedding == 'sinusoidal':
            t_emb = timestep_sinusoidal_embedding(t, self.dim_time_embedding, max_period=10000)
        elif self.time_embedding == 'gaussian':
            gaussian_emb = GaussianFourierProjection(self.dim_time_embedding, device=x.device)
            t_emb = gaussian_emb(t)

        t = t_emb.repeat(1, x.shape[1], 1)
        x = torch.cat([x, t], dim=-1)
        h = self.phi(x) 
        h_sum, h_mean = h.sum(1, keepdim=False), h.mean(1, keepdim=False) 

        if self.pool == 'sum':  h_pool = h_sum  
        elif self.pool == 'mean':  h_pool = h_mean 
        elif self.pool == 'mean_sum': h_pool = torch.cat([h_mean, h_sum], dim=1)

        h_pool_repeated = h_pool.unsqueeze(1).repeat(1, x.shape[1], 1)
        enhanced_features = torch.cat([h, h_pool_repeated], dim=-1)
        return self.rho(enhanced_features)

