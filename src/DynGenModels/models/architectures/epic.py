import torch
from torch import nn
from torch.nn import functional as F
from DynGenModels.models.architectures.utils import timestep_sinusoidal_embedding, GaussianFourierProjection, get_activation_function
import torch.nn.utils.weight_norm as WeightNorm

class EPiC(nn.Module):
    ''' Wrapper class for the EPiC architecture
    '''
    def __init__(self, config):
        super(EPiC, self).__init__()
        self.device = config.DEVICE

        if config.POOLING=='mean_sum':
            self.epic = EPiC_Network(dim_features = config.DIM_INPUT,
                                     dim_global = config.DIM_GLOBAL, 
                                     time_embedding = config.TIME_EMBEDDING,
                                     dim_time_embedding = config.DIM_TIME_EMB,
                                     dim_hidden = config.DIM_HIDDEN, 
                                     num_layers = config.NUM_EPIC_LAYERS,
                                     act_func = get_activation_function(config.ACTIVATION),
                                     dropout = config.DROPOUT,
                                     pool = config.POOLING,
                                     skip_connection = config.USE_SKIP_CONNECTIONS)
            
    def forward(self, t, x, mask=None):
        t = t.to(self.device)   
        x = x.to(self.device)
        self.epic = self.epic.to(self.device)
        return self.epic.forward(t, x)


class EPiC_Network(nn.Module):
    def __init__(self, 
                 dim_features = 3,
                 dim_global = 10,
                 time_embedding = 'sinusoidal',  
                 dim_time_embedding = 10,
                 dim_hidden = 256, 
                 num_layers = 3,
                 act_func = nn.LeakyReLU(),
                 dropout = 0.1,
                 pool = 'mean_sum',
                 skip_connection = False):
        
        super(EPiC_Network, self).__init__()
        self.time_embedding = time_embedding
        self.dim_time_embedding = dim_time_embedding
        self.num_layers = num_layers
        self.skip_connection = skip_connection

        #...projection network

        self.epic_proj = EPiC_Projection(dim_local=dim_features + dim_time_embedding,
                                         dim_global=dim_global,
                                         dim_hidden=dim_hidden,
                                         act_func=act_func,
                                         pool = pool,
                                         dropout=dropout)

        #...epic layers:

        self.epic_layers = nn.ModuleList() #dim_loc, dim_glob, dim_hidden, dim_context
        for _ in range(num_layers):
            self.epic_layers.append(EPiC_layer(dim_loc=dim_hidden, 
                                               dim_glob=dim_global,
                                               dim_hidden=dim_hidden,
                                               dim_context=dim_time_embedding,
                                               act_func=act_func,
                                               pool = pool,
                                               dropout=dropout))
                                            
        #...output layer:

        self.output = nn.Linear(dim_hidden, dim_features)

    def forward(self, t, x):  
        if self.time_embedding == 'sinusoidal':
            t_emb = timestep_sinusoidal_embedding(t.squeeze(-1), self.dim_time_embedding, max_period=10000)
        elif self.time_embedding == 'gaussian':
            gaussian_emb = GaussianFourierProjection(self.dim_time_embedding, device=x.device)
            t_emb = gaussian_emb(t.squeeze(-1))
        t = t_emb.repeat(1, x.shape[1], 1)
        x_local = torch.cat([x, t], dim=-1)  

        #...local to global:
        x_global, x_local = self.epic_proj(x_local)
        if self.skip_connection:
            x_global_skip = x_global.clone() 
            x_local_skip = x_local.clone()

        #...equivariant layers:
        for i in range(self.num_layers):
            x_global, x_local = self.epic_layers[i](x_global, x_local, t)   
            if self.skip_connection:
                x_global += x_global_skip 
                x_local += x_local_skip
    
        output = self.output(x_local)
        
        return output     #[batch, points, feats]



# class SelfAttentionPooling(nn.Module):
#     def __init__(self, dim_input, dim_output):  
#         super(SelfAttentionPooling, self).__init__()
#         self.query = nn.Linear( dim_input,  dim_input)
#         self.key = nn.Linear( dim_input,  dim_input)
#         self.value =nn.Linear( dim_input,  dim_input)
#         self.fc_out = nn.Sequential(nn.Linear( dim_input, dim_output), nn.LayerNorm(dim_output))

#     def forward(self, x_local):
#         Q = self.query(x_local)
#         K = self.key(x_local)
#         V = self.value(x_local)
#         QK = torch.matmul(Q, K.transpose(2, 1))
#         A = torch.nn.functional.softmax(QK, dim=-1)
#         glob = torch.matmul(A, V).sum(dim=1)
#         glob = self.fc_out(glob)
#         return glob

class SelfAttention(nn.Module):
    def __init__(self, dim_input, dim_output):  
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(dim_input, dim_input)
        self.key = nn.Linear(dim_input, dim_input)
        self.value =nn.Linear(dim_input, dim_input)
        self.fc_out = nn.Sequential(nn.Linear(dim_input, dim_output), nn.LayerNorm(dim_output), nn.LeakyReLU())

    def forward(self, x_local):
        Q, K, V = self.query(x_local), self.key(x_local), self.value(x_local)
        QK = torch.matmul(Q, K.transpose(2, 1))
        softQK = torch.nn.functional.softmax(QK, dim=-1)
        loc = torch.matmul(softQK, V)
        loc = self.fc_out(loc)
        return loc

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, dim_input, dim_output, num_heads=10):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.dim_input = dim_input
        self.dim_per_head = dim_input // num_heads

        # Ensure the input dimension can be evenly divided by num_heads
        assert dim_input % num_heads == 0, "Dimension of input must be divisible by the number of heads"

        self.query = nn.Linear(dim_input, dim_input)
        self.key = nn.Linear(dim_input, dim_input)
        self.value = nn.Linear(dim_input, dim_input)

        self.fc_out = nn.Linear(dim_input, dim_output)
        self.norm = nn.LayerNorm(dim_output)

        # Residual connection and layer normalization for each sub-layer
        self.dropout = nn.Dropout(0.1)

    def split_heads(self, x, batch_size):
        # Split the last dimension into (num_heads, dim_per_head)
        x = x.view(batch_size, -1, self.num_heads, self.dim_per_head)
        # Transpose to get dimensions (batch_size, num_heads, seq_length, dim_per_head)
        return x.transpose(1, 2)

    def forward(self, x):
        batch_size = x.shape[0]

        # Apply the linear transformations and split into num_heads
        Q = self.split_heads(self.query(x), batch_size)
        K = self.split_heads(self.key(x), batch_size)
        V = self.split_heads(self.value(x), batch_size)

        # Perform scaled dot-product attention for each head
        scale = self.dim_per_head ** 0.5
        QK = torch.matmul(Q, K.transpose(-2, -1)) / scale
        softQK = F.softmax(QK, dim=-1)
        softQK = self.dropout(softQK)  # Apply dropout to the attention weights
        loc = torch.matmul(softQK, V)

        # Concatenate the heads back together
        loc = loc.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.dim_per_head)

        # Apply final linear transformation and residual connection
        loc = self.fc_out(loc)
        loc += x  # Add residual connection here (assume x has same shape as loc)
        loc = self.norm(loc)  # Apply layer normalization after adding the residual

        return loc


class EPiC_Projection(nn.Module):
    def __init__(self, dim_local, dim_global, dim_hidden, act_func=nn.LeakyReLU(), dropout=0.1, pool='mean_sum'):
        super(EPiC_Projection, self).__init__()

        self.pool = pool
        self.fc_local = nn.Sequential(WeightNorm(nn.Linear(dim_local, dim_hidden)),
                                      act_func)
        self.att_local = MultiHeadAttention(dim_hidden, dim_hidden)  # local projection_mlp
        dim_input = 2 * dim_hidden if pool=='mean_sum' else dim_hidden
        self.block_glob = nn.Sequential(nn.Linear(dim_input, dim_hidden),
                                        nn.BatchNorm1d(dim_hidden),  
                                        act_func,
                                        nn.Linear(dim_hidden, dim_global),
                                        act_func,
                                        nn.Dropout(p=dropout))
    def pooling(self, loc):
        if self.pool =='mean_sum': 
            x_mean = loc.mean(1, keepdim=False)
            x_sum = loc.sum(1, keepdim=False) 
            return torch.cat([x_mean, x_sum], dim=1) 
    
    def forward(self, x_local):
        loc = self.fc_local(x_local)
        loc = self.att_local(loc)  
        glob = self.pooling(loc)
        glob = self.block_glob(glob)  
        return glob, loc


# class EPiC_Projection(nn.Module):
#     def __init__(self, dim_local, dim_global, dim_hidden, act_func=nn.LeakyReLU(), dropout=0.1, pool='mean_sum'):
#         super(EPiC_Projection, self).__init__()

#         self.pool = pool

#         self.fc_local = nn.Sequential(WeightNorm(nn.Linear(dim_local, dim_hidden)),
#                                       act_func)
#         if pool =='attention':
#             self.attention = SelfAttentionPooling(dim_hidden, dim_hidden)
                                                                                 
#         dim_input = 1 * dim_hidden if pool=='mean_sum' else dim_hidden

#         self.block_glob = nn.Sequential(nn.Linear(dim_input, dim_hidden),
#                                         nn.BatchNorm1d(dim_hidden),  
#                                         act_func,
#                                         nn.Linear(dim_hidden, dim_global),
#                                         act_func,
#                                         nn.Dropout(p=dropout))
        
#     def pooling(self, loc):
#         if self.pool =='mean_sum': 
#             x_mean = loc.mean(1, keepdim=False)
#             x_sum = loc.sum(1, keepdim=False) 
#             return torch.cat([x_mean, x_sum], dim=1) 
#         elif self.pool =='attention':
#             return self.attention(loc)
    
#     def forward(self, x_local):
#         loc = self.fc_local(x_local)  
#         glob = self.attention(loc)
#         # print(glob.shape)

#         glob = self.block_glob(glob)  
#         return glob, loc


class EPiC_layer(nn.Module):

    # based on https://github.com/uhh-pd-ml/EPiC-GAN/blob/main/models.py

    def __init__(self, dim_loc, dim_glob, dim_hidden, dim_context, act_func=nn.LeakyReLU(), dropout=0.1, pool='mean_sum', use_second_block=False):
        super(EPiC_layer, self).__init__()

        self.pool = pool
        z = 2 if pool=='mean_sum' else 1

        #...global blocks:

        self.block_glob_1 = nn.Sequential(nn.Linear(dim_glob + int(z * dim_loc), dim_hidden),
                                          nn.BatchNorm1d(dim_hidden),  
                                          act_func,
                                          nn.Dropout(p=dropout),
                                          nn.Linear(dim_hidden, dim_glob), 
                                          nn.BatchNorm1d(dim_glob), 
                                          act_func, 
                                          nn.Dropout(p=dropout))

        self.block_glob_2 = nn.Sequential(nn.Linear(dim_glob, dim_hidden), 
                                          nn.BatchNorm1d(dim_hidden), 
                                          act_func, 
                                          nn.Dropout(p=dropout),
                                          nn.Linear(dim_hidden, dim_glob), 
                                          nn.BatchNorm1d(dim_glob), 
                                          act_func, 
                                          nn.Dropout(p=dropout)) if use_second_block else nn.Identity()
        
        #...local blocks:

        self.block_loc_1 = nn.Sequential(WeightNorm(nn.Linear(dim_loc + dim_glob + dim_context, dim_hidden)),  
                                         act_func,
                                         nn.Dropout(p=dropout),
                                         WeightNorm(nn.Linear(dim_hidden, dim_loc)), 
                                         act_func, 
                                         nn.Dropout(p=dropout))
        
        self.block_loc_2 = nn.Sequential(WeightNorm(nn.Linear(dim_loc, dim_hidden)), 
                                         act_func, 
                                         nn.Dropout(p=dropout),
                                         WeightNorm(nn.Linear(dim_hidden, dim_hidden)), 
                                         act_func, 
                                         nn.Dropout(p=dropout)) if  use_second_block else nn.Identity()

    def pooling(self, loc):
        x_mean = loc.mean(1, keepdim=False)
        x_sum = loc.sum(1, keepdim=False) 
        if self.pool =='mean': return x_mean
        elif self.pool =='sum': return x_sum
        elif self.pool =='mean_sum': return torch.cat([x_mean, x_sum], dim=1) 

    def localize(self, glob, num_points):
        dim_global = glob.size(1)
        return glob.view(-1,1,dim_global).repeat(1, num_points, 1)

    def forward(self, x_global, x_local, context):   # shapes: x_global.shape = [bs, dim_global], x_local.shape = [bs, num_points, dim_local]

        #...local to global:
        pool = self.pooling(x_local)
        glob = torch.cat([pool, x_global], dim=1) 
        glob = self.block_glob_1(glob)      
        glob = self.block_glob_2(glob + x_global) 
        
        #...global to local:
        loc = self.localize(glob, num_points=x_local.size(1))
        loc = torch.cat([loc, x_local, context], dim=2) 
        loc = self.block_loc_1(loc)      
        loc = self.block_loc_2(loc + x_local) 

        return glob, loc







# class EPiC_Projection(nn.Module):
#     def __init__(self, dim_latent_global, latent_local, dim_hidden):
#         super(EPiC_Projection, self).__init__()

#         self.local_0 = nn.Linear(latent_local, dim_hidden)  # local projection_mlp
#         self.global_0 = nn.Linear(2*dim_hidden, dim_hidden) # local to global projection_mlp
#         self.global_1 = nn.Linear(dim_hidden, dim_hidden)
#         self.global_2 = nn.Linear(dim_hidden, dim_latent_global)

#     def meansum_pooling(self, x_local):
#         x_mean = x_local.mean(1, keepdim=False)
#         x_sum = x_local.sum(1, keepdim=False) 
#         x_global = torch.cat([x_mean, x_sum], 1) 
#         return x_global

#     def forward(self, x_local):
#         x_local = F.leaky_relu(self.local_0(x_local)) 
#         x_global = self.meansum_pooling(x_local)
#         x_global = F.leaky_relu(self.global_0(x_global))      
#         x_global = F.leaky_relu(self.global_1(x_global))
#         x_global = F.leaky_relu(self.global_2(x_global))   
#         return x_global, x_local



# class EPiC_layer(nn.Module):

#     # modified from https://github.com/uhh-pd-ml/EPiC-GAN/blob/main/models.py

#     def __init__(self, dim_loc, dim_glob, dim_hidden, dim_context, act_func=nn.LeakyReLU(), dropout=0.1, pool='mean_sum'):
#         super(EPiC_layer, self).__init__()

#         self.pool = pool

#         #...global blocks:

#         self.res_block_glob_1 = nn.Sequential(nn.Linear(dim_glob + int(2 * dim_loc), dim_hidden),  
#                                               act_func,
#                                               nn.BatchNorm1d(dim_hidden), 
#                                               nn.Dropout(p=dropout),
#                                               nn.Linear(dim_hidden, dim_glob), 
#                                               act_func, 
#                                               nn.BatchNorm1d(dim_glob), 
#                                               nn.Dropout(p=dropout))
        
#         self.res_block_glob_2 = nn.Sequential(nn.Linear(dim_glob, dim_hidden), 
#                                               act_func, 
#                                               nn.BatchNorm1d(dim_hidden), 
#                                               nn.Dropout(p=dropout),
#                                               nn.Linear(dim_hidden, dim_glob), 
#                                               act_func, 
#                                               nn.BatchNorm1d(dim_glob), 
#                                               nn.Dropout(p=dropout))
        
#         #...local blocks:

#         self.res_block_loc_1 = nn.Sequential( WeightNorm(nn.Linear(dim_loc + dim_glob + dim_context, dim_hidden)),  
#                                               act_func,
#                                               nn.Dropout(p=dropout),
#                                               WeightNorm(nn.Linear(dim_hidden, dim_loc)), 
#                                               act_func, 
#                                               nn.Dropout(p=dropout))
        
#         self.res_block_loc_2 = nn.Sequential(WeightNorm(nn.Linear(dim_loc, dim_hidden)), 
#                                              act_func, 
#                                              nn.Dropout(p=dropout),
#                                              WeightNorm(nn.Linear(dim_hidden, dim_loc)), 
#                                              act_func, 
#                                              nn.Dropout(p=dropout))

#     def pooling(self, loc):
#         x_mean = loc.mean(1, keepdim=False)
#         x_sum = loc.sum(1, keepdim=False) 
#         if self.pool =='mean':
#             return x_mean
#         elif self.pool =='sum':
#             return x_sum
#         elif self.pool =='mean_sum':
#            return torch.cat([x_mean, x_sum], dim=1) 

#     def localize(self, glob, num_points):
#         dim_global = glob.size(1)
#         return glob.view(-1,1,dim_global).repeat(1, num_points, 1)

#     def forward(self, x_global, x_local, context):   # shapes: x_global.shape = [bs, dim_global], x_local.shape = [bs, num_points, dim_local]

#         #...local to global:
#         pool = self.pooling(x_local)
#         glob = torch.cat([pool, x_global], dim=1) 

#         #...global blocks:
#         glob1 = self.res_block_glob_1(glob)      
#         glob2 = self.res_block_glob_2(glob1 + x_global) 
#         glob = glob1 + glob2
        
#         #...global to local:
#         loc = self.localize(glob, num_points=x_local.size(1))
#         loc = torch.cat([loc, x_local, context], dim=2) 

#         #...local blocks:
#         loc1 = self.res_block_loc_1(loc)      
#         loc2 = self.res_block_loc_2(loc1 + x_local) 
#         loc = loc1 + loc2

#         return glob, loc
        




# class SelfAttentionPooling(nn.Module):
#     def __init__(self, in_dim, out_dim):
#         super(SelfAttentionPooling, self).__init__()
#         self.query = nn.Linear(in_dim, in_dim)
#         self.key = nn.Linear(in_dim, in_dim)
#         self.value =nn.Linear(in_dim, in_dim)
#         self.fc_out = nn.Sequential(nn.Linear(in_dim, out_dim),
#                                     nn.LayerNorm(out_dim))

#     def forward(self, x, mask=None):
#         Q = self.query(x)
#         K = self.key(x)
#         V = self.value(x)
#         QK = torch.matmul(Q, K.transpose(2, 1))
#         A = torch.nn.functional.softmax(QK, dim=-1)
#         glob = torch.matmul(A, V).sum(dim=1)
#         glob = self.fc_out(glob)
#         return glob






