import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

import torch.nn.utils.weight_norm as weight_norm


class DeepSets(nn.Module):
    ''' Wrapper class for the DeepSets architecture
    '''
    def __init__(self, config):
        super(DeepSets, self).__init__()
        self.device = config.DEVICE
        self.deepsets = _DeepSets(dim=config.DIM_INPUT, 
                                  dim_hidden=config.DIM_HIDDEN, 
                                  num_layers_1=config.NUM_LAYERS_1,
                                  num_layers_2=config.NUM_LAYERS_2,
                                  pool=config.POOLING,
                                  time_varying=True)
                        
    def forward(self, t, x, mask):
        t = t.unsqueeze(1).repeat(1, x.shape[1], 1)
        x = torch.cat([x, t], dim=-1)
        x = x.to(self.device)
        mask = mask.to(self.device) if mask is not None else None
        self.deepsets = self.deepsets.to(self.device)
        return self.deepsets.forward(x, mask)


class _DeepSets(torch.nn.Module):
    def __init__(self, 
                 dim, 
                 dim_hidden=64, 
                 num_layers_1=3,
                 num_layers_2=3,
                 pool='sum',
                 time_varying=False, 
                 ):
        
        super().__init__()
        
        self.pool = pool
        self.time_varying = time_varying
        s = 3 if pool == 'mean_sum' else 2     
    
        phi_layers = [torch.nn.Linear(dim + (1 if time_varying else 0), dim_hidden), torch.nn.SELU()]
        for _ in range(num_layers_1-1): 
            phi_layers.extend([torch.nn.Linear(dim_hidden, dim_hidden), torch.nn.SELU()])
        phi_layers.append(torch.nn.Linear(dim_hidden, dim_hidden))
        self.phi = torch.nn.Sequential(*phi_layers)

        rho_layers = [torch.nn.Linear(s * dim_hidden, dim_hidden), torch.nn.SELU()]
        for _ in range(num_layers_2-1): 
            rho_layers.extend([torch.nn.Linear(dim_hidden, dim_hidden), torch.nn.SELU()])
        rho_layers.append(torch.nn.Linear(dim_hidden, dim))
        self.rho = torch.nn.Sequential(*rho_layers)

    def forward(self, x, mask):
        h = self.phi(x)  # Apply phi to each point's features
        h_sum = (h * mask).sum(1, keepdim=False)   
        h_mean = h_sum / mask.sum(1, keepdim=False) 
        if self.pool == 'sum':  h_pool = h_sum  
        elif self.pool == 'mean':  h_pool = h_mean 
        elif self.pool == 'mean_sum': h_pool = torch.cat([h_mean, h_sum], dim=1)
        h_pool_repeated = h_pool.unsqueeze(1).repeat(1, x.shape[1], 1)
        enhanced_features = torch.cat([h, h_pool_repeated], dim=-1)
        f = self.rho(enhanced_features)
        return f


    # def forward(elf, x, mask):
    #     h = self.phi(x)
    #     h_sum = (h * mask).sum(1, keepdim=False)   
    #     h_mean = h_sum / mask.sum(1, keepdim=False)  
    #     print(1, h_sum.shape, h_mean.shape)
    #     if self.pool == 'sum': h_pool = h_sum  
    #     elif self.pool == 'mean': h_pool = h_mean 
    #     elif self.pool == 'mean_sum': h_pool = torch.cat([h_mean, h_sum], dim=1) 
    #     print(2, h_pool.shape)
    #     f = self.rho(h_pool)
    #     print(3, f.shape)
    #     return f                        
    


#...EPiC Network:

class EPiC(nn.Module):
    ''' Wrapper class for the EPiC architecture
    '''
    def __init__(self, config):
        super(EPiC, self).__init__()
        self.device = config.DEVICE

        if config.POOLING=='mean_sum':
            self.epic = EPiC_Network(feats = config.DIM_INPUT, 
                                    latent_global = config.DIM_GLOBAL,
                                    latent_local = config.DIM_INPUT,
                                    hid_d = config.DIM_HIDDEN, 
                                    equiv_layers = config.NUM_EPIC_LAYERS,
                                    time_varying=True)
            
        elif config.POOLING=='attention':
            self.epic = EPiC_Attention_Network(feats = config.DIM_INPUT,
                                               latent_global = config.DIM_GLOBAL,
                                               latent_local = config.DIM_INPUT,
                                               hid_d = config.DIM_HIDDEN, 
                                               equiv_layers = config.NUM_EPIC_LAYERS,
                                               time_varying=True)
                        
    def forward(self, t, x, mask=None, sampling=False):
        t = t.repeat(1, x.shape[1], 1) if not sampling else t
        x = torch.cat([x, t], dim=-1)
        x = x.to(self.device)
        mask = mask[..., None].to(self.device) if mask is not None else torch.ones_like(t).to(self.device)
        print(2, x.shape, t.shape)
        self.epic = self.epic.to(self.device)
        return self.epic.forward(x, mask)



class EPiC_layer(nn.Module):
    # from https://github.com/uhh-pd-ml/EPiC-GAN/blob/main/models.py
    def __init__(self, local_in_dim, hid_dim, latent_dim):
        super(EPiC_layer, self).__init__()
        self.fc_global1 = weight_norm(nn.Linear(int(2*hid_dim)+latent_dim, hid_dim)) 
        self.fc_global2 = weight_norm(nn.Linear(hid_dim, latent_dim)) 
        self.fc_local1 = weight_norm(nn.Linear(local_in_dim+latent_dim, hid_dim))
        self.fc_local2 = weight_norm(nn.Linear(hid_dim, hid_dim))

    def forward(self, x_global, x_local):   # shapes: x_global[b,latent], x_local[b,n,latent_local]
        _, num_points, _ = x_local.size()
        latent_global = x_global.size(1)

        x_pooled_sum = x_local.sum(1, keepdim=False)
        x_pooled_mean = x_local.mean(1, keepdim=False)
        x_pooledCATglobal = torch.cat([x_pooled_mean, x_pooled_sum, x_global], 1)
        x_global1 = F.leaky_relu(self.fc_global1(x_pooledCATglobal))  # new intermediate step
        x_global = F.leaky_relu(self.fc_global2(x_global1) + x_global) # with residual connection before AF

        x_global2local = x_global.view(-1,1,latent_global).repeat(1,num_points,1) # first add dimension, than expand it
        x_localCATglobal = torch.cat([x_local, x_global2local], 2)
        x_local1 = F.leaky_relu(self.fc_local1(x_localCATglobal))  # with residual connection before AF
        x_local = F.leaky_relu(self.fc_local2(x_local1) + x_local)

        return x_global, x_local


class EPiC_Network(nn.Module):
    def __init__(self, 
                 feats = 3,
                 latent_global = 10,    # used for latent size of equiv concat
                 latent_local = 3,
                 hid_d = 256, 
                 equiv_layers = 3,
                 time_varying = False):
        
        super(EPiC_Network, self).__init__()
        self.latent_global = latent_global   # used for latent size of equiv concat
        self.latent_local = latent_local + (1 if time_varying else 0)  # noise
        self.hid_d = hid_d   
        self.feats = feats
        self.equiv_layers = equiv_layers
        self.time_varying = time_varying
        
        self.local_0 = weight_norm(nn.Linear(self.latent_local, self.hid_d))  # local projection_mlp
        self.global_0 = weight_norm(nn.Linear(2 * self.latent_local, self.latent_global)) # local to global projection_mlp
        self.global_1 = weight_norm(nn.Linear(self.latent_global, self.hid_d))
        self.global_2 = weight_norm(nn.Linear(self.hid_d, self.latent_global))
        
        self.epic_layers = nn.ModuleList()
        for _ in range(self.equiv_layers):
            self.epic_layers.append(EPiC_layer(local_in_dim=self.hid_d, 
                                               hid_dim=self.hid_d,
                                               latent_dim=self.latent_global))
                                            
        self.local_1 = weight_norm(nn.Linear(self.hid_d, self.feats))

    def meansum_pooling(self, z_local, mask):
        z_mean = (mask * z_local).mean(1, keepdim=False)
        z_sum = z_local.sum(1, keepdim=False) / mask.sum(1, keepdim=False)  
        z_global = torch.cat([z_mean, z_sum], 1) 
        return z_global

    def forward(self, z_local, mask):   # shape: [batch, points, feats]

        #...local to global:
        z_global = self.meansum_pooling(z_local, mask)
        z_global = F.leaky_relu(self.global_0(z_global))      
        z_global = F.leaky_relu(self.global_1(z_global))
        z_global = F.leaky_relu(self.global_2(z_global))   
        z_global_in = z_global.clone()

        #...local:
        z_local = F.leaky_relu(self.local_0(z_local))   
        z_local_in = z_local.clone()

        #...equivariant layers:
        for i in range(self.equiv_layers):
            z_global, z_local = self.epic_layers[i](z_global, z_local)   
            z_global += z_global_in   
            z_local += z_local_in    
        
        output = self.local_1(z_local)
        
        return output     #[batch, points, feats]


#...EPiC with Self-Attention

class SelfAttentionPooling(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(SelfAttentionPooling, self).__init__()
        self.query = weight_norm(nn.Linear(in_dim, in_dim))
        self.key = weight_norm(nn.Linear(in_dim, in_dim))
        self.value = weight_norm(nn.Linear(in_dim, in_dim))
        self.fc_out = weight_norm(nn.Linear(in_dim, out_dim))

    def forward(self, x, mask=None):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        QK = torch.matmul(Q, K.transpose(2, 1))

        if mask is None:
            M = torch.zeros_like(QK)
        else:
            M = torch.ones_like(QK) * float('-inf')
            mask = mask.squeeze(-1)
            mask = mask.unsqueeze(1) * mask.unsqueeze(2)
            M = M.masked_fill(mask == 1, 0)  # Set valid positions to zero
        A = torch.nn.functional.softmax(QK + M, dim=-1)
        global_features = torch.matmul(A, V).sum(dim=1)
        global_features = self.fc_out(global_features)
        return global_features


class EPiC_Attention_Network(nn.Module):
    def __init__(self, 
                 feats = 3,
                 latent_global = 10,    # used for latent size of equiv concat
                 latent_local = 3,
                 hid_d = 256, 
                 equiv_layers = 3,
                 time_varying = False):
        
        super(EPiC_Attention_Network, self).__init__()
        self.latent_global = latent_global   # used for latent size of equiv concat
        self.latent_local = latent_local + (1 if time_varying else 0)  # noise
        self.hid_d = hid_d   
        self.feats = feats
        self.equiv_layers = equiv_layers
        self.time_varying = time_varying
        
        self.attention_pooling = SelfAttentionPooling(self.latent_local, self.latent_global)
        self.local_0 = weight_norm(nn.Linear(self.latent_local, self.hid_d))  # local projection_mlp
        self.global_0 = weight_norm(nn.Linear(2 * self.latent_local, self.latent_global)) # local to global projection_mlp
        self.global_1 = weight_norm(nn.Linear(self.latent_global, self.hid_d))
        self.global_2 = weight_norm(nn.Linear(self.hid_d, self.latent_global))
        
        self.epic_layers = nn.ModuleList()
        for _ in range(self.equiv_layers):
            self.epic_layers.append(EPiC_layer(local_in_dim=self.hid_d, 
                                               hid_dim=self.hid_d,
                                               latent_dim=self.latent_global))
                                            
        self.local_1 = weight_norm(nn.Linear(self.hid_d, self.feats))

    def forward(self, z_local, mask):   # shape: [batch, points, feats]

        #...local to global:
        z_global = self.attention_pooling(z_local, mask)    
        z_global = F.leaky_relu(self.global_1(z_global))
        z_global = F.leaky_relu(self.global_2(z_global))   
        z_global_in = z_global.clone()

        #...local:
        z_local = F.leaky_relu(self.local_0(z_local))   
        z_local_in = z_local.clone()

        #...equivariant layers:
        for i in range(self.equiv_layers):
            z_global, z_local = self.epic_layers[i](z_global, z_local)   
            z_global += z_global_in   
            z_local += z_local_in    
        
        output = self.local_1(z_local)
        
        return output     #[batch, points, feats]










