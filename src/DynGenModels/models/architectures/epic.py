import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.utils.weight_norm as weight_norm
from DynGenModels.models.architectures.utils import timestep_sinusoidal_embedding, GaussianFourierProjection

class EPiC(nn.Module):
    ''' Wrapper class for the EPiC architecture
    '''
    def __init__(self, config):
        super(EPiC, self).__init__()
        self.device = config.DEVICE

        if config.POOLING=='mean_sum':
            self.epic = EPiC_Network(dim_features = config.DIM_INPUT,
                                     dim_latent_global = config.DIM_GLOBAL, 
                                     time_embedding = config.TIME_EMBEDDING,
                                     dim_time_embedding = config.DIM_TIME_EMB,
                                     dim_hidden = config.DIM_HIDDEN, 
                                     num_layers = config.NUM_EPIC_LAYERS,
                                     skip_connection = config.USE_SKIP_CONNECTION)
            
    def forward(self, t, x, mask=None):
        t = t.to(self.device)   
        x = x.to(self.device)
        self.epic = self.epic.to(self.device)
        return self.epic.forward(t, x)


class EPiC_Network(nn.Module):
    def __init__(self, 
                 dim_features = 3,
                 dim_latent_global = 10,
                 time_embedding = 'sinusoidal',  
                 dim_time_embedding = 10,
                 dim_hidden = 256, 
                 num_layers = 3,
                 skip_connection = False):
        
        super(EPiC_Network, self).__init__()
        self.time_embedding = time_embedding
        self.dim_time_embedding = dim_time_embedding
        self.num_layers = num_layers
        self.skip_connection = skip_connection

        #...projection network

        self.epic_proj = EPiC_Projection(dim_latent_global=dim_latent_global,
                                         latent_local=dim_features + dim_time_embedding,
                                         dim_hidden=dim_hidden)

        #...epic layers:

        self.epic_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.epic_layers.append(EPiC_layer(local_in_dim=dim_hidden, 
                                               hid_dim=dim_hidden,
                                               latent_dim=dim_latent_global,
                                               context_dim=dim_time_embedding))
                                            
        #...output layer:

        self.output = weight_norm(nn.Linear(dim_hidden, dim_features))

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

class EPiC_Projection(nn.Module):
    def __init__(self, dim_latent_global, latent_local, dim_hidden):
        super(EPiC_Projection, self).__init__()

        self.local_0 = weight_norm(nn.Linear(latent_local, dim_hidden))  # local projection_mlp
        self.global_0 = weight_norm(nn.Linear(2*dim_hidden, dim_hidden)) # local to global projection_mlp
        self.global_1 = weight_norm(nn.Linear(dim_hidden, dim_hidden))
        self.global_2 = weight_norm(nn.Linear(dim_hidden, dim_latent_global))

    def meansum_pooling(self, x_local):
        x_mean = x_local.mean(1, keepdim=False)
        x_sum = x_local.sum(1, keepdim=False) 
        x_global = torch.cat([x_mean, x_sum], 1) 
        return x_global

    def forward(self, x_local):
        x_local = F.leaky_relu(self.local_0(x_local)) 
        x_global = self.meansum_pooling(x_local)
        x_global = F.leaky_relu(self.global_0(x_global))      
        x_global = F.leaky_relu(self.global_1(x_global))
        x_global = F.leaky_relu(self.global_2(x_global))   
        return x_global, x_local

class EPiC_layer(nn.Module):
    # from https://github.com/uhh-pd-ml/EPiC-GAN/blob/main/models.py
    def __init__(self, local_in_dim, hid_dim, latent_dim, context_dim):
        super(EPiC_layer, self).__init__()
        self.fc_global1 = weight_norm(nn.Linear(int(2*hid_dim) + latent_dim, hid_dim)) 
        self.fc_global2 = weight_norm(nn.Linear(hid_dim, latent_dim)) 
        self.fc_local1 = weight_norm(nn.Linear(local_in_dim + latent_dim + context_dim, hid_dim))
        self.fc_local2 = weight_norm(nn.Linear(hid_dim, hid_dim))

    def forward(self, x_global, x_local, context):   # shapes: x_global[b,latent], x_local[b,n,latent_local]
        _, num_points, _ = x_local.size()
        dim_latent_global = x_global.size(1)

        x_pooled_sum = x_local.sum(1, keepdim=False)
        x_pooled_mean = x_local.mean(1, keepdim=False)
        x_pooledCATglobal = torch.cat([x_pooled_mean, x_pooled_sum, x_global], 1)
        x_global1 = F.leaky_relu(self.fc_global1(x_pooledCATglobal))  # new intermediate step
        x_global = F.leaky_relu(self.fc_global2(x_global1) + x_global) # with residual connection before AF

        x_global2local = x_global.view(-1,1,dim_latent_global).repeat(1,num_points,1) # first add dimension, than expand it
        x_localCATglobal = torch.cat([x_local, x_global2local, context], 2)
        x_local1 = F.leaky_relu(self.fc_local1(x_localCATglobal))  # with residual connection before AF
        x_local = F.leaky_relu(self.fc_local2(x_local1) + x_local)

        return x_global, x_local
        

# class EPiC_Attention_Network(nn.Module):
#     def __init__(self, 
#                  feats = 3,
#                  dim_latent_global = 10,    # used for latent size of equiv concat
#                  latent_local = 3,
#                  dim_hidden = 256, 
#                  num_layers = 3,
#                  time_varying = False):
        
#         super(EPiC_Attention_Network, self).__init__()
#         self.dim_latent_global = dim_latent_global   # used for latent size of equiv concat
#         self.latent_local = latent_local + (1 if time_varying else 0)  # noise
#         self.dim_hidden = dim_hidden   
#         self.feats = feats
#         self.num_layers = num_layers
#         self.time_varying = time_varying
        
#         self.attention_pooling = SelfAttentionPooling(self.latent_local, self.dim_latent_global)
#         self.local_0 = weight_norm(nn.Linear(self.latent_local, self.dim_hidden))  # local projection_mlp
#         self.global_0 = weight_norm(nn.Linear(2 * self.latent_local, self.dim_latent_global)) # local to global projection_mlp
#         self.global_1 = weight_norm(nn.Linear(self.dim_latent_global, self.dim_hidden))
#         self.global_2 = weight_norm(nn.Linear(self.dim_hidden, self.dim_latent_global))
        
#         self.epic_layers = nn.ModuleList()
#         for _ in range(self.num_layers):
#             self.epic_layers.append(EPiC_layer(local_in_dim=self.dim_hidden, 
#                                                hid_dim=self.dim_hidden,
#                                                latent_dim=self.dim_latent_global))
                                            
#         self.local_1 = weight_norm(nn.Linear(self.dim_hidden, self.feats))

#     def forward(self, z_local, mask):   # shape: [batch, points, feats]

#         #...local to global:
#         z_global = self.attention_pooling(z_local, mask)    
#         z_global = F.leaky_relu(self.global_1(z_global))
#         z_global = F.leaky_relu(self.global_2(z_global))   
#         z_global_in = z_global.clone()

#         #...local:
#         z_local = F.leaky_relu(self.local_0(z_local))   
#         z_local_in = z_local.clone()

#         #...equivariant layers:
#         for i in range(self.num_layers):
#             z_global, z_local = self.epic_layers[i](z_global, z_local)   
#             z_global += z_global_in   
#             z_local += z_local_in    
        
#         output = self.local_1(z_local)
        
#         return output     #[batch, points, feats]


# class SelfAttentionPooling(nn.Module):
#     def __init__(self, in_dim, out_dim):
#         super(SelfAttentionPooling, self).__init__()
#         self.query = weight_norm(nn.Linear(in_dim, in_dim))
#         self.key = weight_norm(nn.Linear(in_dim, in_dim))
#         self.value = weight_norm(nn.Linear(in_dim, in_dim))
#         self.fc_out = weight_norm(nn.Linear(in_dim, out_dim))

#     def forward(self, x, mask=None):
#         Q = self.query(x)
#         K = self.key(x)
#         V = self.value(x)
#         QK = torch.matmul(Q, K.transpose(2, 1))

#         if mask is None:
#             M = torch.zeros_like(QK)
#         else:
#             M = torch.ones_like(QK) * float('-inf')
#             mask = mask.squeeze(-1)
#             mask = mask.unsqueeze(1) * mask.unsqueeze(2)
#             M = M.masked_fill(mask == 1, 0)  # Set valid positions to zero
#         A = torch.nn.functional.softmax(QK + M, dim=-1)
#         global_features = torch.matmul(A, V).sum(dim=1)
#         global_features = self.fc_out(global_features)
#         return global_features







