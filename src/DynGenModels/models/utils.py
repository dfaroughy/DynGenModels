import math
import torch
import torch.nn.functional as F
import torch.nn as nn

def get_activation_function(name: str='ReLU'):
    if name is not None:
        activation_functions = {"ReLU": nn.ReLU(),
                                "LeakyReLU": nn.LeakyReLU(),
                                "ELU": nn.ELU(),
                                "SELU": nn.SELU(),
                                "GLU": nn.GLU(),
                                "GELU": nn.GELU(),
                                "CELU": nn.CELU(),
                                "PReLU": nn.PReLU(),
                                "Sigmoid": nn.Sigmoid(),
                                "Tanh": nn.Tanh(),
                                "Hardswish": nn.Hardswish(),
                                "Hardtanh": nn.Hardtanh(),
                                "LogSigmoid": nn.LogSigmoid(),
                                "Softplus": nn.Softplus(),
                                "Softsign": nn.Softsign(),
                                "Softshrink": nn.Softshrink(),
                                "Softmin": nn.Softmin(),
                                "Softmax": nn.Softmax()}
        return activation_functions[name]
    else: return None

def transformer_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
  assert len(timesteps.shape) == 1 
  half_dim = embedding_dim // 2
  # magic number 10000 is from transformers
  emb = math.log(max_positions) / (half_dim - 1)
  emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
  emb = timesteps.float()[:, None] * emb[None, :]
  emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
  if embedding_dim % 2 == 1:  # zero pad
    emb = F.pad(emb, (0, 1), mode='constant')
  assert emb.shape == (timesteps.shape[0], embedding_dim)
  return emb

def loretz_scalar_embedding(x):
    j1 = x[...,:4]
    j2 = x[...,4:]
    p1p1 = (j1[...,0]**2 - j1[...,1]**2 - j1[...,2]**2 - j1[...,3]**2)[:, None]
    p2p2 = (j2[...,0]**2 - j2[...,1]**2 - j2[...,2]**2 - j2[...,3]**2)[:, None]
    p1p2 = (j1[...,0]*j2[...,0] - j1[...,1]*j2[...,1] - j1[...,2]*j2[...,2] - j1[...,3]*j2[...,3])[:, None]
    return torch.cat([x, p1p1, p2p2, p1p2], dim=-1)