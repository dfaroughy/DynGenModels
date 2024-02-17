import math
import torch
import torch.nn.functional as F
import torch.nn as nn

def fc_block(dim_input, dim_output, dim_hidden, num_layers, activation, dropout, use_batch_norm=False):

  BatchNorm = nn.BatchNorm1d if use_batch_norm else nn.Identity

  layers = [torch.nn.Linear(dim_input, dim_hidden), BatchNorm(dim_hidden), activation]
  if dropout: layers.append(nn.Dropout(dropout)) 

  for _ in range(num_layers-2): 
      layers.extend([torch.nn.Linear(dim_hidden, dim_hidden), BatchNorm(dim_hidden), activation])
      if dropout: layers.extend([nn.Dropout(dropout)]) 

  layers.append(torch.nn.Linear(dim_hidden, dim_output))
  return torch.nn.Sequential(*layers)


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


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps.

    Inspired by https://colab.research.google.com/drive/120kYYBOVa1i0TD85RjlEkFjaWDxSFUx3?usp=sharing#scrollTo=YyQtV7155Nht
    """

    def __init__(self, embed_dim, scale=30.0, device='cpu'):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim // 2, device=device) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[..., None] * self.W[None, ...] * 2 * math.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


def timestep_sinusoidal_embedding(timesteps, dim, max_period=10000):
    """Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element. These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp( -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device) / half)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
