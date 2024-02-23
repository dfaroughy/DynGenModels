import torch
from dataclasses import dataclass

class TorchdynWrapper(torch.nn.Module):
    """ Wraps model to torchdyn compatible format.
    """
    def __init__(self, net, mask=None):
        super().__init__()
        self.nn = net
        self.mask = mask
    def forward(self, t, x):
        t = t.repeat(x.shape[0])
        t = reshape_time_like(t, x)
        return self.nn(t=t, x=x, mask=self.mask)

def reshape_time_like(t, x):
	if isinstance(t, (float, int)): return t
	else: return t.reshape(-1, *([1] * (x.dim() - 1)))