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
        # t = t.repeat(x.shape[:-1]+(1,), 1)
        t = t.repeat(x.shape[0])[:, None]
        return self.nn(t=t, x=x, mask=self.mask)
