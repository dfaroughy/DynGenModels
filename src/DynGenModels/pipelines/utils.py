import torch

class TorchdynWrapper(torch.nn.Module):
    """Wraps model to torchdyn compatible format.
    """
    def __init__(self, net):
        super().__init__()
        self.nn = net
    def forward(self, t, x):
        t = t.repeat(x.shape[:-1]+(1,), 1)
        return self.nn(t=t, x=x)