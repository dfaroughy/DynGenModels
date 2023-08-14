import torch

class torchdyn_wrapper(torch.nn.Module):
    """ Wraps model to torchdyn compatible format.
    """
    def __init__(self, vector_field, context, mask):
        super().__init__()

        self.vector_field = vector_field
        self.context = context
        self.mask = mask

    def forward(self, t, x):
        t = t.repeat(x.shape[:-1]+(1,), 1)
        return self.vector_field(t=t, x=x, context=self.context, mask=self.mask)