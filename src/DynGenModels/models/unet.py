from torch import nn
from torch.nn import functional as F
from torchcfm.models.unet import UNetModel

class Unet(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.device = configs.DEVICE
        self.unet = UNetModel(dim=configs.input_shape, 
                              num_channels=configs.dim_hidden, 
                              num_res_blocks=configs.num_res_blocks)
        self.to(self.device)

    def forward(self, t, x, context=None, mask=None):
        x = self.unet(t, x, y=context)
        return x