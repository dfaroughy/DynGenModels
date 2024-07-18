import torch
from torch import nn
from torch.nn import functional as F
from torchcfm.models.unet import UNetModel
from DynGenModels.models.architectures.utils import get_activation_function, transformer_timestep_embedding


class UnetCFM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config.DEVICE
        self.unet = UNetModel(dim=config.INPUT_SHAPE, 
                              num_channels=config.DIM_HIDDEN, 
                              num_res_blocks=config.NUM_RES_BLOCKS)
        self.to(self.device)

    def forward(self, t, x, context=None, mask=None):

        x = x.to(self.device)
        t = t.to(self.device)
        if context is not None: context = context.to(self.device)
        x = self.unet(t, x, y=context)
        return x
    