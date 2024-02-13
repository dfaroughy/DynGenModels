import torch
from torch import nn
from torch.nn import functional as F
from torchcfm.models.unet import UNetModel
from DynGenModels.models.architectures.utils import get_activation_function, transformer_timestep_embedding


class Unet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config.DEVICE
        self.unet = UNetModel(dim=config.INPUT_SHAPE, 
                              num_channels=config.DIM_HIDDEN, 
                              num_res_blocks=config.NUM_RES_BLOCKS)
        self.to(self.device)

    def forward(self, t, x, context=None, mask=None):
        x = self.unet(t, x, y=context)
        return x
    

class UNetLight(nn.Module):

    def __init__(self, config):

        super().__init__()
        self.input_shape = config.INPUT_SHAPE
        self.dim_time_emb = config.DIM_TIME_EMB
        self.dim_hidden = config.DIM_HIDDEN
        # self.act_fn = get_activation_function(config.ACTIVATION)
        self.Encoder()
        self.TimeEmbedding()
        self.Decoder()
        self.to(config.DEVICE)

    def Encoder(self):
        self.init_conv = ResidualConvBlock(1, self.dim_hidden , is_res=True)
        self.down1 = Down(self.dim_hidden, self.dim_hidden)
        self.down2 = Down(self.dim_hidden, 2 * self.dim_hidden)
        self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())

    def TimeEmbedding(self):
        self.timeembed1 = EmbedFC(1, 2 * self.dim_hidden)
        self.timeembed2 = EmbedFC(1, self.dim_hidden)

    def Decoder(self):
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * self.dim_hidden, 2 * self.dim_hidden, 7, 7), 
            nn.GroupNorm(8, 2 * self.dim_hidden),
            nn.GELU())

        self.up1 = Up(4 * self.dim_hidden, self.dim_hidden)
        self.up2 = Up(2 * self.dim_hidden, self.dim_hidden)
        self.out = nn.Sequential(nn.Conv2d(2 * self.dim_hidden, self.dim_hidden, kernel_size=3, stride=1, padding=1),
                                 nn.GroupNorm(8, self.dim_hidden),
                                 nn.GELU(),
                                 nn.Conv2d(self.dim_hidden, 1, kernel_size=3, stride=1, padding=1)
                                 )
    def forward(self, t, x, context=None, mask=None):
        x = self.init_conv(x)
        
        # encode:
        down1 = self.down1(x)
        down2 = self.down2(down1)
        hiddenvec = self.to_vec(down2)

        # embed:
        temb1 = self.timeembed1(t).view(-1, self.dim_hidden * 2, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.dim_hidden, 1, 1)
        
        # decode:
        up1 = self.up0(hiddenvec)
        up2 = self.up1(up1 + temb1, down2) 
        up3 = self.up2(up2 + temb2, down1)
        out = self.out(torch.cat((up3, x), 1))

        return out.permute(0, 2, 3, 1) 


class ResidualConvBlock(nn.Module):
    def __init__( self, in_channels: int, out_channels: int, is_res: bool = False):
        super().__init__()
        '''
        standard ResNet style convolutional block
        '''
        self.same_channels = in_channels==out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(out_channels),
                                   nn.GELU(),
                                   )
        self.conv2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(out_channels),
                                   nn.GELU(),
                                   )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2 
            return out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.down = nn.Sequential(ResidualConvBlock(in_channels, out_channels), nn.MaxPool2d(kernel_size=2))
    def forward(self, x):
        return self.down(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                                ResidualConvBlock(out_channels, out_channels),
                                ResidualConvBlock(out_channels, out_channels),)
    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.up(x)
        return x

class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        self.input_dim = input_dim
        self.fc_model = nn.Sequential(nn.Linear(input_dim, emb_dim), nn.GELU(), nn.Linear(emb_dim, emb_dim),)
    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.fc_model(x)
