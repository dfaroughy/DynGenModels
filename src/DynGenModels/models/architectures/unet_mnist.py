import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torchcfm.models.unet import UNetModel
from DynGenModels.models.architectures.utils import get_activation_function, transformer_timestep_embedding


class Unet28x28(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.input_shape = config.INPUT_SHAPE
        self.dim_time_emb = config.DIM_TIME_EMB
        self.dim_hidden = config.DIM_HIDDEN
        self.dropout = config.DROPOUT
        self.device = config.DEVICE
        self.act = get_activation_function(config.ACTIVATION)
        self.Embeddings()
        self.Encoder()
        self.Decoder()
        self.to(config.DEVICE)

    def Embeddings(self):
        self.projection = nn.Conv2d(1, self.dim_hidden, kernel_size=3, stride=1, padding=1)
        self.time_embedding = Time_embedding(self.dim_time_emb, self.dim_hidden, nn.GELU())

    def Encoder(self):
        self.down1 = Down(in_channels=self.dim_hidden, out_channels=self.dim_hidden, time_channels=self.dim_hidden, activation=self.act, dropout=self.dropout)
        self.down2 = Down(in_channels=self.dim_hidden, out_channels=2*self.dim_hidden, time_channels=self.dim_hidden,  activation=self.act, dropout=self.dropout)
        self.pool = nn.Sequential(nn.AvgPool2d(7), self.act)

    def Decoder(self):
        self.up0 = nn.Sequential(nn.ConvTranspose2d(2 * self.dim_hidden, 2 * self.dim_hidden, 7, 7), 
                                 nn.GroupNorm(8, 2 * self.dim_hidden),
                                 self.act)

        self.up1 = Up(in_channels=4 * self.dim_hidden, out_channels=self.dim_hidden, time_channels=self.dim_hidden,  activation=self.act, dropout=self.dropout)
        self.up2 = Up(in_channels=2 * self.dim_hidden, out_channels=self.dim_hidden, time_channels=self.dim_hidden,  activation=self.act, dropout=self.dropout)
        self.output = nn.Sequential(nn.Conv2d(2 * self.dim_hidden, self.dim_hidden, kernel_size=3, stride=1, padding=1),
                                    nn.GroupNorm(8, self.dim_hidden),
                                    self.act,
                                    nn.Conv2d(self.dim_hidden, 1, kernel_size=3, stride=1, padding=1)
                                    )

    def forward(self, t, x, context=None, mask=None):

        x = x.to(self.device)
        t = t.to(self.device)
        
        #...embed inputs:
        temb = self.time_embedding(t)
        x = self.projection(x)

        #...encode:
        down1 = self.down1(x, temb)
        down2 = self.down2(down1, temb)
        h = self.pool(down2)
        
        #...decode:
        up1 = self.up0(h)
        up2 = self.up1(up1, temb, down2) 
        up3 = self.up2(up2, temb, down1)
        output = self.output(torch.cat((up3, x), 1))

        return output


class TemporalResidualConvBlock(nn.Module):
    def __init__( self, 
                 in_channels: int, 
                 out_channels: int, 
                 time_channels: int,
                 activation: nn.Module=nn.GELU(),
                 dropout: float=0.1):
        super().__init__()
        '''
        standard ResNet style convolutional block
        '''
        self.same_channels = in_channels==out_channels

        self.conv_1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(out_channels),
                                    activation,
                                    )
        
        self.conv_2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(out_channels),
                                    activation,
                                    )

        self.time_emb_1 = nn.Sequential(nn.Linear(time_channels, out_channels),
                                       nn.BatchNorm1d (out_channels), 
                                       activation, 
                                       nn.Dropout(dropout)) 

        self.time_emb_2 = nn.Sequential(nn.Linear(time_channels, out_channels),
                                       nn.BatchNorm1d (out_channels), 
                                       activation, 
                                       nn.Dropout(dropout)) 

        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1)) if not self.same_channels else nn.Identity()

    def forward(self, x, t):
        h = self.conv_1(x)
        h += self.time_emb_1(t).view(-1, h.shape[1], 1, 1)
        h = self.conv_2(h)
        h += self.time_emb_2(t).view(-1, h.shape[1], 1, 1)
        h += self.skip(x)
        return h / (np.sqrt(2.0) if self.same_channels else 1.0) 
    
class Down(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels, activation, dropout):
        super(Down, self).__init__()
        self.conv_block = TemporalResidualConvBlock(in_channels=in_channels, out_channels=out_channels, time_channels=time_channels, activation=activation, dropout=dropout)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x, t):
        x = self.conv_block(x, t)
        x = self.pool(x) 
        return x

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels, activation, dropout):
        super(Up, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, 2, 2)
        self.conv_block1 = TemporalResidualConvBlock(in_channels=out_channels, out_channels=out_channels, time_channels=time_channels, activation=activation, dropout=dropout)
        self.conv_block2 = TemporalResidualConvBlock(in_channels=out_channels, out_channels=out_channels, time_channels=time_channels, activation=activation, dropout=dropout)

    def forward(self, x, t, skip):
        x = torch.cat((x, skip), dim=1)
        x = self.upsample(x)
        x = self.conv_block1(x, t)
        x = self.conv_block2(x, t)
        return x


class Time_embedding(nn.Module):
    def __init__(self, dim_time_emb, dim_hidden, activation_fn=nn.GELU()):
        super(Time_embedding, self).__init__()

        self.dim_time_emb = dim_time_emb

        layers = [ nn.Linear(dim_time_emb, dim_hidden),
                   activation_fn,
                   nn.Linear(dim_hidden, dim_hidden),
                  ]
        self.fc = nn.Sequential(*layers)

    def forward(self, t):
        temb = transformer_timestep_embedding(t.squeeze(), self.dim_time_emb, max_positions=10000)
        return self.fc(temb)



# #... main


if __name__ == '__main__':

    from dataclasses import dataclass

    @dataclass
    class config:
        INPUT_SHAPE=(1, 28, 28)
        DIM_TIME_EMB=32
        DIM_HIDDEN=64
        DROPOUT=0.1
        ACTIVATION="ELU"
        DEVICE='cpu'
        
    unet = Unet28x28(config)
    x = torch.randn(10, 1, 28, 28)
    t = torch.randn(10)
    y = unet(t, x)
    print(y.shape)

