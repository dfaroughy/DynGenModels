from torch import nn
from torch.nn import functional as F
from torchcfm.models.unet import UNetModel

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
    

# class UNetLight(nn.Module):

#     def __init__(self, config):

#         super().__init__()
#         self.dimensions = config.data0.dimensions
#         self.vocab_size = config.data0.vocab_size
#         self.time_embed_dim = config.temporal_network.time_embed_dim
#         self.hidden_dim = config.temporal_network.hidden_dim
#         self.Encoder()
#         self.TimeEmbedding()
#         self.Decoder()
#         self.to(device)
#         self.expected_output_shape = [28, 28, self.vocab_size]

#     def Encoder(self):
#         self.init_conv = ResidualConvBlock(1, self.hidden_dim , is_res=True)
#         self.down1 = UnetDown(self.hidden_dim, self.hidden_dim)
#         self.down2 = UnetDown(self.hidden_dim, 2 * self.hidden_dim)
#         self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())

#     def TimeEmbedding(self):
#         self.timeembed1 = EmbedFC(1, 2*self.hidden_dim)
#         self.timeembed2 = EmbedFC(1, 1*self.hidden_dim)

#     def Decoder(self):
#         self.up0 = nn.Sequential(
#             nn.ConvTranspose2d(2 * self.hidden_dim, 2 * self.hidden_dim, 7, 7), 
#             nn.GroupNorm(8, 2 * self.hidden_dim),
#             nn.GELU())

#         self.up1 = UnetUp(4 * self.hidden_dim, self.hidden_dim)
#         self.up2 = UnetUp(2 * self.hidden_dim, self.hidden_dim)
#         self.out = nn.Sequential(
#             nn.Conv2d(2 * self.hidden_dim, self.hidden_dim, 3, 1, 1),
#             nn.GroupNorm(8, self.hidden_dim),
#             nn.GELU(),
#             nn.Conv2d(self.hidden_dim, self.vocab_size, 3, 1, 1),
#         )

#     def forward(self, x, times):
#         x = self.init_conv(x)
        
#         # encode:
#         down1 = self.down1(x)
#         down2 = self.down2(down1)
#         hiddenvec = self.to_vec(down2)

#         # embed:
#         temb1 = self.timeembed1(times).view(-1, self.hidden_dim * 2, 1, 1)
#         temb2 = self.timeembed2(times).view(-1, self.hidden_dim, 1, 1)
        
#         # decode:
#         up1 = self.up0(hiddenvec)
#         up2 = self.up1(up1 + temb1, down2) 
#         up3 = self.up2(up2 + temb2, down1)
#         out = self.out(torch.cat((up3, x), 1))

#         return out.permute(0, 2, 3, 1) 


# class ResidualConvBlock(nn.Module):
#     def __init__(
#         self, in_channels: int, out_channels: int, is_res: bool = False
#     ) -> None:
#         super().__init__()
#         '''
#         standard ResNet style convolutional block
#         '''
#         self.same_channels = in_channels==out_channels
#         self.is_res = is_res
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, 3, 1, 1),
#             nn.BatchNorm2d(out_channels),
#             nn.GELU(),
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(out_channels, out_channels, 3, 1, 1),
#             nn.BatchNorm2d(out_channels),
#             nn.GELU(),
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         if self.is_res:
#             x1 = self.conv1(x)
#             x2 = self.conv2(x1)
#             if self.same_channels:
#                 out = x + x2
#             else:
#                 out = x1 + x2 
#             return out / 1.414
#         else:
#             x1 = self.conv1(x)
#             x2 = self.conv2(x1)
#             return x2


# class UnetDown(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(UnetDown, self).__init__()
#         '''
#         process and downscale the image feature maps
#         '''
#         layers = [ResidualConvBlock(in_channels, out_channels), nn.MaxPool2d(2)]
#         self.model = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.model(x)


# class UnetUp(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(UnetUp, self).__init__()
#         '''
#         process and upscale the image feature maps
#         '''
#         layers = [
#             nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
#             ResidualConvBlock(out_channels, out_channels),
#             ResidualConvBlock(out_channels, out_channels),
#         ]
#         self.model = nn.Sequential(*layers)

#     def forward(self, x, skip):
#         x = torch.cat((x, skip), 1)
#         x = self.model(x)
#         return x


# class EmbedFC(nn.Module):
#     def __init__(self, input_dim, emb_dim):
#         super(EmbedFC, self).__init__()
#         '''
#         generic one layer FC NN for embedding things  
#         '''
#         self.input_dim = input_dim
#         layers = [
#             nn.Linear(input_dim, emb_dim),
#             nn.GELU(),
#             nn.Linear(emb_dim, emb_dim),
#         ]
#         self.model = nn.Sequential(*layers)

#     def forward(self, x):
#         x = x.view(-1, self.input_dim)
#         return self.model(x)
