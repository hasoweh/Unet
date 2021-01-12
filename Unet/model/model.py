from .blocks import *
import torch.nn as nn

class Unet(nn.Module):
    def __init__(self, 
                 in_channels, 
                 n_class, 
                 bias = True
                 ):
        super(Unet, self).__init__()

        self.dblock1 = ConvBlock(in_channels, 
                               64, 
                               3, 
                               block_num = 1,
                               bias = bias)
        self.dblock2 = ConvBlock(64, 
                               128, 
                               3, 
                               block_num = 2,
                               bias = bias)
        self.dblock3 = ConvBlock(128, 
                               256, 
                               3, 
                               block_num = 3,
                               bias = bias)
        self.dblock4 = ConvBlock(256, 
                               512, 
                               3, 
                               block_num = 4,
                               bias = bias)
        self.center = ConvBlock(512, 
                               1024, 
                               3, 
                               block_num = 5,
                               bias = bias,
                               down = False)
        self.ublock1 = DeconvBlock(1024, 
                                   512, 
                                   kernel = 3, 
                                   block_num = 6,
                                   bias = bias)
        self.ublock2 = DeconvBlock(512, 
                                   256, 
                                   kernel = 3, 
                                   block_num = 7,
                                   bias = bias)
        self.ublock3 = DeconvBlock(256, 
                                   128, 
                                   kernel = 3, 
                                   block_num = 8,
                                   bias = bias)
        self.ublock4 = DeconvBlock(128, 
                                   64, 
                                   kernel = 3, 
                                   block_num = 9,
                                   bias = bias)
        self.pred = Logits(64, 
                           n_class)
        
    def forward(self, x):
        residual1, x = self.dblock1(x)
        residual2, x = self.dblock2(x)
        residual3, x = self.dblock3(x)
        residual4, x = self.dblock4(x)

        x = self.center(x)

        x = self.ublock1(x, residual4)
        x = self.ublock2(x, residual3)
        x = self.ublock3(x, residual2)
        x = self.ublock4(x, residual1)

        x = self.pred(x)
        
        return x
