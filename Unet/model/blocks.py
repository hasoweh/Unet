import torch
import torch.nn as nn
import torch.nn.functional as F

def conv(in_channels, out_channels, kernel, padding, bias):
    return nn.Conv2d(in_channels, 
                     out_channels, 
                     kernel_size = kernel, 
                     padding = padding, 
                     bias = bias)

class ConvLayer(nn.Sequential):
    def __init__(self, 
                 n_block, 
                 n_layer, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 padding, 
                 bias):
        super(ConvLayer, self).__init__()
        
        self.add_module('conv%d_%d' % (n_block, n_layer), 
                        nn.Conv2d(in_channels, 
                                  out_channels, 
                                  kernel_size, 
                                  padding = padding, 
                                  bias = bias)
                       )
        self.add_module('bnorm%d_%d' % (n_block, n_layer), 
                        nn.BatchNorm2d(out_channels)
                        )
        self.add_module('relu%d_%d' % (n_block, n_layer), 
                        nn.ReLU()
                        )

class ConvBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel, 
                 block_num,
                 bias,
                 down = True
                 ):
        
        super(ConvBlock, self).__init__()
        
        self.conv1 = ConvLayer(block_num,
                               1,
                               in_channels, 
                               out_channels, 
                               kernel_size = 3, 
                               padding = 1, 
                               bias = bias)
        self.conv2 = ConvLayer(block_num,
                               2,
                               out_channels, 
                               out_channels, 
                               kernel_size = 3, 
                               padding = 1, 
                               bias = bias)
        if down:                       
            self.downsample = nn.MaxPool2d(2, 2)
        else:
            self.downsample = None
        
    def forward(self, x):
        if self.downsample is not None:
            x = self.conv1(x)
            residual = self.conv2(x)
            out = self.downsample(residual)
            return residual, out
        else:
            x = self.conv1(x)
            x = self.conv2(x)
            return x
            
            
class DeconvBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 kernel, 
                 block_num,
                 bias
                 ):
        
        super(DeconvBlock, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, 
                                           kernel_size=2, stride=2)
        self.conv1 = ConvLayer(block_num,
                               1,
                               in_channels, 
                               out_channels, 
                               kernel, 
                               padding = 1,
                               bias = bias)
        self.conv2 = ConvLayer(block_num,
                               2,
                               out_channels, 
                               out_channels, 
                               kernel, 
                               padding = 1,
                               bias = bias)
        
        
    def forward(self, x, residual):
        x = self.upsample(x)
        x = torch.cat((x, residual), dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class Logits(nn.Sequential):
    def __init__(self, 
                 in_channels, 
                 n_class
                 ):
      super(Logits, self).__init__()

      self.conv = self.add_module('conv_out', 
                        nn.Conv2d(in_channels, 
                                  n_class, 
                                  kernel_size = 1
                                  )
                       )
      self.activ = self.add_module('sigmoid_out', 
                        nn.Sigmoid()
                       )
                       
                       
                       
