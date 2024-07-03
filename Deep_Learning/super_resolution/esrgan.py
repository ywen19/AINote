"""Inspired from: https://medium.com/analytics-vidhya/esrgan-enhanced-super-resolution-gan-96a28821634"""

import torch
from torch import nn
from torchvision.models.vgg import vgg19
import math


class ResidualDenseBlock(nn.Module):
    def __init__(self, channels, inc_channels, kernel_size, padding, beta=0.2):
        super(ResidualDenseBlock, self).__init__()

        self.beta = beta

        self.conv1 = nn.Conv2d(channels, inc_channels, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(channels+inc_channels, inc_channels, kernel_size=kernel_size, padding=padding)
        self.conv3 = nn.Conv2d(channels+2*inc_channels, inc_channels, kernel_size=kernel_size, padding=padding)
        self.conv4 = nn.Conv2d(channels+3*inc_channels, inc_channels, kernel_size=kernel_size, padding=padding)
        self.conv5 = nn.Conv2d(channels+4*inc_channels, channels, kernel_size=kernel_size, padding=padding)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x1,x), dim=1)))
        x3 = self.lrelu(self.conv3(torch.cat((x2,x1,x), dim=1)))
        x4 = self.lrelu(self.conv4(torch.cat((x3,x2,x1,x), dim=1)))
        out = self.lrelu(self.conv5(torch.cat((x4,x3,x2,x1,x), dim=1)))
        # return is elementwise sum of the input and block calculation result(this is a skip connection)
        return x+self.beta*out

class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self, channels, out_channels, kernel_size, padding, beta=0.2):
        super(ResidualInResidualDenseBlock, self).__init__()
        
        self.RDB = ResidualDenseBlock(channels, out_channels, kernel_size, kernel_size//2, beta)
        self.beta = beta

    def forward(self, x):
        x1 = self.RDB(x)
        x1 = self.RDB(x1)
        x1 = self.RDB(x1)

        return x+self.beta*x1


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, upscale, kernel_size, padding):
        super(UpsampleBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, in_channels*upscale**2, kernel_size=kernel_size, padding=padding)
        # rearrange elements in a tensor with upscale factor: (∗,C×r**2,H,W) -> (∗,C,H×r,W×r)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=upscale)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pixel_shuffle(x)
        x = self.lrelu(x)
        return x

        
class ConvolutionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvolutionBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.leakyrelu(x)
        return x
                

class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        # initialize model configuration from passed-in config
        self.scaling_factor = config.scaling_factor
        self.small_kernel_size = config.G.small_kernel_size
        self.large_kernel_size = config.G.large_kernel_size
        self.n_channels = config.G.n_channels
        self.n_blocks = config.G.n_blocks
        self.out_channels = 32

        # initialize conv layer and PReLU layer before getting into residual blocks
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.n_channels, kernel_size=self.large_kernel_size, padding=self.large_kernel_size//2)
        self.lrelu = nn.LeakyReLU(0.2)

        # initialize residual blocks based on VGG architecture
        self.B_RRDB_Block = nn.Sequential()
        for i in range(self.n_blocks):
            self.B_RRDB_Block.add_module("BRDB_{}".format(i+1), ResidualInResidualDenseBlock(
                self.n_channels, 
                self.out_channels,
                self.small_kernel_size, 
                self.small_kernel_size//2))

        # initialize the final residual block
        self.conv2  = nn.Conv2d(in_channels=self.n_channels, 
                                out_channels=self.n_channels, 
                                kernel_size=self.small_kernel_size, 
                                padding=self.small_kernel_size//2)
        #self.bn1 = nn.BatchNorm2d(self.n_channels)

        # initialize upsmaple blocks
        self.Upsample_Block = nn.Sequential()
        for i in range(int(math.log2(self.scaling_factor))):
            self.Upsample_Block.add_module("Upsample_{}".format(i+1), UpsampleBlock(self.n_channels, 
                                                                                    int(self.scaling_factor**0.5), 
                                                                                    self.small_kernel_size, 
                                                                                    self.small_kernel_size//2))
        # initialize the final convolutional blcok and activation
        self.conv3 = nn.Conv2d(in_channels=self.n_channels, out_channels=3, kernel_size=self.large_kernel_size, padding=self.large_kernel_size//2)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.lrelu(self.conv1(x))
        x1 = self.B_RRDB_Block(x)
        x1 = self.conv2(x1)
        x1 = x+x1

        x1 = self.Upsample_Block(x1)
        x1 = self.conv3(x1)
        x1 = self.tanh(x1)
        return x1
        
        
        
class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        # initialize model configuration from passed-in config
        self.kernel_size = config.D.kernel_size
        self.n_channels = config.D.n_channels
        self.n_blocks = config.D.n_blocks
        self.fc_size = config.D.fc_size

        in_channels = 3
        self.conv_blocks = nn.Sequential()
        for i in range(self.n_blocks):
            out_channels = (self.n_channels if i==0 else in_channels * 2) if i % 2==0 else in_channels
            self.conv_blocks.add_module("Conv_{}".format(i+1), ConvolutionBlock(in_channels, 
                                                                    out_channels, 
                                                                    kernel_size=self.kernel_size,
                                                                   padding=1,
                                                                   stride=1 if i%2==0 else 2))
            self.conv_blocks.add_module("LeakyReLU_{}".format(i+1), nn.LeakyReLU(0.2))
            in_channels = out_channels
        self.adaptive_pool = nn.AdaptiveAvgPool2d((6,6))
        self.fc1 = nn.Linear(out_channels * 6 * 6, self.fc_size)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(self.fc_size, 1)
        """
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(out_channels, self.fc_size, kernel_size=1)
        self.leakyrelu = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(self.fc_size, 1, kernel_size=1)"""
    
    def forward(self, x):
        batch_size = x.shape[0]
        """x = self.conv_blocks(x)
        x = self.adaptive_pool(x)
        x = self.conv1(x)
        x = self.leakyrelu(x)
        x = self.conv2(x)
        """
        x = self.conv_blocks(x)
        x = self.adaptive_pool(x)
        x = self.fc1(x.view(batch_size, -1))
        x = self.leakyrelu(x)
        x = self.fc2(x)
        #x = x.view(batch_size, -1)
        return x


class TruncatedVGG19(nn.Module):
    """
    A truncated VGG19 network, such that its output is the 'feature map obtained by the j-th convolution (after activation)
    before the i-th maxpooling layer within the VGG19 network'
    """
    def __init__(self, i, j):
        super(TruncatedVGG19, self).__init__()
        
        self.vgg = vgg19(pretrained=True) # use pretrained VGG19 for loss-related calculation

        # inspired by: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Super-Resolution/blob/master/models.py
        pool_amount = 0
        conv_amount = 0
        trunc_at = 0
        for layer in self.vgg.features.children():
            trunc_at += 1 

            # count pool layers and convolutional layers after each max pool
            if isinstance(layer, nn.Conv2d):
                conv_amount += 1
            if isinstance(layer, nn.MaxPool2d):
                pool_amount += 1
                conv_amount = 0

            if pool_amount==i-1 and conv_amount==j:
                break
        # truncate to the jth convolution (+ activation) before the ith maxpool layer
        self.truncated_vgg19 = nn.Sequential(*list(self.vgg.features.children())[:trunc_at + 1])

    def forward(self, x):
        x = self.truncated_vgg19(x)
        return x
                
        





