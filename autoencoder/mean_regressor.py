import numpy as np
import torch
from torch import nn
from torch.nn.functional import relu
import sys


#double 3x3 convolution
def dual_conv(in_channel, out_channel, kernel_size=3, padding=1):
    conv = nn.Sequential(
        nn.Conv3d(in_channel, out_channel, kernel_size=kernel_size, padding=padding),
        nn.BatchNorm3d(out_channel),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_channel, out_channel, kernel_size=kernel_size, padding=padding),
        nn.BatchNorm3d(out_channel),
        nn.ReLU(inplace=True)
    )
    return conv

def linear(in_features, out_features):
    lin = nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.ReLU(inplace=True)
    )
    return lin

class Mean_regressor(nn.Module):
    def __init__(self, input_channels=25):
        super(Mean_regressor, self).__init__()            

        # Encoder (contracting path) -> this is the same of the Unet model
        self.dwn_conv1 = dual_conv(input_channels, 64)
        self.dwn_conv2 = dual_conv(64, 128)
        self.dwn_conv3 = dual_conv(128, 256)
        self.dwn_conv4 = dual_conv(256, 512)
        self.dwn_conv5 = dual_conv(512, 1024)
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2, padding=1)        
        self.flatten = nn.Flatten() # 12288

        #Regressor
        self.regr1 = linear(12288, 1)
        
    def forward(self, image):

        #forward pass for Encoder
        y = self.dwn_conv1(image)
        y = self.maxpool(y)
        y = self.dwn_conv2(y)
        y = self.maxpool(y)
        y = self.dwn_conv3(y)
        y = self.maxpool(y)
        y = self.dwn_conv4(y)
        y = self.maxpool(y)
        y = self.dwn_conv5(y)
        y = self.flatten(y)

        #forward pass for Regressor
        y = self.regr1(y)

        return y
