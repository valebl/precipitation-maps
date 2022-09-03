import numpy as np
import torch
from torch import nn
from torch.nn.functional import relu
import sys

class Conv_autoencoder(nn.Module):
    def __init__(self, input_size=25):
        super().__init__()
        #Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(input_size, 32, kernel_size=(7,5,5), padding=0, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        #Decoder
        self.decoder = nn.Sequential(
            nn.Unflatten(-1,(32, 19, 2, 2)),
            nn.ReLU(),
            nn.ConvTranspose3d(32, input_size, kernel_size=(7,5,5), padding=0, stride=1)
        )
    def forward(self, X):
        return self.decoder(self.encoder(X))


class Conv_autoencoder_deep(nn.Module):
    def __init__(self, input_size=25):
        super().__init__()
        #Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(input_size, 32, kernel_size=(5,3,3), padding=(1,1,1), stride=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(3,3,3), padding=0, stride=1),
            nn.Conv3d(32, 16, kernel_size=(5,3,3), padding=0, stride=1),
            nn.ReLU(),
            nn.Flatten() # 1088
            )
        #Decoder
        self.decoder = nn.Sequential(
            nn.Unflatten(-1,(16, 17, 2, 2)),
            nn.ReLU(),
            nn.ConvTranspose3d(16, 32, kernel_size=(5,3,3), padding=0, stride=1),
            nn.Upsample(size=(21,4,4)),
            nn.ReLU(),
            nn.ConvTranspose3d(32, input_size, kernel_size=(5,3,3), padding=0, stride=1)
        )
    def forward(self, X):
        return self.decoder(self.encoder(X))

class Conv_autoencoder_deep_2(nn.Module):
    def __init__(self, input_size=25):
        super().__init__()
        #Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(input_size, 32, kernel_size=(5,3,3), padding=(0,1,1), stride=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=(5,3,3), padding=(0,1,1), stride=1),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=(5,3,3), padding=0, stride=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(3,3,3), padding=0, stride=1),
            nn.Flatten(), # 2816
            nn.Linear(2816, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            )
        #Decoder
        self.decoder = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2816),
            nn.Unflatten(-1,(64, 11, 2, 2)),
            nn.Upsample(size=(13,4,4)),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 64, kernel_size=(5,3,3), padding=0, stride=1),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 32, kernel_size=(5,3,3), padding=(0,1,1), stride=1),
            nn.ReLU(),
            nn.ConvTranspose3d(32, input_size, kernel_size=(5,3,3), padding=(0,1,1), stride=1)
            )
    def forward(self, X):
        return self.decoder(self.encoder(X))


class Conv_autoencoder_deep_3(nn.Module):
    def __init__(self, input_size=25):
        super().__init__()
        #Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(input_size, 32, kernel_size=(5,3,3), padding=(0,1,1), stride=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=(5,3,3), padding=(0,1,1), stride=1),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=(5,3,3), padding=0, stride=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(3,3,3), padding=0, stride=1),
            nn.Flatten() # 2816
            )
        #Decoder
        self.decoder = nn.Sequential(
            nn.Unflatten(-1,(64, 11, 2, 2)),
            nn.Upsample(size=(13,4,4)),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 64, kernel_size=(5,3,3), padding=0, stride=1),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 32, kernel_size=(5,3,3), padding=(0,1,1), stride=1),
            nn.ReLU(),
            nn.ConvTranspose3d(32, input_size, kernel_size=(5,3,3), padding=(0,1,1), stride=1)
            )
    def forward(self, X):
        return self.decoder(self.encoder(X))


class Conv_predictor(nn.Module):
    def __init__(self, input_size=25):
        super().__init__()
        #Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(input_size, 32, kernel_size=(5,3,3), padding=(0,1,1), stride=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=(5,3,3), padding=(0,1,1), stride=1),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=(5,3,3), padding=0, stride=1),
            nn.ReLU(),
            #nn.MaxPool3d(kernel_size=(3,3,3), padding=0, stride=1),
            nn.Flatten() # 2816
            )
        self.regressor = nn.Sequential(
            nn.Linear(13312, 1),
            #nn.ReLU(),
            #nn.Linear(1024, 256),
            #nn.ReLU(),
            #nn.Linear(256, 1),
            nn.ReLU()
            )

    def forward(self, X):
        out = self.regressor(self.encoder(X))
        return out #torch.exp(out)


#double 3x3 convolution 
def dual_conv(in_channel, out_channel, kernel_size=3, padding=1):
    conv = nn.Sequential(
        nn.Conv3d(in_channel, out_channel, kernel_size=kernel_size, padding=padding),
        nn.BatchNorm3d(out_channel),
        nn.ReLU(inplace= True),
        nn.Conv3d(out_channel, out_channel, kernel_size=kernel_size, padding=padding),
        nn.BatchNorm3d(out_channel),
        nn.ReLU(inplace= True)
    )
    return conv

# crop the image(tensor) to equal size 
# as shown in architecture image , half left side image is concated with right side image
def crop_tensor(target_tensor, tensor):
    target_size2 = target_tensor.size()[2] 
    tensor_size2 = tensor.size()[2] 
    delta2 = tensor_size2 - target_size2 
    delta2 = delta2 // 2 
    target_size3 = target_tensor.size()[3] 
    tensor_size3 = tensor.size()[3] 
    delta3 = tensor_size3 - target_size3 
    delta3 = delta3 // 2 
    target_size4 = target_tensor.size()[4] 
    tensor_size4 = tensor.size()[4]
    delta4 = tensor_size4 - target_size4 
    delta4 = delta4 // 2 
    assert tensor_size2 >= target_size2
    assert tensor_size3 >= target_size3
    assert tensor_size4 >= target_size4
    return tensor[:, :, delta2:tensor_size2-delta2, delta3:tensor_size3-delta3, delta4:tensor_size4-delta4]  # [:, : , 1:3, 1:3]

class Unet(nn.Module):
    def __init__(self, input_channels=25):
        super(Unet, self).__init__()

        # Left side (contracting path)
        self.dwn_conv1 = dual_conv(input_channels, 64)
        self.dwn_conv2 = dual_conv(64, 128)
        self.dwn_conv3 = dual_conv(128, 256)
        self.dwn_conv4 = dual_conv(256, 512)
        self.dwn_conv5 = dual_conv(512, 1024)
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2, padding=1)
        self.dwn_conv6 = dual_conv(64,32)

        #Right side  (expansion path) 
        #transpose convolution is used showna as green arrow in architecture image
        self.trans1 = nn.ConvTranspose3d(1024,512, kernel_size=2, stride= 2, padding=1)
        self.up_conv1 = dual_conv(1024,512) # in channels = out channels of transp * 2 due to cat
        self.trans2 = nn.ConvTranspose3d(512,256, kernel_size=2, stride= 2, padding=1, output_padding=(1,1,1))
        self.up_conv2 = dual_conv(512,256)
        self.trans3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2, padding=1, output_padding=(1,0,0))
        self.up_conv3 = dual_conv(256,128)
        self.trans4 = nn.ConvTranspose3d(128,64, kernel_size=2, stride= 2, padding=1, output_padding=(1,0,0,))
        self.up_conv4 = dual_conv(128,64)

        #output layer
        self.out = nn.Conv3d(64, input_channels, kernel_size=1, padding=0)

    def forward(self, image):

        #forward pass for Left side
        x1 = self.dwn_conv1(image)
        x2 = self.maxpool(x1)
        x3 = self.dwn_conv2(x2)
        x4 = self.maxpool(x3)
        x5 = self.dwn_conv3(x4)
        x6 = self.maxpool(x5)
        x7 = self.dwn_conv4(x6)
        x8 = self.maxpool(x7)
        x9 = self.dwn_conv5(x8)

        #forward pass for Right side
        x = self.trans1(x9)
        y = crop_tensor(x, x7)
        x = self.up_conv1(torch.cat([x,y], 1))
    
        x = self.trans2(x)
        y = crop_tensor(x, x5)
        x = self.up_conv2(torch.cat([x,y], 1))     

        x = self.trans3(x)
        y = crop_tensor(x, x3)
        x = self.up_conv3(torch.cat([x,y], 1))

        x = self.trans4(x)
        y = crop_tensor(x, x1)
        x = self.up_conv4(torch.cat([x,y], 1))        

        x = self.out(x)

        return x


