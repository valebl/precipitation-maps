import torch.nn as nn

class Conv_autoencoder(nn.Module):
    def __init__(self, input_size=25):
        super().__init__()
       
        #Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(input_size, 32, kernel_size=(3,3,5), padding=1, stride=2), 
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=(3,3,5), padding=1, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1280,12)
        )
        
        #Decoder
        self.decoder = nn.Sequential(
            nn.Linear(12,1280),
            nn.Unflatten(-1,(64, 2, 2, 5)),
            nn.ConvTranspose3d(64, 32, kernel_size=(3,3,5), padding=1, stride=2),
            nn.ReLU(),
            nn.ConvTranspose3d(32, input_size, kernel_size=(3,3,6), padding=1, stride=2, output_padding=1)
        )
        
    def forward(self, X):
#         return self.encoder(X)
        return self.decoder(self.encoder(X))