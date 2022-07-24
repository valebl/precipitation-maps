import torch.nn as nn

class Conv_autoencoder(nn.Module):
    def __init__(self, input_size=25):
        super().__init__()

        #Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(input_size, 16, kernel_size=(3,3,3), padding=1, stride=2),
#             nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 8, kernel_size=(3,3,3), padding=1, stride=2),
            #nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Flatten(),
#             nn.Linear(22,224)
        )

        #Decoder
        self.decoder = nn.Sequential(
#             nn.Linear(8,224),
            nn.Unflatten(-1,(8, 2, 2, 7)),
            nn.ConvTranspose3d(8, 16, kernel_size=(3,3,3), padding=1, stride=2),
            #nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.ConvTranspose3d(16, input_size, kernel_size=(3,3,3), padding=1, stride=2, output_padding=(1,1,0))
        )
    
    def forward(self, X):
#         return self.encoder(X)
        return self.decoder(self.encoder(X))
