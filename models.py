import numpy as np
from torch import nn
from torch_geometric import nn as geometric_nn
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data, Batch

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


class CNN_GNN_deep_3(nn.Module):
    def __init__(self, input_size=25, hidden_features=100):
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
        # GNN
        self.gnn = geometric_nn.Sequential('x, edge_index', [
            (SAGEConv(3+2816, hidden_features, aggr='max'),  'x, edge_index -> x'), # max, mean, add ...
            nn.LogSoftmax(dim=1),
            (SAGEConv(hidden_features, 1, aggr='mean'), 'x, edge_index -> x'), # max, mean, add ...
            ])
    def forward(self, X_batch, data_batch, device): # data_batch is a list of Data objects
        encoding = self.encoder(X_batch)
        for i, data in enumerate(data_batch):
            data = data.to(device)
            features = torch.zeros((data.num_nodes, 3 + encoding.shape[1])).to(device)
            features[:,:3] = data.x[:,:3]
            features[:,3:] = encoding[i,:]
            data.__setitem__('x', features)
        data_batch = Batch.from_data_list(data_batch)
        y_pred = self.gnn(data_batch.x, data_batch.edge_index)
        return y_pred, data_batch.y


class CNN_GNN_deep_3layers(nn.Module):
    def __init__(self, input_size=25, hidden_features=100):
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
        # GNN
        self.gnn = geometric_nn.Sequential('x, edge_index', [
            (SAGEConv(3+2816, hidden_features, aggr='max'),  'x, edge_index -> x'), # max, mean, add ...
            nn.ReLU(),
            (SAGEConv(hidden_features, 50, aggr='mean'), 'x, edge_index -> x'),
            nn.ReLU(),
            (SAGEConv(50, 1, aggr='mean'), 'x, edge_index -> x'), # max, mean, add ...
            ])
    def forward(self, X_batch, data_batch, device): # data_batch is a list of Data objects
        encoding = self.encoder(X_batch)
        for i, data in enumerate(data_batch):
            data = data.to(device)
            features = torch.zeros((data.num_nodes, 3 + encoding.shape[1])).to(device) 
            features[:,:3] = data.x[:,:3]
            features[:,3:] = encoding[i,:]
            data.__setitem__('x', features)
        data_batch = Batch.from_data_list(data_batch)
        y_pred = self.gnn(data_batch.x, data_batch.edge_index)
        return y_pred, data_batch.y


