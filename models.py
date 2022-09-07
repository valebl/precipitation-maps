import numpy as np
import torch
from torch import nn
from torch.nn.functional import relu
from torch_geometric import nn as geometric_nn
from torch_geometric.nn import SAGEConv, GATConv
from torch_geometric.data import Data, Batch
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


#----- Unet -----

#double 3x3 convolution
def dual_conv(in_channel, out_channel, kernel_size=3, padding=1):
    conv = nn.Sequential(
        nn.Conv3d(in_channel, out_channel, kernel_size=kernel_size, padding=padding),
        nn.BatchNorm3d(out_channel),
        nn.ReLU(),
        nn.Conv3d(out_channel, out_channel, kernel_size=kernel_size, padding=padding),
        nn.BatchNorm3d(out_channel),
        nn.ReLU()
    )
    return conv

# crop the left tensor to the same size of the right tensor (for concatenation)
def crop_tensor(target_tensor, tensor):

    target_sizes = list(target_tensor.shape[2:])
    tensor_sizes = list(tensor.shape[2:])
    delta = [(tensor_sizes[i] - target_sizes[i]) // 2 for i in range(len(target_sizes))]

    assert tensor_sizes[0] >= target_sizes[0] and tensor_sizes[1] >= target_sizes[1] and tensor_sizes[2] >= target_sizes[2]
   
    return tensor[:, :, delta[0]:tensor_sizes[0]-delta[0], delta[1]:tensor_sizes[1]-delta[1], delta[2]:tensor_sizes[2]-delta[2]]


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



##########################################################
################### CNN for regression ################### 
##########################################################

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

def linear(in_features, out_features):
    lin = nn.Sequential(
        nn.Linear(in_features, out_features),
    )
    return lin

#----- Unet based to predict mean precipitation inside ERA5 cells -----

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
        self.regr_conv1 = linear(12288, 1)

        
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
        y = torch.exp(y)

        return y

def conv(in_channel, out_channel, kernel_size=3, padding=1):
    conv = nn.Sequential(
        nn.Conv3d(in_channel, out_channel, kernel_size=kernel_size, padding=padding),
        nn.BatchNorm3d(out_channel),
        nn.LeakyReLU()
    )
    return conv

class Mean_cnn_regressor(nn.Module):
    def __init__(self, input_channels=25):
        super(Mean_cnn_regressor, self).__init__()            

        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2, padding=1)        
        self.flatten = nn.Flatten()

        # Encoder (contracting path) -> this is the same of the Unet model
        self.dwn_conv1 = dual_conv(input_channels, 64)
        self.dwn_conv2 = dual_conv(64, 128)
        self.dwn_conv3 = dual_conv(128, 256)
        self.dwn_conv4 = dual_conv(256, 512)
        self.dwn_conv5 = dual_conv(512, 1024)       

        #Regressor
        self.regr_conv1 = conv(1024, 512)
        self.regr_conv2 = conv(512,256)
        self.regr_conv3 = conv(256,128)
        self.regr_conv4 = conv(128, 64)
        self.regr_conv5 = conv(64, 24)
        self.regr_lin1 = linear(192,1)

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

        #forward pass for Regressor
        y = self.regr_conv1(y)
        y = self.maxpool(y)
        y = self.regr_conv2(y)
        y = self.maxpool(y)
        y = self.regr_conv3(y)
        y = self.maxpool(y)
        y = self.regr_conv4(y)
        y = self.maxpool(y)
        y = self.regr_conv5(y)
        y = self.flatten(y)
        y = self.regr_lin1(y)

        return torch.log(torch.exp(y)+1)


##########################################################
########################### GNN ########################## 
##########################################################


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

class CNN_GNN_3layers_SAGEConv(nn.Module):
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
            nn.ReLU()
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


class CNN_GNN_7layers_SAGEConv(nn.Module):
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
        # GNN
        self.gnn = geometric_nn.Sequential('x, edge_index', [
            (SAGEConv(3+2816, 2048, aggr='mean'),  'x, edge_index -> x'), # max, mean, add ...
            nn.ReLU(),
            (SAGEConv(2048, 1024, aggr='mean'), 'x, edge_index -> x'),
            nn.ReLU(),
            (SAGEConv(1024, 512, aggr='mean'), 'x, edge_index -> x'),
            nn.ReLU(),
            (SAGEConv(512, 128, aggr='mean'), 'x, edge_index -> x'),
            nn.ReLU(),
            (SAGEConv(128, 64, aggr='mean'), 'x, edge_index -> x'),
            nn.ReLU(),
            (SAGEConv(64, 32, aggr='mean'), 'x, edge_index -> x'),
            nn.ReLU(),
            (SAGEConv(32, 1, aggr='mean'), 'x, edge_index -> x'), # max, mean, add ...
            nn.ReLU()
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


class CNN_GNN_7layers_GATConv(nn.Module):
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
            # GNN
        self.gnn = geometric_nn.Sequential('x, edge_index', [
            (GATConv(3+2816, 2048, aggr='mean'),  'x, edge_index -> x'), # max, mean, add ...
            nn.ReLU(),
            (GATConv(2048, 1024, aggr='mean'), 'x, edge_index -> x'),
            nn.ReLU(),
            (GATConv(1024, 512, aggr='mean'), 'x, edge_index -> x'),
            nn.ReLU(),
            (GATConv(512, 128, aggr='mean'), 'x, edge_index -> x'),
            nn.ReLU(),
            (GATConv(128, 64, aggr='mean'), 'x, edge_index -> x'),
            nn.ReLU(),
            (GATConv(64, 32, aggr='mean'), 'x, edge_index -> x'),
            nn.ReLU(),
            (GATConv(32, 1, aggr='mean'), 'x, edge_index -> x'), # max, mean, add ...
            nn.ReLU()
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


class GATConv_1(nn.Module):
    def __init__(self, input_size=25, hidden_features=1024):
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
            (GATConv(3+2816, hidden_features, aggr='mean'),  'x, edge_index -> x'), # max, mean, add ...
            nn.ReLU(),
            (GATConv(hidden_features, hidden_features, aggr='mean'), 'x, edge_index -> x'),
            nn.ReLU(),
            (GATConv(hidden_features, hidden_features, aggr='mean'), 'x, edge_index -> x'),
            nn.ReLU(),
            (GATConv(hidden_features, hidden_features, aggr='mean'), 'x, edge_index -> x'),
            nn.ReLU(),
            (GATConv(hidden_features, hidden_features, aggr='mean'), 'x, edge_index -> x'),
            nn.ReLU(),
            (GATConv(hidden_features, hidden_features, aggr='mean'), 'x, edge_index -> x'),
            nn.ReLU(),
            (GATConv(hidden_features, 1, aggr='mean'), 'x, edge_index -> x'), # max, mean, add ...
            nn.ReLU()
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


class GATConv_2(nn.Module):
    def __init__(self, input_size=25, hidden_features=1024):
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
            (GATConv(3+2816, hidden_features, aggr='mean'),  'x, edge_index -> x'), # max, mean, add ...
            (GATConv(hidden_features, hidden_features, aggr='mean'), 'x, edge_index -> x'),
            (GATConv(hidden_features, hidden_features, aggr='mean'), 'x, edge_index -> x'),
            (GATConv(hidden_features, hidden_features, aggr='mean'), 'x, edge_index -> x'),
            (GATConv(hidden_features, hidden_features, aggr='mean'), 'x, edge_index -> x'),
            (GATConv(hidden_features, hidden_features, aggr='mean'), 'x, edge_index -> x'),
            (GATConv(hidden_features, 1, aggr='mean'), 'x, edge_index -> x'), # max, mean, add ...
            nn.ReLU()
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


class Unet_GNN(nn.Module):
    def __init__(self, input_channels=25, hidden_features=1024):
        super().__init__()

        # Encoder (contracting path) -> this is the same of the Unet model
        self.dwn_conv1 = dual_conv(input_channels, 64)
        self.dwn_conv2 = dual_conv(64, 128)
        self.dwn_conv3 = dual_conv(128, 256)
        self.dwn_conv4 = dual_conv(256, 512)
        self.dwn_conv5 = dual_conv(512, 1024)
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2, padding=1)
        self.flatten = nn.Flatten() # 1228

        #gnn
        self.gnn = geometric_nn.Sequential('x, edge_index', [
            (GATConv(3+12288, hidden_features, aggr='mean'),  'x, edge_index -> x'), # max, mean, add ...
            (GATConv(hidden_features, hidden_features, aggr='mean'), 'x, edge_index -> x'),
            (GATConv(hidden_features, hidden_features, aggr='mean'), 'x, edge_index -> x'),
            (GATConv(hidden_features, hidden_features, aggr='mean'), 'x, edge_index -> x'),
            (GATConv(hidden_features, hidden_features, aggr='mean'), 'x, edge_index -> x'),
            (GATConv(hidden_features, hidden_features, aggr='mean'), 'x, edge_index -> x'),
            (GATConv(hidden_features, 1, aggr='mean'), 'x, edge_index -> x'), # max, mean, add ...
            ])

    def forward(self, X_batch, data_batch, device): # data_batch is a list of Data objects

        #forward pass for Encoder
        encoding = self.dwn_conv1(X_batch.cuda())
        encoding = self.maxpool(encoding)
        encoding = self.dwn_conv2(encoding)
        encoding = self.maxpool(encoding)
        encoding = self.dwn_conv3(encoding)
        encoding = self.maxpool(encoding)
        encoding = self.dwn_conv4(encoding)
        encoding = self.maxpool(encoding)
        encoding = self.dwn_conv5(encoding)
        encoding = self.flatten(encoding)

        for i, data in enumerate(data_batch):
            data = data.to(device)
            features = torch.zeros((data.num_nodes, 3 + encoding.shape[1])).to(device)
            features[:,:3] = data.x[:,:3]
            features[:,3:] = encoding[i,:]
            data.__setitem__('x', features)
        data_batch = Batch.from_data_list(data_batch)
        y_pred = self.gnn(data_batch.x, data_batch.edge_index)

        return torch.log(torch.exp(y_pred)+1), data_batch.y


