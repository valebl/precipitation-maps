import numpy as np
import torch
from torch import nn
from torch.nn.functional import relu
from torch_geometric import nn as geometric_nn
from torch_geometric.nn import SAGEConv, GATConv
from torch_geometric.data import Data, Batch
import sys


class Conv_Regressor(nn.Module):
    def __init__(self, input_size=25):
        super().__init__()
        #Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(input_size, 64, kernel_size=(3,3,3), padding=(0,1,1), stride=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=(3,3,3), padding=(0,1,1), stride=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2,2,2), padding=(1,1,1), stride=2),
            nn.Conv3d(64, 128, kernel_size=(3,3,3), padding=(0,1,1), stride=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(128, 128, kernel_size=(3,3,3), padding=(0,1,1), stride=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2,2,2), padding=(1,1,1), stride=2),
            nn.Conv3d(128, 256, kernel_size=(3,3,3), padding=(0,0,0), stride=1),
            nn.ReLU(),
            nn.Flatten() # 512
            )
        #Decoder
        self.regressor = nn.Sequential(
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128,1)
        )
    def forward(self, X):
        return self.regressor(self.encoder(X))


class CNN_GRU_ae(nn.Module):
    def __init__(self, input_size=5, input_dim=128, hidden_dim=128, output_dim=128, n_layers=2):
        super().__init__() 
        self.output_dim = output_dim
        self.encoder = nn.Sequential(
            nn.Conv3d(input_size, 64, kernel_size=3, padding=(1,1,1), stride=1), # input of shape = (batch_size, n_levels, n_vars, lat, lon)
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=(1,1,1), stride=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, padding=(1,1,1), stride=2),   
            nn.Conv3d(64, 256, kernel_size=3, padding=(1,1,1), stride=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, padding=(1,0,0), stride=2),   
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()            
            )   

        # define the decoder modules
        self.gru = nn.Sequential(
            nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True),        
            )

        self.decoder = nn.Sequential(
            nn.Linear(output_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 2048),
            nn.Unflatten(-1,(256, 2, 2, 2)),
            nn.Upsample(size=(3,4,4)),
            nn.ReLU(),
            nn.ConvTranspose3d(256, 64, kernel_size=3, padding=(1,1,1), stride=1),
            nn.ReLU(),
            nn.Upsample(size=(5,6,6)),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 64, kernel_size=3, padding=(1,1,1), stride=1),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 5, kernel_size=3, padding=(1,1,1), stride=1),
            )

    def forward(self, X): # X.shape = (batch_size, time, features, levels, lat, lon)
        # X.shape = (batch_size*time, features, levels, lat, lon)
        s = X.shape
        X = X.reshape(s[0]*s[1], s[2], s[3], s[4], s[5])
        X = self.encoder(X)
        X = X.reshape(s[0], s[1], self.output_dim)
        out, h = self.gru(X)
        out = out.reshape(s[0]*s[1], self.output_dim) # (batch_size*25, 128)
        out = self.decoder(out)
        out = out.reshape(s[0], s[1], s[2], s[3], s[4], s[5])
        return out


class CNN_GRU(nn.Module):
    def __init__(self, input_size=5, input_dim=128, hidden_dim=128, output_dim=128, n_layers=2):
        super().__init__() 
        self.output_dim = output_dim
        self.encoder = nn.Sequential(
            nn.Conv3d(input_size, 64, kernel_size=3, padding=(1,1,1), stride=1), # input of shape = (batch_size, n_levels, n_vars, lat, lon)
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=(1,1,1), stride=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, padding=(1,1,1), stride=2),   
            nn.Conv3d(64, 256, kernel_size=3, padding=(1,1,1), stride=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, padding=(1,0,0), stride=2),   
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()            
            )   

        # define the decoder modules
        self.gru = nn.Sequential(
            nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True),        
            )

        self.decoder = nn.Sequential( # (batch_size, 25, 128)
            nn.Flatten(),
            nn.Linear(output_dim*25, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 1)
            #nn.BatchNorm1d(128),
            #nn.ReLU(),
            #nn.Dropout(p=0.5),
            #nn.Linear(128, 1),
            )
            #nn.Conv1d(25, 24, kernel_size=3, padding=0, stride=1),
            #nn.BatchNorm1d(24),
            #nn.ReLU(),
            #nn.Dropout(p=0.2),
            #nn.Conv1d(24,16, kernel_size=3, padding=0, stride=1),
            #nn.BatchNorm1d(16),
            #nn.ReLU(),
            #nn.Dropout(p=0.2),
            #nn.MaxPool1d(kernel_size=2, padding=0, stride=2),
            #nn.Conv1d(16,8, kernel_size=3, padding=0, stride=1),
            #nn.BatchNorm1d(8),
            #nn.ReLU(),
            #nn.Flatten(),
            #nn.Linear(480, 128),
            #nn.BatchNorm1d(128),
            #nn.ReLU(),
            #nn.Linear(128, 1)
            #)
            #nn.Sigmoid()

    def forward(self, X): # X.shape = (batch_size, time, features, levels, lat, lon)
        # X.shape = (batch_size*time, features, levels, lat, lon)
        s = X.shape
        X = X.reshape(s[0]*s[1], s[2], s[3], s[4], s[5])
        X = self.encoder(X)
        X = X.reshape(s[0], s[1], self.output_dim)
        out, h = self.gru(X)
        out = self.decoder(out)
        return out

class CNN_GRU_classifier(nn.Module):
    def __init__(self, input_size=5, input_dim=256, hidden_dim=256, output_dim=256, n_layers=2):
        super().__init__() 
        self.output_dim = output_dim
        self.encoder = nn.Sequential(
            nn.Conv3d(input_size, 64, kernel_size=3, padding=(1,0,0), stride=1), # input of shape = (batch_size, n_levels, n_vars, lat, lon)
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=(1,1,1), stride=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, padding=1, stride=2),   
            nn.Conv3d(64, 256, kernel_size=3, padding=(1,1,1)),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, padding=1, stride=2),   
            nn.Flatten(),
            nn.Linear(2048, 576),
            nn.BatchNorm1d(576),
            nn.ReLU(),
            nn.Linear(576, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()            
            )   

        # define the decoder modules
        self.gru = nn.Sequential(
            nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True),        
            )

        self.decoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(output_dim*25, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 2),
            nn.Softmax(dim=-1)
            )

    def forward(self, X): # X.shape = (batch_size, time, features, levels, lat, lon)
        # X.shape = (batch_size*time, features, levels, lat, lon)
        s = X.shape
        X = X.reshape(s[0]*s[1], s[2], s[3], s[4], s[5])
        X = self.encoder(X)
        X = X.reshape(s[0], s[1], self.output_dim)
        out, h = self.gru(X)
        out = self.decoder(out)
        return out

class CNN_GRU_GNN(nn.Module):
    def __init__(self, input_size=5, input_dim=256, hidden_dim=256, output_dim=256, n_layers=2, hidden_features=256):
        super().__init__()
        self.output_dim = output_dim
        self.encoder = nn.Sequential(
            nn.Conv3d(input_size, 64, kernel_size=3, padding=(1,1,1), stride=1), # input of shape = (batch_size, n_levels, n_vars, lat, lon)
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=(1,1,1), stride=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, padding=(1,1,1), stride=2),
            nn.Conv3d(64, 256, kernel_size=3, padding=(1,1,1), stride=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, padding=(1,0,0), stride=2),
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
            )

        # define the decoder modules
        self.gru = nn.Sequential(
            nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True),
            )

        self.decoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(output_dim*25, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            )

        #gnn
        self.gnn = geometric_nn.Sequential('x, edge_index', [
            (GATConv(3+128, hidden_features, aggr='mean'),  'x, edge_index -> x'), # max, mean, add ...
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
            ])

    def forward(self, X_batch, data_batch, device): # data_batch is a list of Data object
        s = X_batch.shape
        X_batch = X_batch.reshape(s[0]*s[1], s[2], s[3], s[4], s[5])
        X_batch = self.encoder(X_batch.to(device))
        X_batch = X_batch.reshape(s[0], s[1], self.output_dim)
        out, h = self.gru(X_batch)
        encoding = self.decoder(out)
            
        for i, data in enumerate(data_batch):
            data = data.to(device)
            features = torch.zeros((data.num_nodes, 3 + encoding.shape[1])).to(device)
            features[:,:3] = data.x[:,:3]
            features[:,3:] = encoding[i,:]
            data.__setitem__('x', features)
        data_batch = Batch.from_data_list(data_batch)
        y_pred = self.gnn(data_batch.x, data_batch.edge_index)
        return torch.sigmoid(y_pred), data_batch.y.to(torch.float32)
        #return torch.log(torch.exp(y_pred)+1), data_batch.y


class CNN_GRU_stacked(nn.Module):
    def __init__(self, input_size=5, input_dim=160, hidden_dim=160, output_dim=32, n_layers=2):
        super().__init__()        
        self.encoder = nn.Sequential(
            nn.Conv2d(input_size, 64, kernel_size=3, padding=(1,1), stride=1), # input of shape = (batch_size, n_levels, n_vars, lat, lon)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=(1,1), stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, padding=1, stride=2),   
            nn.Conv2d(64, 256, kernel_size=3, padding=(0,0)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, padding=1, stride=2),   
            nn.Flatten(),
            nn.Linear(1024, 576),
            nn.BatchNorm1d(576),
            nn.ReLU(),
            nn.Linear(576, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()            
            )   

        # define the decoder modules
        self.gru = nn.Sequential(
            nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True),        
            )

        self.decoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim*25, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1),
            #nn.Sigmoid()
            )

    def forward(self, X):

        embedding_q = torch.zeros((X.shape[0],25,32)).cuda()
        embedding_t = torch.zeros((X.shape[0],25,32)).cuda()
        embedding_u = torch.zeros((X.shape[0],25,32)).cuda()
        embedding_v = torch.zeros((X.shape[0],25,32)).cuda()
        embedding_z = torch.zeros((X.shape[0],25,32)).cuda()

        for i in range(25):
            X_q = torch.squeeze(torch.squeeze(X[:,0,:,i,:,:], 3), 1)
            X_t = torch.squeeze(torch.squeeze(X[:,1,:,i,:,:], 3), 1)
            X_u = torch.squeeze(torch.squeeze(X[:,2,:,i,:,:], 3), 1)
            X_v = torch.squeeze(torch.squeeze(X[:,3,:,i,:,:], 3), 1)
            X_z = torch.squeeze(torch.squeeze(X[:,4,:,i,:,:], 3), 1)

            embedding_q[:,i,:] = self.encoder(X_q)
            embedding_t[:,i,:] = self.encoder(X_t)
            embedding_u[:,i,:] = self.encoder(X_u)
            embedding_v[:,i,:] = self.encoder(X_v)
            embedding_z[:,i,:] = self.encoder(X_z)     
            
            embedding = torch.cat((embedding_q,embedding_t,embedding_u,embedding_v,embedding_z),2).cuda() # shape(batch_size, time_size, 32)

        out, h = self.gru(embedding)
        out = self.decoder(out)
        out = torch.log(torch.exp(out)+1)
        return out


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


if __name__ == "__main__":

    model = CNN_GRU_ae()
    model = model.cuda()

    X = torch.ones((64,25,5,5,6,6)).cuda()

    y = model(X)

    print(y.shape)
