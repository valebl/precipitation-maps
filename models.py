import numpy as np
import torch
from torch import nn
from torch.nn.functional import relu
from torch_geometric import nn as geometric_nn
from torch_geometric.nn import SAGEConv, GATConv, GATv2Conv
from torch_geometric.data import Data, Batch
import sys


class CNN_GRU_ae_new(nn.Module):
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

        self.linear == nn.Sequential(
            nn.Linear(output_dim, 2048),
            nn.nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.nn.BatchNorm1d(128),
            nn.ReLU()
            )

        self.decoder = nn.Sequential(
            nn.Linear(128, 512),
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
        out = self.linear(out)
        out = self.decoder(out)
        out = out.reshape(s[0], s[1], s[2], s[3], s[4], s[5])
        return out


class CNN_GRU(nn.Module):
    def __init__(self, input_size=5, input_dim=256, hidden_dim=256, output_dim=256, n_layers=2):
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
            nn.ReLU(),
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
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1),
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


class CNN_GRU_GNN_classifier(nn.Module):
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
            nn.Linear(output_dim*25, 512),
            nn.ReLU()
        )

        #gnn
        self.gnn = geometric_nn.Sequential('x, edge_index', [
            (geometric_nn.BatchNorm(3+512), 'x -> x'),
            (GATv2Conv(3+512, 128, heads=2, aggr='mean', dropout=0.5),  'x, edge_index -> x'), # max, mean, add ...
            (geometric_nn.BatchNorm(256), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(256, 128, heads=2, aggr='mean'), 'x, edge_index -> x'),
            (geometric_nn.BatchNorm(256), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(256, 128, aggr='mean'), 'x, edge_index -> x'),
            (geometric_nn.BatchNorm(128), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(128, 128, aggr='mean'), 'x, edge_index -> x'),
            (geometric_nn.BatchNorm(128), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(128, 128, aggr='mean'), 'x, edge_index -> x'),
            (geometric_nn.BatchNorm(128), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(128, 128, aggr='mean'), 'x, edge_index -> x'),
            (geometric_nn.BatchNorm(128), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(128, 2, aggr='mean'), 'x, edge_index -> x'),
            nn.Softmax(dim=-1)
            ])

    def forward(self, X_batch, data_batch, device): # data_batch is a list of Data object
        s = X_batch.shape
        X_batch = X_batch.reshape(s[0]*s[1], s[2], s[3], s[4], s[5])
        X_batch = self.encoder(X_batch.to(device))
        X_batch = X_batch.reshape(s[0], s[1], self.output_dim)
        encoding, h = self.gru(X_batch)
        encoding = encoding.reshape(s[0], s[1]*self.output_dim)
        encoding = self.decoder(encoding)
            
        for i, data in enumerate(data_batch):
            data = data.to(device)
            features = torch.zeros((data.num_nodes, 3 + encoding.shape[1])).to(device)
            features[:,:3] = data.x[:,:3]
            features[:,3:] = encoding[i,:]
            data.__setitem__('x', features)
        data_batch = Batch.from_data_list(data_batch)
        y_pred = self.gnn(data_batch.x, data_batch.edge_index)
        return y_pred, data_batch.y.squeeze().to(torch.long), None, None


class CNN_GRU_GNN_classifier_2(nn.Module):
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

        #gnn
        self.gnn = geometric_nn.Sequential('x, edge_index', [
            (GATConv(3+256*25, hidden_features, aggr='mean'),  'x, edge_index -> x'), # max, mean, add ...
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
            (GATConv(hidden_features, 2, aggr='mean'), 'x, edge_index -> x'), # max, mean, add ...
            nn.Softmax(dim=-1)
            ])

    def forward(self, X_batch, data_batch, device): # data_batch is a list of Data object
        s = X_batch.shape
        X_batch = X_batch.reshape(s[0]*s[1], s[2], s[3], s[4], s[5])
        X_batch = self.encoder(X_batch.to(device))
        X_batch = X_batch.reshape(s[0], s[1], self.output_dim)
        encoding, h = self.gru(X_batch)
        encoding = encoding.reshape(s[0], s[1]*self.output_dim)
        #encoding = self.decoder(out)
            
        for i, data in enumerate(data_batch):
            data = data.to(device)
            features = torch.zeros((data.num_nodes, 3 + encoding.shape[1])).to(device)
            features[:,:3] = data.x[:,:3]
            features[:,3:] = encoding[i,:]
            data.__setitem__('x', features)
        data_batch = Batch.from_data_list(data_batch)
        y_pred = self.gnn(data_batch.x, data_batch.edge_index)
        #print(y_pred.shape, data_batch.y.shape)
        return y_pred, data_batch.y.squeeze().to(torch.long)
        #return torch.log(torch.exp(y_pred)+1), data_batch.y


class CNN_GRU_GNN_regressor(nn.Module):
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
            nn.Linear(output_dim*25, 512),
            nn.ReLU()
        )

        #gnn
        self.gnn = geometric_nn.Sequential('x, edge_index', [
            (geometric_nn.BatchNorm(3+512), 'x -> x'),
            (GATv2Conv(3+512, 128, heads=2, aggr='mean', dropout=0.5),  'x, edge_index -> x'), # max, mean, add ...
            (geometric_nn.BatchNorm(256), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(256, 128, heads=2, aggr='mean'), 'x, edge_index -> x'),
            (geometric_nn.BatchNorm(256), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(256, 128, aggr='mean'), 'x, edge_index -> x'),
            (geometric_nn.BatchNorm(128), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(128, 128, aggr='mean'), 'x, edge_index -> x'),
            (geometric_nn.BatchNorm(128), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(128, 128, aggr='mean'), 'x, edge_index -> x'),
            (geometric_nn.BatchNorm(128), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(128, 128, aggr='mean'), 'x, edge_index -> x'),
            (geometric_nn.BatchNorm(128), 'x -> x'),
            nn.ReLU(),
            (GATv2Conv(128, 1, aggr='mean'), 'x, edge_index -> x'),
            ])

    def forward(self, X_batch, data_batch, device): # data_batch is a list of Data object
        s = X_batch.shape
        X_batch = X_batch.reshape(s[0]*s[1], s[2], s[3], s[4], s[5])
        X_batch = self.encoder(X_batch.to(device))
        X_batch = X_batch.reshape(s[0], s[1], self.output_dim)
        encoding, h = self.gru(X_batch)
        encoding = encoding.reshape(s[0], s[1]*self.output_dim)
        encoding = self.decoder(encoding)
            
        for i, data in enumerate(data_batch):
            data = data.to(device)
            features = torch.zeros((data.num_nodes, 3 + encoding.shape[1])).to(device)
            features[:,:3] = data.x[:,:3]
            features[:,3:] = encoding[i,:]
            data.__setitem__('x', features)
        data_batch = Batch.from_data_list(data_batch)
        y_pred = self.gnn(data_batch.x, data_batch.edge_index)
        mask = data_batch.mask.squeeze()
        return y_pred.squeeze()[mask], data_batch.y.squeeze()[mask], data_batch.weights[mask], data_batch.batch[mask]


class CNN_GRU_GNN_regressor_small(nn.Module):
    def __init__(self, input_size=5, input_dim=128, hidden_dim=128, output_dim=128, n_layers=2, hidden_features=128):
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
            nn.Linear(output_dim*25, 512),
            nn.ReLU()
        )

        #gnn
        self.gnn = geometric_nn.Sequential('x, edge_index', [
            (geometric_nn.BatchNorm(3+512), 'x -> x'),
            (GATv2Conv(3+512, 128, heads=1, aggr='mean', dropout=0.5),  'x, edge_index -> x'), # max, mean, add ...
            (geometric_nn.BatchNorm(128), 'x -> x'),
            nn.ReLU(),
            #(GATv2Conv(256, 128, heads=2, aggr='mean'), 'x, edge_index -> x'),
            #(geometric_nn.BatchNorm(256), 'x -> x'),
            #nn.ReLU(),
            #(GATv2Conv(256, 128, aggr='mean'), 'x, edge_index -> x'),
            #(geometric_nn.BatchNorm(128), 'x -> x'),
            #nn.ReLU(),
            #(GATv2Conv(128, 128, aggr='mean'), 'x, edge_index -> x'),
            #(geometric_nn.BatchNorm(128), 'x -> x'),
            #nn.ReLU(),
            #(GATv2Conv(128, 128, aggr='mean'), 'x, edge_index -> x'),
            #(geometric_nn.BatchNorm(128), 'x -> x'),
            #nn.ReLU(),
            #(GATv2Conv(128, 128, aggr='mean'), 'x, edge_index -> x'),
            #(geometric_nn.BatchNorm(128), 'x -> x'),
            #nn.ReLU(),
            (GATv2Conv(128, 1, aggr='mean'), 'x, edge_index -> x'),
            ])

    def forward(self, X_batch, data_batch, device): # data_batch is a list of Data object
        s = X_batch.shape
        X_batch = X_batch.reshape(s[0]*s[1], s[2], s[3], s[4], s[5])
        X_batch = self.encoder(X_batch.to(device))
        X_batch = X_batch.reshape(s[0], s[1], self.output_dim)
        encoding, h = self.gru(X_batch)
        encoding = encoding.reshape(s[0], s[1]*self.output_dim)
        encoding = self.decoder(encoding)
            
        for i, data in enumerate(data_batch):
            data = data.to(device)
            features = torch.zeros((data.num_nodes, 3 + encoding.shape[1])).to(device)
            features[:,:3] = data.x[:,:3]
            features[:,3:] = encoding[i,:]
            data.__setitem__('x', features)
        data_batch = Batch.from_data_list(data_batch)
        y_pred = self.gnn(data_batch.x, data_batch.edge_index)
        mask = data_batch.mask.squeeze()
        return y_pred.squeeze()[mask], data_batch.y.squeeze()[mask], data_batch.weights[mask], data_batch.batch[mask]

class CNN_GRU_GNN_regressor_2(nn.Module):
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

        #gnn
        self.gnn = geometric_nn.Sequential('x, edge_index', [
            (GATConv(3+output_dim*25, hidden_features, aggr='mean'),  'x, edge_index -> x'), # max, mean, add ...
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
            (GATConv(hidden_features, 1, aggr='mean'), 'x, edge_index -> x'),
            ])

    def forward(self, X_batch, data_batch, device): # data_batch is a list of Data object
        s = X_batch.shape
        X_batch = X_batch.reshape(s[0]*s[1], s[2], s[3], s[4], s[5])
        X_batch = self.encoder(X_batch.to(device))
        X_batch = X_batch.reshape(s[0], s[1], self.output_dim)
        encoding, h = self.gru(X_batch)
        encoding = encoding.reshape(s[0], s[1]*self.output_dim)
        #encoding = self.decoder(out)

        for i, data in enumerate(data_batch):
            data = data.to(device)
            features = torch.zeros((data.num_nodes, 3 + encoding.shape[1])).to(device)
            features[:,:3] = data.x[:,:3]
            features[:,3:] = encoding[i,:]
            data.__setitem__('x', features)
        data_batch = Batch.from_data_list(data_batch)
        y_pred = self.gnn(data_batch.x, data_batch.edge_index)
        #data_batch = data_batch.to_data_list()
        #mask = data_batch.mask # for results
        #return data_batch # for results
        mask = data_batch.mask.squeeze()
        return y_pred.squeeze()[mask], data_batch.y.squeeze()[mask], data_batch.batch[mask]


if __name__ == "__main__":

    model = CNN_GRU_ae()
    model = model.cuda()

    X = torch.ones((64,25,5,5,6,6)).cuda()

    y = model(X)

    print(y.shape)
