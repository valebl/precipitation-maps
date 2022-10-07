import numpy as np
import pickle
import sys

import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_convert

from torch_geometric.data import Data, Batch

class Clima_dataset(Dataset):

    def _load_data_into_memory(self, path, input_file, target_file, data_file, idx_file, net_type, mask_file, weights_file):
        with open(path + input_file, 'rb') as f:
            input = pickle.load(f) # time, features, levels,lat, lon
            if self.net_type == "cnn":
                s = input.shape
                input = input.reshape(s[0], s[1]*s[2], s[3], s[4]) # time, features*levels, lat, lon
        with open(path + idx_file,'rb') as f:
            idx_to_key = pickle.load(f)
        if net_type == "cnn" or net_type == "gnn" or net_type == "gru":
            with open(path + target_file, 'rb') as f:
                target = pickle.load(f)
        else:
            target = None
        if net_type == "gnn":
            with open(path + data_file, 'rb') as f:
                data = pickle.load(f)
            if mask_file is not None:
                with open(path + mask_file, 'rb') as f:
                    mask = pickle.load(f)
            else:
                mask = None
            if weights_file is not None:
                with open(path + weights_file, 'rb') as f:
                    weights = pickle.load(f)
            else:
                weights = None
        else:
            data, mask, weights = None, None, None

        return input, target, data, idx_to_key, mask, weights

    def __init__(self, path, input_file, target_file, data_file, idx_file, net_type, get_key=False, mask_file=None, weights_file=None, **kwargs):
        super().__init__()
        self.get_key = get_key
        self.PAD = 2
        self.LAT_DIM = 43 # number of points in the GRIPHO rectangle (0.25 grid)
        self.LON_DIM = 49
        self.SPACE_IDXS_DIM = self.LAT_DIM * self.LON_DIM
        self.SHIFT = 2 # relative shift between GRIPHO and ERA5 (idx=0 in ERA5 corresponds to 2 in GRIPHO)

        self.net_type = net_type
        self.path = path
        self.input_file, self.target_file, self.data_file, self.idx_file, self.mask_file, self.weights_file = input_file, target_file, data_file, idx_file, mask_file, weights_file

        self.input, self.target, self.data, self.idx_to_key, self.mask, self.weights = self._load_data_into_memory(self.path,
                self.input_file, self.target_file, self.data_file, self.idx_file, self.net_type, self.mask_file, self.weights_file)
        self.length = len(self.idx_to_key)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        k = self.idx_to_key[idx]
        time_idx = k // self.SPACE_IDXS_DIM
        space_idx = k % self.SPACE_IDXS_DIM
        lat_idx = space_idx // self.LON_DIM
        lon_idx = space_idx % self.LON_DIM
        #-- derive input
        if self.get_key:
            return k
        if self.net_type == "gru" or "gnn":
            input = self.input[time_idx - 24 : time_idx+1, :, :, lat_idx - self.PAD + 2 : lat_idx + self.PAD + 4, lon_idx - self.PAD + 2 : lon_idx + self.PAD + 4]
        else:
            input = self.input[time_idx - 24 : time_idx+1, :, :, lat_idx - self.PAD + 2 : lat_idx + self.PAD + 4, lon_idx - self.PAD + 2 : lon_idx + self.PAD + 4]
        #-- derive gnn data
        if self.net_type == "cnn" or self.net_type == "gnn" or self.net_type == "gru":
            y = torch.tensor(self.target[k])
            if self.net_type == "cnn" or self.net_type == "gru":
                return input, y
            else:
                edge_index = torch.tensor(self.data[space_idx]['edge_index'])
                x = torch.tensor(self.data[space_idx]['x'])
                if self.mask is not None:
                    mask = torch.tensor(self.mask[k].astype(bool))  #torch.where(y==0, False, True)
                    if self.weights is not None:
                        weights = torch.tensor(self.weights[k])
                        data = Data(x=x, edge_index=edge_index, y=y, mask=mask, weights=weights)
                    else:
                        data = Data(x=x, edge_index=edge_index, y=y, mask=mask, weights=None)
                else:
                    data = Data(x=x, edge_index=edge_index, y=y, mask=None, weights=None)
                #print(y, torch.where(y.squeeze()>=np.log(0.1),1,0))
                #sys.exit()
                return input, data
        else:
            return input

def custom_collate_fn_ae(batch):
    input = np.array(batch)
    input = default_convert(input)
    #input.requires_grad = True
    return input

def custom_collate_fn_cnn(batch):
    input = np.array([item[0] for item in batch])
    y = np.array([item[1] for item in batch])
    input = default_convert(input)
    y = default_convert(y)
    #input.requires_grad = True
    #y.requires_grad = True
    return input, y

def custom_collate_fn_gnn(batch):
    input = np.array([item[0] for item in batch])
    data = [item[1] for item in batch]
    input = default_convert(input)
    #input.requires_grad = True
    return input, data
    
def custom_collate_fn_gru(batch):
    input = np.array([item[0] for item in batch])
    y = np.array([item[1] for item in batch])
    input = default_convert(input)
    y = default_convert(y)
    #y = y.to(torch.float32)
    #input.requires_grad = True
    #y.requires_grad = True
    return input, y

def custom_collate_fn_get_key(batch):
    return batch
