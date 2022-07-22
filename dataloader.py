import torch
import numpy as np
import pickle
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_convert

class Clima_dataset(Dataset):

    def _load_data_into_memory(self, input_path, target_path):
        with open(input_path, 'rb') as f:
            input_data = pickle.load(f) # input.shape = (n_levels, lon_dim, lat_dim, time_year_dim)
        with open(target_path, 'rb') as f:
            target_data = pickle.load(f)
        return input_data, target_data

    def __init__(self, input_path, target_path, **kwargs):
        super().__init__()
        self.PAD = 2
        self.TIME = 140256 - 24
        self.input_path = input_path
        self.target_path = target_path
        self.input_data, self.target_data = self._load_data_into_memory(self.input_path, self.target_path)
        self.lon_size = self.input_data.shape[2]
        self.lat_size = self.input_data.shape[3]
        self.idxs_space = np.array([[(i,j) for j in range(self.PAD,self.lat_size-(self.PAD+1))] for i in range(self.PAD,self.lon_size-(self.PAD+1))])
        self.idxs_space = self.idxs_space.reshape(-1, self.idxs_space.shape[-1]) # flatten the lon and lat dimensions into a single dimension
        self.idxs = np.tile(self.idxs_space, (self.TIME,1))

    def __len__(self):
        return self.idxs_space.shape[0] * self.TIME

    def __getitem__(self, idx):
        idx_time = idx // self.idxs_space.shape[0] + 24
        idx_space = idx % self.idxs_space.shape[0]        
        lon_idx = self.idxs_space[idx_space][0]
        lat_idx = self.idxs_space[idx_space][1]
        input = self.input_data[:, :, lon_idx-self.PAD:lon_idx+self.PAD+2, lat_idx-2:lat_idx+self.PAD+2, idx_time-24:idx_time+1]
        target = self.target_data[lon_idx-2, lat_idx-2].copy()
        target['pr'] = self.target_data[lon_idx-2, lat_idx-2]['pr'][idx_time,:].copy()
        return input, target

def custom_collate_fn(batch):
    data = np.array([item[0] for item in batch])
    target = [item[1] for item in batch]
    data = default_convert(data)
    data = torch.flatten(data, start_dim=1, end_dim=2)
    data.requires_grad =True
    target = target
    return [data, target]