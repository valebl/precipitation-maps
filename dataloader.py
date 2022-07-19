import torch
import numpy as np
import pickle
from torch.utils.data import Dataset


class Clima_dataset(Dataset):

    def _load_data_into_memory(self, input_path, target_path):
        with open(input_path, 'rb') as f:
            input_data = pickle.load(f) # input.shape = (lon_dim, lat_dim, n_levels, time_year_dim)
        with open(target_path, 'rb') as f:
            target_data = pickle.load(f)
        return input_data, target_data

    def __init__(self, input_path, target_path, **kwargs):
        super().__init__()
        self.pad = 2
        self.input_path = input_path
        self.target_path = target_path
        self.input_data, self.target_data = self._load_data_into_memory(self.input_path, self.target_path)
        self.lon_size = self.input_data.shape[0]
        self.lat_size = self.input_data.shape[1]
        self.idxs = np.array([[(i,j) for j in range(self.pad,self.lat_size-(self.pad+1))] for i in range(self.pad,self.lon_size-(self.pad+1))])
        self.idxs = self.idxs.reshape(-1, self.idxs.shape[-1])

    def __len__(self):
        return self.idxs.shape[0]

    def __getitem__(self, idx):
        lon_idx = self.idxs[idx][0]
        lat_idx = self.idxs[idx][1]
        input = self.input_data[lon_idx-self.pad:lon_idx+self.pad+2, lat_idx-2:lat_idx+self.pad+2, :, :]
        target = self.target_data[lon_idx-2, lat_idx-2]
        return input, target
        
def custom_collate_fn(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    data = torch.tensor(data, dtype=torch.float32)
    return [data, target]
