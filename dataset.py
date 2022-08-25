import numpy as np
import pickle

import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_convert

from torch_geometric.data import Data, Batch

class Clima_dataset(Dataset):

    def _load_data_into_memory(self, path, input_file, target_file, data_file, idx_file):
        with open(path + input_file, 'rb') as f:
            input = pickle.load(f) # input.shape = (n_levels, lon_dim, lat_dim, time_year_dim)
        with open(path + target_file, 'rb') as f:
            target = pickle.load(f)
        with open(path + data_file, 'rb') as f:
            data = pickle.load(f)
        with open(path + idx_file,'rb') as f:
            idx_to_key = pickle.load(f)
        return input, target, data, idx_to_key

    def __init__(self, path, input_file, target_file, data_file, idx_file,**kwargs):
        super().__init__()
        self.PAD = 2
        self.LAT_DIM = 43
        self.LON_DIM = 49
        self.SPACE_IDXS_DIM = self.LAT_DIM * self.LON_DIM

        self.path = path
        self.input_file, self.target_file, self.data_file, self.idx_file = input_file, target_file, data_file, idx_file

        self.input, self.target, self.data, self.idx_to_key = self._load_data_into_memory(self.path,
                self.input_file, self.target_file, self.data_file, self.idx_file)
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
        input = self.input[:, time_idx - 25 : time_idx, lat_idx + self.PAD - 2 : lat_idx + self.PAD + 4, lon_idx + self.PAD - 2 : lon_idx + self.PAD + 4]
        #-- derive gnn data
        edge_index = torch.tensor(self.data[space_idx]['edge_index'])
        x = torch.tensor(self.data[space_idx]['x'])
        y = torch.tensor(self.target[k])
        data = Data(x=x, edge_index=edge_index, y=y)
        return input, data

def custom_collate_fn(batch):
    input = np.array([item[0] for item in batch])
    data = [item[1] for item in batch]
    input = default_convert(input)
    input.requires_grad = True
    #data = Batch.from_data_list(data)
    #data = default_convert(data)
    return [input, data]
