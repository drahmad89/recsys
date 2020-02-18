import numpy as np
import math
import pandas as pd
import torch
import sklearn
from torch.utils import data
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

class Dataset(data.Dataset):

    def __init__(self, table):
        self.table = table

    def __len__(self):
        return len(self.table)

    def __getitem__(self, index):
        return torch.from_numpy(self.table[index]).type(torch.DoubleTensor).cuda()


def data_split(data, test_size= 0.25):
    data = data.astype(np.float32)
    binTable = np.isfinite(data)
    user_count = np.sum(binTable, axis = 1)
    item_count = np.sum(binTable, axis = 0)
    item_idx = np.where(item_count > 1)[0]
    user_idx = np.where(user_count > 1)[0]
    assert user_idx.any(), "Are you using a unique table??"

    user_unique_idx = list(set(range(data.shape[0])) - set (user_idx))
    item_unique_idx = list(set(range(data.shape[1])) - set (item_idx))

    data_non_unique = data[user_idx][:, item_idx]
    data_unique = data[user_unique_idx][:, item_unique_idx]

    #user_unique = [x for x in range(data.shape[0]) if x not in  user_idx]
    #item_unique = [x for x in range(data.shape[1]) if x not in  item_idx]

    dupl_entry_x, dupl_entry_y = np.where(np.isfinite(data_non_unique))

    dupl_entry = [(x, y, data_non_unique[x,y]) for x,y in zip(dupl_entry_x,
                                                             dupl_entry_y)]
    #dupl_entry = [(row,column,data[row,column]) for column in item_idx for row
    #              in user_idx if not math.isnan(data[row,column]) ]

    unique_entry_x, unique_entry_y = np.where(np.isfinite(data_unique))

    unique_entry = [(x, y, data_unique[x,y]) for x,y in zip(unique_entry_x,
                                                             unique_entry_y)]
    train_subset_0 , test_set = train_test_split(dupl_entry, test_size=test_size)
    train_subset_1 = unique_entry
    train_set = train_subset_0 + train_subset_1

    return train_set, test_set



def data_loading(dataFrame_dir = "/home/ahmad/recsys_data/mid_pivot.hdf"):

    params = {'batch_size':1500, 'shuffle':True,}
    table_pivot = pd.read_hdf(dataFrame_dir, mode='r')
    table_data = table_pivot.values.astype(np.float)
    train_data, test_data = data_split(table_data, test_size=0.2)
    train_set =  Dataset(np.array(train_data))
    train_loader = DataLoader(train_set, **params)
    test_set =  Dataset(np.array(test_data))
    test_loader =DataLoader(test_set, **params)

    return table_pivot, train_loader, test_loader 

