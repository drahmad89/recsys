import numpy as np
import math
import pandas as pd
import torch
import sklearn
from torch.utils import data
from torch.utils.data import DataLoader
from db_connector import Database_connector
from sklearn.model_selection import train_test_split
class Dataset(data.Dataset):

    def __init__(self, table):
        self.table = table

    def __len__(self):
        return len(self.table)

    def __getitem__(self, index):
        return torch.from_numpy(self.table[index]).type(torch.DoubleTensor).cuda()


def data_split(data, test_size = 0.25):
    #triplets=[]
    #user_track={}
    #item_track={}
    #for row in range(data.shape[0]):
    #    row_counter = 0
    #    for column in range(data.shape[1]):
    #        if not data[row, column]:
    #            continue

    #        if column in item_track.keys:
    #            item_track[column]+=1

    #        else:
    #            item_track[column]=1

    #        row_counter += 1
    #        triplets.append((row, column, data[row, column]))
    #    user_track[row] = row_counter
    binTable = np.isfinite(data)
    user_count = np.sum(binTable, axis = 1)
    item_count = np.sum(binTable, axis = 0)
    item_idx = np.where(item_count > 1)[0]
    user_idx = np.where(user_count > 1)[0]

    user_unique = [x for x in range(data.shape[0]) if x not in  user_idx]
    item_unique = [x for x in range(data.shape[1]) if x not in  item_idx]

    dupl_entry = [(row,column,data[row,column]) for column in item_idx for row
                  in user_idx if not math.isnan(data[row,column]) ]
    train_subset_0 , test_set = train_test_split(dupl_entry, test_size=test_size)
    train_subset_1 = [(row, column,data[row,column]) for column in item_unique
                      for row in user_unique if not math.isnan(data[row,column])]
    train_set = train_subset_0 + train_subset_1

    return train_set, test_set



def data_loading(dataFrame_dir = "/home/ahmad/recsys_data/pivot3.pkl"):

    table_pivot = pd.read_hdf('./mid_pivot.hdf', mode='r')
    table_data = table_pivot.values

    train_data, test_data = data_split(table_data, test_size=0.25)

    train_set =  Dataset(np.array(train_data))
    train_loader = DataLoader(train_set, **params)

    test_set =  Dataset(np.array(test_data))
    test_loader =DataLoader(test_set, **params)

    return table_pivot, train_loader, test_loader 

