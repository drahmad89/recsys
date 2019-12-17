import numpy
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
        return len(self.table.size)

    def __getitem__(self, index):
        return [index]

def data_loading():
    params = {'batch_size':64,
                'shuffle':True,
                }

    database_connector = Database_connector('localhost', 'root', 'fadwa123', 'rs_101')
    database_connector.query("SELECT AuthenticationID, MOBI_MCC_ID, MerchantID, F_score,\
                             M_score, L_score, P_score FROM UserSnapshots WHERE SnapshotTag = \
                             '20180901-20190401' AND UserID IS NULL")
    _  = database_connector.hashmap_pk( 'MOBI_MCC_ID')
    table_subset = database_connector.columns_subset("UserKey", "ItemKey", "M_score")

    table_pivot = database_connector.pivot('UserKey', 'ItemKey', 'M_score')

    table_data = table_subset.values

    train_data, test_data = train_test_split(table_data, test_size=0.25)

    train_data = pd.DataFrame(train_data)
    test_data = pd.DataFrame(test_data)

    train_set =  Dataset(train_data.to_numpy())
    train_loader = DataLoader(train_set, **params)

    test_set =  Dataset(test_data.to_numpy())
    test_loader =DataLoader(test_set, params)

    return table_pivot, train_loader, test_loader 

