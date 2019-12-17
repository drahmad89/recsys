import numpy
import pandas
import torch
import sklearn
from torch.utils import data
from query import Database_connector
from sklearn.model_selection import train_test_split

class Dataset(data.Dataset):

    def __init__(self, table):
        self.table = table

    def __len__(self):
        return len(self.table.size)

    def __getitem__(self, index):
        return index

params = {'batch_size':64,
            'shuffle':True,
            'num_workers':6}

database_connector = Database_connector('localhost', 'root', 'fadwa123', 'rs_101')
database_connector.query("SELECT AuthenticationID, MOBI_MCC_ID, MerchantID, F_score,\
                         M_score, L_score, P_score FROM UserSnapshots WHERE SnapshotTag = \
                         '20180901-20190401' AND UserID IS NULL")
table_subset = databse_connector.columns_subset("AuthenticationID", "MOBI_MCC_ID", "M_score").astype(int)

data = table_subset.values

train_data, test_data = train_test_split(data, test_size=0.25)

train_data = pd.DataFrame(train_data)
test_data = pd.DataFrame(test_data)

train_set =  Dataset(train_data)
train_loader = torch.utils.data.Dataloader(train_set, params)

test_set =  Dataset(test_data)
test_loader = torch.utils.data.Dataloader(test_set, params)

