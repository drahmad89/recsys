#!/usr/bin/env python
import mysql.connector as mariadb
import numpy as np
import pandas as pd
import pdb
import os
import sys


def dataloading(user_column, item_column, feature, file_dir_name,  query = False):
    database_connector = Database_connector('localhost', 'root', 'fadwa123',
                                            'recsys_1')
    if query:
        database_connector.query(("SELECT {}, {}, {} 
                                 FROM carholdermccscorings 
                                 WHERE SnapshotTag =
                                 '20170401-20190401'").format(user_column,
                                                             item_column,
                                                            feature))
    database_connector.query(query=None)
    database_connector.columns_subset(user_column,item_column,feature)
    database_connector.hashmap_pk(user_column, item_column, query=True)
    database_connector.pivot(user_column, item_column, feature)
    database_connector.save_df(dataframe=file_dir_name)

class Database_connector:

    def __init__(self, host, user, password, database):
        try:
            self.mariadb_connection = mariadb.connect(host='localhost', user= user,
                                                      password=password,
                                                      database='recsys_1')
        except:
            print('Connection to database has failed!!')
            self.mariadb_connection = None
        self.df = None

    def query(self, query=None, direc = None):
        if query:
            self.df =  pd.read_sql(query, con=self.mariadb_connection)
            return self.df

        self.df = pd.read_pickle(direc, compression=None)
        return self.df

    def columns_subset(self, *args):
        self.df = self.df[list(args)]
        return

    def _hashmap(self, df, column, column_key):
        values = sorted(df[column].unique())
        d = dict([(y, x) for x, y in enumerate(values)])
        df[column_key] = df[column].map(d)
        return df

    def hashmap_pk(self,user_column, item_column,  query = None):
        if query:
            self.df = self._hashmap(self.df,user_column, 'UserKey')
            self.df = self._hashmap(self.df, item_column, 'ItemKey')
            self.df.sort_values(by=['UserKey', 'ItemKey'], inplace= True)
        return self.df

    def pivot(self, index, columns, values):
        self.df = self.df.pivot(index= index, columns=columns, values=values)
        return self.df

    def save_df(self, dataframe, direc='./', name= 'df.pkl'):
        try:
            dataframe.to_pickle(direc+name)
            print("Dataframe saved to {}{}".format(direc,name))
            return True 
        except:
            print("Could not save the dataframe: {}".format(dataframe))
            return False

    def __del__(self):
        try:
            print ("Closing mariadb connection session...\n")
            self.mariadb_connection.close()
        except:
            print("Loading data is done!")

if __name__ == "__main__":
    test_class('CardholderID', 'MOBI_MCC_ID', 'M_score')
