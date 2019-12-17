#!/usr/bin/env python
import mysql.connector as mariadb
import pandas as pd
import numpy as np
import pdb

def test_class():
    database_connector = Database_connector('localhost', 'root', 'fadwa123', 'rs_101')
    database_connector.query("SELECT AuthenticationID, MOBI_MCC_ID, MerchantID, F_score, M_score, L_score, P_score FROM UserSnapshots WHERE SnapshotTag = '20180901-20190401' AND UserID IS NULL")
    database_connector.pivot("AuthenticationID", "MOBI_MCC_ID", "M_score")
    database_connector.save_df(dataframe='pivot_df') 


class Database_connector:

    def __init__(self, host, user, password, database):
        self.mariadb_connection = mariadb.connect(host='localhost', user= 'root', password='fadwa123', database='rs_101')
        self.df = None

    def query(self, query):
        self.df =  pd.read_sql(query, con=self.mariadb_connection)
        return self.df

    def columns_subset(self, *args):
        return self.df[list(args)]

    def _hashmap(self, df, column, column_key):
        values = sorted(df[column].unique())
        d = dict([(y, x) for x, y in enumerate(values)])
        df[column_key] = df[column].map(d)
        return df

    def hashmap_pk(self, item):
        self.df = self._hashmap(self.df, 'AuthenticationID', 'UserKey')
        self.df = self._hashmap(self.df, 'MOBI_MCC_ID', 'ItemKey')
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
        print ("Closing mariadb connection session...\n")
        self.mariadb_connection.close()

if __name__ == "__main__":
    test_class()
