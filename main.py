import numpy as np
import torch 
import warnings
import argparse
from dataloading import data_loading
from train import train
from model import NMF


def main(direc):
    table_pivot, train_loader, valid_loader\
    =data_loading(direc)
    train(table_pivot, train_loader, valid_loader, loss_fct ='cross_entropy')
    return


if __name__ == "__main__":
    warnings.simplefilter(action='ignore',category=FutureWarning)
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument("data_file", help='path to file name', type=str)
    args = parser.parse_args()
    main(args.data_file)
